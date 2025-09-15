from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from models.decode import grasp_pose_decode,_center_branch_decode_ori,_topk,_transpose_and_gather_feat,_gather_feat
from utils.utils import AverageMeter
import pdb
from numpy import array
from scipy.spatial.transform import Rotation as R
import cv2
import sys
import math
import yaml
sys.path.append('../EPro-PnP-6DoF_v2/lib/ops/pnp')
from levenberg_marquardt import LMSolver, RSLMSolver
from epropnp import EProPnP6DoF
from camera import PerspectiveCamera
from cost_fun import AdaptiveHuberPnPCost
import matplotlib.pyplot as plt
import numpy as np
#from cost_fun import AdaptiveHuberPnPCost
#from camera import PerspectiveCamera
sys.path.append('../EPro-PnP-6DoF_v2/lib')
print(sys.path)
from config import config
from models.monte_carlo_pose_loss import MonteCarloPoseLoss
from functools import partial
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap


def evaluate_pnp(x3d, x2d, w2d, pose, camera, cost_fun,
                 out_jacobian=False, out_residual=False, out_cost=False, **kwargs):
    x2d_proj, jac_cam = camera.project(
        x3d, pose, out_jac=(
            out_jacobian.view(x2d.shape[:-1] + (2, out_jacobian.size(-1))
                              ) if isinstance(out_jacobian, torch.Tensor)
            else out_jacobian), **kwargs)

    residual, cost, jacobian = cost_fun.compute(
        x2d_proj, x2d, w2d, jac_cam=jac_cam,
        out_residual=out_residual,
        out_cost=out_cost,
        out_jacobian=out_jacobian)

    return residual, cost, jacobian




class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)

    return outputs[-1], loss, loss_stats

 
class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader,cfg):
    return self.run_epoch('val', epoch, data_loader,cfg)

  def train(self, epoch, data_loader,cfg):
    return self.run_epoch('train', epoch, data_loader, cfg)
  



  def run_epoch(self, phase, epoch, data_loader,cfg):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    # train_num * 5 / batch_size = 66
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    # iter_id 0-66 
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)
      
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)    


      
      output, loss, loss_stats = model_with_loss(batch)
      model = self.model_with_loss
      l1_regularization = torch.tensor(0.0, device=model.parameters().__next__().device)

      for param in model.parameters():
          l1_regularization += torch.norm(param, p=1)

      loss_mc_all = 0.0
      loss_mc_all = compute_loss(output, self, cfg, loss_stats, batch)
      loss = loss_mc_all * float(self.opt.loss_mc_weight)  + loss
      loss = loss + l1_regularization * float(self.opt.L1_weight)

      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        # for name, param in model_with_loss.named_parameters():
        #   if "kpts_offset" in name:
        #     print(name, param.grad)
        # exit()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats, loss_mc_all
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results



def get_config_path():
    return '../EPro-PnP-6DoF_v2/tools/exps_cfg/epropnp_v2_cdpn_init.yaml'
def load_config():
    config_path = get_config_path()
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg




class Config:
    def __init__(self, config_path):
        self.config_path = config_path

    def parse(self):
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return config_data


def convert_pose(pose_init):
    if pose_init.dim() == 4:
       batchsize, num_poses, _, _ = pose_init.size()
    else:
      pose_init = pose_init.unsqueeze(1)
      batchsize, num_poses, _, _ = pose_init.size()

    poses_7d = torch.zeros((batchsize, num_poses, 7))
    for i in range(batchsize):
        for j in range(num_poses):
            translation = pose_init[i, j, :3, 3]
            rotation_matrix = pose_init[i, j, :3, :3].cpu().numpy()
            rotation = R.from_matrix(rotation_matrix)
            quaternion = torch.from_numpy(rotation.as_quat()).float()
            poses_7d[i, j, :3] = translation
            poses_7d[i, j, 3:] = quaternion

    return poses_7d



def compute_loss(output, self, cfg, loss_stats, batch):
    loss_mc_all = torch.tensor(1.0, requires_grad=True)
    bs = output['hm'].size(0)
    device = torch.device("cuda:0")
    torch.autograd.set_detect_anomaly(True)
    dets = grasp_pose_decode(
        self.opt,
        output['hm'], output['w'], output['kpts_center_offset'], 
        scales=None,
        reg=output['reg'], hm_kpts=None, kpts_offset=None, K=128
    )
    meta = {'c': torch.tensor([320., 240.], dtype=torch.float32).to(device), 
            's': torch.tensor([672., 512.], dtype=torch.float32).to(device), 
            'out_height': 128, 'out_width': 168, 'img_idx': 0}

    det_list = []
    for j in range(bs):
        dets_pro= dets[j]
        new_det = dets_pro.unsqueeze(dim=0)
        det = self.post_process(new_det, meta, scale=1)
        det_list.append(det)
    dets_tensor = torch.stack(det_list)
    kpts_2d_pred = dets_tensor[:, :, 2:10].reshape(bs, 128, 4, 2)
    #kpts_2d_pred = kpts_2d_pred.permute(0, 3, 1, 2)
    x2d = kpts_2d_pred.flatten(1, 2) 
    reshaped_tensor = output['w2d'].permute(0, 2, 3, 1)
    flattened_tensor = reshaped_tensor.reshape((bs, 1024, 2))
    processed_tensors = []

    for i in range(bs):
        selected_tensor = flattened_tensor[i]
        summed_tensor = selected_tensor.sum(dim=-1)
        sorted_indices = torch.argsort(summed_tensor)
        selected_sorted_tensor = selected_tensor[sorted_indices[:512]]
        processed_tensors.append(selected_sorted_tensor)
    w2d_selected = torch.stack(processed_tensors)
    w2d = w2d_selected.mean(dim=-1, keepdim=True)
    # w2d = (w2d - w2d.mean()) / (w2d.std() + 1e-8)  
    if torch.isnan(w2d).any() or torch.isinf(w2d).any() or (w2d < 0).any():
        w2d = torch.where(
            torch.isnan(w2d) | torch.isinf(w2d) | (w2d < 0),
            torch.full_like(w2d, 1e-6),
            w2d
        )
    camera_intrinsic_matrix = np.array([[616.36529541,   0.        , 310.25881958],
                                        [  0.        , 616.20294189, 236.59980774],
                                        [  0.        ,   0.        ,   1.        ]], dtype=np.float32)
    cam_intrinsic = torch.from_numpy(camera_intrinsic_matrix).cuda(cfg.pytorch.gpu)

    wh_begin = torch.tensor([274.5000,170.5000])
    wh_begin = wh_begin.unsqueeze(0).repeat(bs, 1)
    wh_unit = torch.tensor([3.0469])
    wh_unit = wh_unit.repeat(bs)
    allowed_border = 30 * wh_unit
    lb = (wh_begin - allowed_border[:, None]).to(device)
    ub = (wh_begin + (cfg.dataiter.out_res - 1) * wh_unit[:, None] + 
          allowed_border[:, None]).to(device)
    camera = PerspectiveCamera(
        cam_mats=cam_intrinsic[None].expand(bs, -1, -1),
        z_min=0.01, lb=lb, ub=ub)

    # EProPnP
    epropnp = EProPnP6DoF(
        mc_samples=512,
        num_iter=4,
        solver=LMSolver(
            dof=6,
            num_iter=5,
            init_solver=RSLMSolver(
                dof=6,
                num_points=4,
                num_proposals=4,
                num_iter=3
            )
        )
    ).cuda(cfg.pytorch.gpu)
    num_proposals = epropnp.solver.init_solver.num_proposals
    num_points = epropnp.solver.init_solver.num_points
    
    cost_fun = AdaptiveHuberPnPCost(relative_delta=0.1)
    cost_fun.set_param(x2d, w2d)
    target_num = batch['grasp_pose'].size(1)  
    for target_id in range(target_num):
      num = 0
      while num < 30:
        # Monte Carlo forward
        pn = x2d.size()[1]
        mean_weight = w2d.mean(dim=-1).reshape(1, bs, pn).expand(num_proposals, -1, -1)
        inds = torch.multinomial(
            mean_weight.reshape(-1, pn), num_points    
        ).reshape(num_proposals, bs, num_points)   
        tensor_kwargs = dict(dtype=x2d.dtype, device=x2d.device)
        points_2d = x2d.reshape(-1,2)[inds].reshape(num_proposals*bs, num_points, 2)
        points_w2d = w2d.reshape(-1,2)[inds].reshape(num_proposals*bs, num_points, 2)
        
        # points_w2d = torch.gather(w2d, 1, inds.unsqueeze(-1).expand(-1, -1, w2d.size(-1)))
        # points_2d = torch.gather(x2d, 1, inds.unsqueeze(-1).expand(-1, -1, x2d.size(-1)))
        points_3d = torch.tensor([[0, 0, 0.05],
                                  [0, 0, -0.05],
                                  [-0.1, 0, 0.05],
                                  [-0.1, 0, -0.05]], dtype=torch.float32).to(device)
        points_3d = points_3d.unsqueeze(0).repeat(num_proposals*bs, 1, 1)
        pose_init = x2d.new_empty(num_proposals, bs, 7)
        pose_init[..., :3] = epropnp.solver.init_solver.center_based_init(points_2d[0], points_3d[0], camera)
        pose_init_temp = torch.randn((num_proposals, bs, 4), dtype=x2d.dtype, device=x2d.device)  
        q_norm = pose_init_temp.norm(dim=-1)  
        pose_temp = pose_init_temp / q_norm.unsqueeze(-1)    
        pose_init[..., 3:] = pose_temp
        mask = (q_norm < epropnp.solver.init_solver.eps).flatten()
        pose_init.view(-1, 7)[mask, 3:] = x2d.new_tensor([1, 0, 0, 0])

        # pose----lm_iter
        jac = torch.empty((bs*num_proposals, num_points* 2, epropnp.solver.init_solver.dof), **tensor_kwargs)
        residual = torch.empty((bs*num_proposals, num_points* 2), **tensor_kwargs)
        cost = torch.empty((bs*num_proposals,), **tensor_kwargs)
        camera_expand = camera.shallow_copy()
        camera_expand.repeat_(num_proposals)
        cost_fun_expand = cost_fun.shallow_copy()
        cost_fun_expand.repeat_(num_proposals)

        evaluate_fun = partial(
            evaluate_pnp,
            x3d=points_3d, x2d=points_2d, w2d=points_w2d, camera=camera_expand, cost_fun=cost_fun_expand,
            clip_jac=True
        )
        pose_init = pose_init.reshape(num_proposals*bs,7)
        evaluate_fun(pose=pose_init, out_jacobian=jac, out_residual=residual, out_cost=cost
                    ,camera=camera_expand, cost_fun=cost_fun_expand)
        jac_new = torch.empty_like(jac)
        residual_new = torch.empty_like(residual)
        cost_new = torch.empty_like(cost)
        radius = x2d.new_full((bs*num_proposals,), epropnp.solver.initial_trust_region_radius)
        decrease_factor = x2d.new_full((bs*num_proposals,), 2.0)
        step_is_successful = x2d.new_zeros((bs*num_proposals,), dtype=torch.bool)
        i = 0
        while i < epropnp.solver.num_iter:
            epropnp.solver._lm_iter(
                pose_init,
                jac, residual, cost,
                jac_new, residual_new, cost_new,
                step_is_successful, radius, decrease_factor,
                evaluate_fun, camera)
            i += 1
        pose = pose_init.reshape(num_proposals, bs, pose_init.size(-1)) 
        cost_init = evaluate_fun(pose=pose_init, out_cost=True, out_jacobian=jac, out_residual=residual
                                ,camera=camera_expand, cost_fun=cost_fun_expand)[1]    
        cost_view = cost.view(num_proposals,bs)
        points_w2d_view = points_w2d.view(num_proposals,bs,num_points,2)
        points_2d_view = points_2d.view(num_proposals,bs,num_points,2)
        min_cost, min_cost_ind = cost_view.min(dim=0)       
        pose_opt = pose[min_cost_ind, torch.arange(bs, device=pose.device)]
        points_w2d_opt = points_w2d_view[min_cost_ind, torch.arange(bs, device=pose.device)]
        points_2d_opt = points_2d_view[min_cost_ind, torch.arange(bs, device=pose.device)]
        points_3d_opt = points_3d[:bs]
        # compare
        pose_gt_init = batch['grasp_pose'][:,target_id,num,:] # batchsize, target, 80, 4, 4
        pose_gt = convert_pose(pose_gt_init)  # 4x4 -- 7
        pose_gt = pose_gt.squeeze(1).to(device) 




        # LM solver loop
        jac = torch.empty((bs, 4 * 2, epropnp.solver.dof), **tensor_kwargs, requires_grad=True)
        residual = torch.empty((bs, 4 * 2), **tensor_kwargs, requires_grad=True)
        cost = torch.empty((bs,), **tensor_kwargs, requires_grad=True)
        evaluate_fun = partial(
            evaluate_pnp,
            x3d=points_3d_opt, x2d=points_2d_opt, w2d=points_w2d_opt, camera=camera, cost_fun=cost_fun,
            clip_jac=True
        )
        _, cost, _ = evaluate_fun(pose=pose_opt, x2d=points_2d_opt, x3d=points_3d_opt, w2d=points_w2d_opt,
                                  out_jacobian=jac, out_residual=residual, out_cost=cost)
        jac_new = torch.empty_like(jac)
        residual_new = torch.empty_like(residual)
        cost_new = torch.empty_like(cost)
        radius = x2d.new_full((bs,), epropnp.solver.initial_trust_region_radius)
        decrease_factor = x2d.new_full((bs,), 2.0)
        step_is_successful = x2d.new_zeros((bs,), dtype=torch.bool)
        i = 0
        while i < epropnp.solver.num_iter:
            epropnp.solver._lm_iter(
                pose_opt,
                jac, residual, cost,
                jac_new, residual_new, cost_new,
                step_is_successful, radius, decrease_factor,
                evaluate_fun, camera)
            i += 1

        new_jac = jac_new.clone()
        new_jac[~step_is_successful] = jac[~step_is_successful]
        jac = new_jac
        jtj = jac.transpose(-1, -2) @ jac
        diagonal = torch.diagonal(jtj, dim1=-2, dim2=-1)  # (num_obj, 4 or 6)
        diagonal += epropnp.solver.init_solver.eps  # add to jtj
        diagonal += 1e-4
        # jtj_det = torch.det(jtj)
        # if torch.all(jtj_det > 1e-6):  
        #     pose_cov = torch.inverse(jtj)
        # else:
        #     print("Matrix is singular or near-singular, using pseudoinverse instead. line473")
        #     pose_cov = torch.linalg.pinv(jtj)

        try:
            pose_cov = torch.inverse(jtj)
        except torch.linalg.LinAlgError as e:
            pose_cov = torch.pinverse(jtj)


        pose_samples = points_3d.new_empty((epropnp.num_iter, epropnp.iter_samples) + pose_opt.size())
        logprobs = points_3d.new_empty((epropnp.num_iter, epropnp.num_iter, epropnp.iter_samples, bs))
        cost_pred = points_3d.new_empty((epropnp.num_iter, epropnp.iter_samples, bs))
        distr_params = epropnp.allocate_buffer(bs, dtype=points_3d.dtype, device=points_3d.device)
        with torch.no_grad():
            epropnp.initial_fit(pose_opt, pose_cov, camera, *distr_params)  
        for i in range(epropnp.num_iter):
            new_trans_distr, new_rot_distr = epropnp.gen_new_distr(i, *distr_params)
            pose_samples[i, :, :, :3] = new_trans_distr.sample((epropnp.iter_samples, ))
            pose_samples[i, :, :, 3:] = new_rot_distr.sample((epropnp.iter_samples, ))

            cost_pred[i] = evaluate_fun(pose=pose_samples[i],out_cost=True)[1]

            # (i + 1, iter_sample, num_obj)
            # all samples (i + 1, iter_sample, num_obj) on new distr (num_obj, ) (4,4,128,3)
            logprobs[i, :i + 1] = new_trans_distr.log_prob(pose_samples[:i + 1, :, :, :3]) \
                                  + new_rot_distr.log_prob(pose_samples[:i + 1, :, :, 3:]).flatten(2)
            if i > 0:
                old_trans_distr, old_rot_distr = epropnp.gen_old_distr(i, *distr_params)
                # (i, iter_sample, num_obj)
                # new samples (iter_sample, num_obj) on old distr (i, 1, num_obj)
                logprobs[:i, i] = old_trans_distr.log_prob(pose_samples[i, :, :, :3]) \
                                  + old_rot_distr.log_prob(pose_samples[i, :, :, 3:]).flatten(2)
            # (i + 1, i + 1, iter_sample, num_obj) -> (i + 1, iter_sample, num_obj)
            mix_logprobs = torch.logsumexp(logprobs[:i + 1, :i + 1], dim=0) - math.log(i + 1)

            # (i + 1, iter_sample, num_obj)
            pose_sample_logweights = -cost_pred[:i + 1] - mix_logprobs

            if i == epropnp.num_iter - 1:
                break  # break at last iter      cost_gt = cost_gt + 1.0  

            with torch.no_grad():
                epropnp.estimate_params(
                    i,
                    pose_samples[:i + 1].reshape(((i + 1) * epropnp.iter_samples, ) + pose_opt.size()),
                    pose_sample_logweights.reshape((i + 1) * epropnp.iter_samples, bs),
                    *distr_params)

        pose_samples = pose_samples.reshape((epropnp.mc_samples, ) + pose_opt.size())      # 512,3,7
        pose_sample_logweights = pose_sample_logweights.reshape(epropnp.mc_samples, bs)       # (4,128,bs)------512,3
        scale = torch.tensor(1.0) 
        loss_monte = MonteCarloPoseLoss(init_norm_factor=1.0, momentum=0.01)
        cost_gt = evaluate_fun(pose=pose_gt, out_cost=True)[1]
        cost_gt = (cost_gt - cost_gt.mean()) / (cost_gt.std() + 1e-8)  
        loss_mc = loss_monte(pose_sample_logweights, cost_gt, scale.detach().mean())
        loss_stats['loss_mc_all'] = loss_stats['loss_mc_all'] + loss_mc
        num += 1
    loss_mc_all =  loss_stats['loss_mc_all']
    return loss_mc_all
