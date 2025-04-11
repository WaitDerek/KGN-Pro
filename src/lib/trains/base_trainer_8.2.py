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
from config import config
from models.monte_carlo_pose_loss import MonteCarloPoseLoss
from functools import partial


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

  def run_epoch_notuse(self, phase, epoch, data_loader,cfg):
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
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()

    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)
      
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)    
      output, loss, loss_stats = model_with_loss(batch)


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
      del output, loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader,cfg):
    return self.run_epoch('val', epoch, data_loader,cfg)

  def train(self, epoch, data_loader,cfg):
    return self.run_epoch('val', epoch, data_loader, cfg)
  



  def run_epoch(self, phase, epoch, data_loader,cfg):
    #model_with_loss 
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

    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)
      
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)    
      # 调用ModelWithLoss 的forward 传入的batch是data_loader
      # output 
      # loss_stats: loss, hm_loss, w_loss, kpts_center_loss, reg_loss 
      output, loss, loss_stats = model_with_loss(batch)
      loss_mc_all = 0.0
      bs = output['hm'].size(0)
      device = torch.device("cuda:0") 
      # 6,100,13 : centers, kps, w, scores, clses, 
      dets = grasp_pose_decode(
        self.opt,
        output['hm'], output['w'], output['kpts_center_offset'], 
        scales = None,
        reg=output['reg'], hm_kpts=None, kpts_offset=None, K=128
        )
      # centers, kps, w, scores, clses, scls = _center_branch_decode_ori(
      #     opt, output['hm'], output['w'], output['kpts_center_offset'], output['reg'], K=100, scales=None)
      meta = {'c': torch.tensor([320., 240.], dtype=torch.float32).to(device), 
        's': torch.tensor([672., 512.], dtype=torch.float32).to(device), 
        'out_height': 128, 'out_width': 168, 'img_idx': 0}
      # x2d(6,512,2)  w2d(6,512,2)
      det_list = []
      for j in range(dets.size(0)):
        dets_cbr = dets[j]
        new_det = dets_cbr.unsqueeze(dim=0)
        det = self.post_process(new_det, meta, scale = 1)
        #print(tensor.size())
        det_list.append(det)
      dets_tensor = torch.stack(det_list)
      breakpoint()
      kpts_2d_pred = dets_tensor[:,:, 2:10].reshape(bs, 128, 4, 2)
      x2d = kpts_2d_pred.reshape(bs,512,2)
      reshaped_tensor = output['w2d'].cpu().permute(0, 2, 3, 1)
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
      if torch.isnan(w2d).any() or torch.isinf(w2d).any() or (w2d < 0).any():
        w2d = torch.where(
            torch.isnan(w2d) | torch.isinf(w2d) | (w2d < 0),
            torch.full_like(w2d, 1e-6),
            w2d
        )
 
      camera_intrinsic_matrix = np.array([[616.36529541,   0.        , 310.25881958],
                                    [  0.        , 616.20294189, 236.59980774],
                                    [  0.        ,   0.        ,   1.        ]],dtype=np.float32)
      #cam_intrinsic_np = cfg.dataset.camera_matrix.astype(np.float32)
      #cam_intrinsic = torch.from_numpy(cam_intrinsic_np).cuda(cfg.pytorch.gpu)
      cam_intrinsic = torch.from_numpy(camera_intrinsic_matrix).cuda(cfg.pytorch.gpu)

      wh_begin = torch.tensor([274.5000,170.5000])
      wh_unit = torch.tensor([3.0469])
      allowed_border = 30 * wh_unit
      lb = (wh_begin - allowed_border[:, None]).to(device)
      ub = (wh_begin + (cfg.dataiter.out_res - 1) * wh_unit[:, None] + 
            allowed_border[:, None]).to(device)
      #
      camera = PerspectiveCamera(
        cam_mats=cam_intrinsic[None].expand(bs, -1, -1),    # intrinsic
        z_min=0.01, lb = lb,
        ub = ub)

      camera_intrinsic_matrix = np.array([[616.36529541,   0.        , 310.25881958],
                                    [  0.        , 616.20294189, 236.59980774],
                                    [  0.        ,   0.        ,   1.        ]])
      dist_coeffs = np.zeros((4, 1))
      r = np.array([], dtype=np.float32)

      epropnp = EProPnP6DoF(
        mc_samples=512,
        num_iter=4,   
        solver=LMSolver(
            dof=6,
            num_iter=5,
            init_solver=RSLMSolver(
                dof=6,
                num_points=16,
                num_proposals=4,
                num_iter=3))).cuda(cfg.pytorch.gpu)


      poses_m = torch.zeros((bs, 80, 7))
      poses_list = [torch.empty((0, 7)) for _ in range(bs)]
      #inds = torch.multinomial(w2d.reshape(6,512), 4).reshape(bs,4) 
      for m in range(80):
          pnp_algorithm = 6
          # x3d x2d w2d
          points_2d_list=[]
          points_3d_list=[]
          points_w2d_list=[]
          obj_3d_points = torch.tensor([[0, 0, 0.05],
                              [0, 0, -0.05],
                              [-0.1, 0, 0.05],
                              [-0.1, 0, -0.05]], dtype=torch.float32)
          obj_3d_points_np = obj_3d_points.cpu().numpy()
          inds = torch.multinomial(w2d.reshape(bs,512), 4).reshape(bs,4) 
          for batchsize in range(bs):
            points_2d_list.append((x2d[batchsize][inds[batchsize]]))
            points_3d_list.append(obj_3d_points)
            points_w2d_list.append((w2d[batchsize][inds[batchsize]]))

          points_2d = torch.stack(points_2d_list, axis=0)
          points_w2d = torch.stack(points_w2d_list, axis=0)
          points_3d = torch.stack(points_3d_list, axis=0)
          points_2d_np = points_2d.detach().cpu().numpy()
          # TODO: 
          for n in range(bs):
            ret, rvec, tvec, reprojectionError = cv2.solvePnPGeneric(
                obj_3d_points_np,
                points_2d_np[n],
                camera_intrinsic_matrix,
                dist_coeffs,
                flags=pnp_algorithm,
                reprojectionError=r
            )
            if ret:
              rvec = np.array(rvec[0])
              tvec = np.array(tvec[0]) 
              rvec = rvec.flatten()
              rotation = R.from_rotvec(rvec)
              quaternion = rotation.as_quat()
              pose = np.concatenate((tvec, quaternion), axis=None)
              pose = torch.from_numpy(pose).float()
              poses_m[n,m]=pose
              poses_list[n] = torch.cat((poses_list[n], pose.unsqueeze(0)), dim=0)
          poses_list_tensor = torch.stack(poses_list)

          cost_fun = AdaptiveHuberPnPCost(
            relative_delta=0.1)
          cost_fun.set_param(x2d, w2d)

          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          pose = torch.FloatTensor(pose).to(device)
          points_3d = torch.FloatTensor(points_3d).to(device)
          points_2d = torch.FloatTensor(points_2d).to(device)
          points_w2d = torch.FloatTensor(points_w2d).to(device)
          evaluate_fun = partial(
            evaluate_pnp,
            x3d=points_3d, x2d=points_2d, w2d=points_w2d, camera=camera, cost_fun=cost_fun, out_cost=True)
          pose_gt = convert_pose(batch['grasp_pose'])
          num_gt = 1
          pose_init = pose_gt[:,num_gt].to(device)
          _, cost_init, _ = evaluate_fun(pose=pose_init)


          num_obj = points_3d.size(0)

          tensor_kwargs = dict(dtype=x2d.dtype, device=x2d.device)
          jac = torch.empty((bs, 8, 6), **tensor_kwargs).to(device)
          residual = torch.empty((bs, 8), **tensor_kwargs).to(device)
          cost = torch.empty((bs,), **tensor_kwargs).to(device)
          distr_params = epropnp.allocate_buffer(num_obj, dtype=points_3d.dtype, device=points_3d.device)
          evaluate_fun(x3d=points_3d,x2d=points_2d,w2d=points_w2d,pose=pose,camera=camera,cost_fun=cost_fun
            ,out_jacobian=jac, out_residual=residual,out_cost=cost)
          jtj = jac.transpose(-1, -2) @ jac
          jtj = jtj + torch.eye(epropnp.solver.dof, device=jtj.device, dtype=jtj.dtype) * epropnp.solver.eps 
          # TODO: pose_cov
          pose_cov = torch.inverse(jtj).to(device)
          epsilon = 1e-6  # 
          identity_matrix = torch.eye(pose_cov.size(-1), device=pose_cov.device) * epsilon
          pose_cov += identity_matrix
          #pose_cov = poses_m[:,:6,:6]
          pose_opt = poses_m[:,1,:].to(device)
          pose_samples = points_3d.new_empty((epropnp.num_iter, epropnp.iter_samples) + pose_opt.size())
          logprobs = points_3d.new_empty((epropnp.num_iter, epropnp.num_iter, epropnp.iter_samples, num_obj))
          cost_pred = points_3d.new_empty((epropnp.num_iter, epropnp.iter_samples, num_obj))
          with torch.no_grad():
              epropnp.initial_fit(pose_opt, pose_cov, camera, *distr_params)  
          for i in range(epropnp.num_iter):
              new_trans_distr, new_rot_distr = epropnp.gen_new_distr(i, *distr_params)
              pose_samples[i, :, :, :3] = new_trans_distr.sample((epropnp.iter_samples, ))
              pose_samples[i, :, :, 3:] = new_rot_distr.sample((epropnp.iter_samples, ))

              cost_pred[i] = evaluate_fun(pose=pose_samples[i])[1]
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
                  break  # break at last iter
              with torch.no_grad():
                  epropnp.estimate_params(
                      i,
                      pose_samples[:i + 1].reshape(((i + 1) * epropnp.iter_samples, ) + pose_opt.size()),
                      pose_sample_logweights.reshape((i + 1) * epropnp.iter_samples, num_obj),
                      *distr_params)

          pose_samples = pose_samples.reshape((epropnp.mc_samples, ) + pose_opt.size())      # 512,3,7
          pose_sample_logweights = pose_sample_logweights.reshape(epropnp.mc_samples, num_obj)       # (4,128,bs)------512,3
          scale = torch.tensor(300.0) 
          loss_monte = MonteCarloPoseLoss(init_norm_factor=1.0, momentum=0.01)
          loss_mc = loss_monte(pose_sample_logweights, cost_init, scale.detach().mean())
          loss_stats['loss_mc_all'] += loss_mc.to(loss_stats['loss_mc_all'].device)




      # config_path = get_config_path()  
      # cfg = config().parse()
      loss = loss_stats['loss_mc_all']
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