from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pdb
import os
import random
import numpy as np
import torch
import torch.utils.data


import _init_paths

from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

import sys
sys.path.append('../EPro-PnP-6DoF_v2/lib')
from config import config
sys.path.append('../EPro-PnP-6DoF_v2/lib/datasets')
from lm import LM




def main(opt,cfg):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  # Dataset  PSGrasp GraspPoseDataset
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt.inp_channel)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  # DLASeg  ctdet_coco_dla_2x.pth  Adam  resume = false 
  # load pretrained model
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=2, 
      shuffle=False,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    # log_dict_train: loss, w_loss, kpts_center_loss, reg_loss, times
    log_dict_train, las = trainer.train(epoch, train_loader,cfg)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      print("\n")
      save_model(os.path.join(opt.save_dir, 'model_{}_{}_{}.pth'.format(mark, opt.loss_mc_weight, opt.L1_weight)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader, cfg)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best_{}_{}.pth'.format(opt.loss_mc_weight, opt.L1_weight)), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last_{}_{}.pth'.format(opt.loss_mc_weight, opt.L1_weight)), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}_{}_{}.pth'.format(epoch, opt.loss_mc_weight, opt.L1_weight)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()

  cfg=config().parse()

  main(opt,cfg)