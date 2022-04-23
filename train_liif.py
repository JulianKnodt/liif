""" Train for generating LIIF, from image to implicit representation.

  Config:
      train_dataset:
        dataset: $spec; wrapper: $spec; batch_size:
      val_dataset:
        dataset: $spec; wrapper: $spec; batch_size:
      (data_norm):
          inp: {sub: []; div: []}
          gt: {sub: []; div: []}
      (eval_type):
      (eval_bsize):

      model: $spec
      optimizer: $spec
      epoch_max:
      (multi_step_lr):
          milestones: []; gamma: 0.5
      (resume): *.pth

      (epoch_val): ; (epoch_save):
"""

import argparse
import os

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import random

import datasets
import models
import utils
from test import eval_psnr


def make_data_loader(spec, tag=''):
  if spec is None: return None

  dataset = datasets.make(spec['dataset'])
  dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

  loader = DataLoader(
    dataset, batch_size=spec['batch_size'],
    shuffle=(tag == 'train'),
    num_workers=1,
    pin_memory=True,
  )
  return loader

def prepare_training():
  if config.get('resume') is not None:
    print("[note]: resuming training")
    sv_file = torch.load(config['resume'])
    model = models.make(sv_file['model'], load_sd=True).cuda()
    optimizer = utils.make_optimizer(model.parameters(), sv_file['optimizer'], load_sd=True)
    epoch_start = sv_file['epoch'] + 1
  else:
    print("[note]: training starting fresh")
    model = models.make(config['model']).cuda()
    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    epoch_start = 1

  if config.get('multi_step_lr') is None: lr_scheduler = None
  else: lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
  for _ in range(epoch_start - 1): lr_scheduler.step()
  return model, optimizer, epoch_start, lr_scheduler

# Computes the difference of the fft of two images
def fft_loss(x, ref):
  got = torch.fft.rfft2(x, dim=(-3, -2), norm="ortho")
  exp = torch.fft.rfft2(ref, dim=(-3, -2), norm="ortho")
  return (got - exp).abs().mean()

def train(train_loader, model, opt):
  model.train()
  #loss_fn = fft_loss
  loss_fn = F.mse_loss
  train_loss = utils.MovingAverager()

  data_norm = config['data_norm']
  t = data_norm['inp']
  inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
  inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
  t = data_norm['gt']
  gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
  gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

  progress = tqdm(train_loader, leave=False, desc='train')
  opt_step = 1
  N = opt_step * 2
  for i, batch in enumerate(progress):
    for k, v in batch.items(): batch[k] = v.cuda()
    gt = (batch['gt'] - gt_sub) / gt_div
    inp = (batch['inp'] - inp_sub) / inp_div

    feat = model.gen_feat(inp)

    # TODO incorporate
    # https://openreview.net/pdf?id=WA39qkJvLi

    # train full model
    pred = model.query_rgb(feat, batch['coord'], batch['cell'])
    loss = loss_fn(pred, gt)
    loss.div(N).backward(retain_graph=True)
    train_loss.add(loss.item())

    total_loss = F.mse_loss(pred, gt).item()

    # train partial model (with zeroed out trailing features)
    new_feat = feat.clone()
    assert(feat.shape[1] % 3 == 0)
    new_feat[:, (random.randint(1, feat.shape[1]//3)*3):] = 0
    pred = model.query_rgb(new_feat, batch['coord'], batch['cell'])
    loss_fn(pred, gt).div(N).backward()

    # ---
    if (i+1) % opt_step == 0:
      opt.step()
      opt.zero_grad()

    progress.set_postfix(
      L=train_loss.item(),
      mse=total_loss,
      LR=f"{opt.param_groups[0]['lr']:.01e}"
    )
  return train_loss.item()


def main(config_, save_path):
    global config
    config = config_
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
      yaml.dump(config, f, sort_keys=False)

    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    print("[note]: Made loaders")
    if config.get('data_norm') is None:
        config['data_norm'] = { 'inp': {'sub': [0], 'div': [1]}, 'gt': {'sub': [0], 'div': [1]} }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    print("[note]: Loaded model and optimizer")

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1: model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
      t_epoch_start = timer.t()
      log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

      train_loss = train(train_loader, model, optimizer)
      if lr_scheduler is not None: lr_scheduler.step()

      log_info.append('train: loss={:.4f}'.format(train_loss))
      model_ = model_.module if n_gpus > 1 else model
      model_spec = config['model']
      model_spec['sd'] = model_.state_dict()
      optimizer_spec = config['optimizer']
      optimizer_spec['sd'] = optimizer.state_dict()
      sv_file = {
        'model': model_spec,
        'optimizer': optimizer_spec,
        'epoch': epoch
      }

      torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

      if (epoch_save is not None) and (epoch % epoch_save == 0):
        torch.save(sv_file, os.path.join(save_path, f'epoch-{epoch}.pth'))

      if (epoch_val is not None) and (epoch % epoch_val == 0):
        model_ = model
        if n_gpus > 1 and (config.get('eval_bsize') is not None): model_ = model.module
        val_res = eval_psnr(
          val_loader,
          model_,
          data_norm=config['data_norm'],
          eval_type=config.get('eval_type'),
          eval_bsize=config.get('eval_bsize')
        )

        log_info.append(f'val: psnr={val_res:.4f}')
        if val_res > max_val_v:
          max_val_v = val_res
          torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append(f'{t_epoch} {t_elapsed}/{t_all}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
      config = yaml.load(f, Loader=yaml.FullLoader)
      print('[note]: config loaded')

    save_name = args.name
    if save_name is None: save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None: save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
