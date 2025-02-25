import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
import torchvision as tv
from tqdm import tqdm
import datasets
import models
import utils


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
      feat = model.gen_feat(inp)
      n = coord.shape[1]
      ql = 0
      preds = []
      while ql < n:
        qr = min(ql + bsize, n)
        pred = model.query_rgb(feat, coord[:, ql:qr, :], cell[:, ql:qr, :])
        preds.append(pred)
        ql = qr
      pred = torch.cat(preds, dim=1)
    return pred

def batched_predict_with_feat(model, feat, coord, cell, bsize):
    with torch.no_grad():
      n = coord.shape[1]
      ql = 0
      preds = []
      while ql < n:
        qr = min(ql + bsize, n)
        pred = model.query_rgb(feat, coord[:, ql:qr, :], cell[:, ql:qr, :])
        preds.append(pred)
        ql = qr
      pred = torch.cat(preds, dim=1)
    return pred

N = 5
def eval_psnr(
  loader,
  model,
  data_norm=None,
  eval_type=None,
  eval_bsize=None,
  verbose=False,
  save_image=True,
):
  model = model.eval()

  if data_norm is None:
    data_norm = {
      'inp': {'sub': [0], 'div': [1]},
      'gt': {'sub': [0], 'div': [1]}
    }

  t = data_norm['inp']
  inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
  inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
  t = data_norm['gt']
  gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
  gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

  if eval_type is None: metric_fn = utils.calc_psnr
  elif eval_type.startswith('div2k'):
    scale = int(eval_type.split('-')[1])
    metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
  elif eval_type.startswith('benchmark'):
    scale = int(eval_type.split('-')[1])
    metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
  else: raise NotImplementedError

  val_res = utils.Averager()

  progress = tqdm(loader, leave=False, desc='val')
  for i, batch in enumerate(progress):
    for k, v in batch.items(): batch[k] = v.cuda()

    inp = (batch['inp'] - inp_sub) / inp_div
    if eval_bsize is None:
      with torch.no_grad():
        pred = model(inp, batch['coord'], batch['cell'])
    else:
      pred = batched_predict(model, inp, batch['coord'], batch['cell'], eval_bsize)
    pred = pred * gt_div + gt_sub
    pred.clamp_(min=0, max=1)

    gt = batch['gt']
    # reshape for shaving-eval
    if eval_type is not None:
      B = batch['inp'].shape[0]
      shape = [B, batch['height'].item(), batch['width'].item(), 3]
      gt = gt.reshape(*shape).permute(0, 3, 1, 2)
      pred = pred.reshape(B, shape[1]//N, shape[2]//N, 3, N, N)\
                 .permute(0, 1, 4, 2, 5, 3)
      pred = pred.reshape(*shape).permute(0, 3, 1, 2)

    res = metric_fn(pred, gt)
    val_res.add(res.item(), inp.shape[0])

    if save_image:
      error = (gt-pred)
      try:
        tv.utils.save_image(torch.cat([
          gt,
          pred,
          error.abs().sqrt(),
        ], dim=0).float(), f"outputs/test_{i:03}.png")
      except Exception as e:
        print()
        ...
        #print(f"Failed to save image with shape {pred.shape} for gt shape {gt.shape}: {e}")
    progress.set_postfix(PSNR=f"{val_res.item():.4f}")
  return val_res.item()



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', required=True)
  parser.add_argument('--model', required=True)
  parser.add_argument('--gpu', default='0')
  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  spec = config['test_dataset']
  dataset = datasets.make(spec['dataset'])
  dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
  loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=1, pin_memory=True)

  model_spec = torch.load(args.model)['model']
  model = models.make(model_spec, load_sd=True).cuda()
  print("[note]: loaded model")

  res = eval_psnr(loader, model,
    data_norm=config.get('data_norm'),
    eval_type=config.get('eval_type'),
    eval_bsize=config.get('eval_bsize'),
    verbose=True,
  )
  print(f"result: {res:.4f}")
if __name__ == '__main__':
  with torch.no_grad():
    main()
