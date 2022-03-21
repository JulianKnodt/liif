# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register


def default_conv(in_channels, out_channels, kernel_size, bias=True):
  return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
  def __init__(
    self,
    conv,
    n_feats,
    kernel_size,
    bias=True,
    bn=False,
    act=nn.LeakyReLU(inplace=True),
    res_scale=1,
  ):
    super(ResBlock, self).__init__()
    m = []
    for i in range(2):
      m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
      if bn: m.append(nn.BatchNorm2d(n_feats))
      if i == 0: m.append(act)

    self.body = nn.Sequential(*m)
    self.res_scale = res_scale
  def forward(self, x): return x.add(self.body(x), alpha=self.res_scale)

url = {
  'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
  'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
  'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
  'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
  'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
  'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt',
}

class EDSR(nn.Module):
  def __init__(self, args, conv=default_conv):
    super(EDSR, self).__init__()
    self.args = args
    n_resblocks = args.n_resblocks
    n_feats = args.n_feats
    kernel_size = 3
    scale = args.scale[0]
    act = nn.LeakyReLU(inplace=True)
    url_name = f"r{n_resblocks}f{n_feats}x{scale}"
    self.url = url.get(url_name, None)

    # define head module
    m_head = [conv(args.n_colors, n_feats, kernel_size)]
    self.head = nn.Sequential(*m_head)

    # define body module
    m_body = [
      ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
      for _ in range(n_resblocks)
    ] + [conv(n_feats, n_feats, kernel_size)]

    self.body = nn.Sequential(*m_body)

    assert(args.no_upsampling)
    self.out_dim = n_feats

  def forward(self, x):
    x = self.head(x)
    return self.body(x).add_(x)

  def load_state_dict(self, state_dict, strict=True):
      own_state = self.state_dict()
      for name, param in state_dict.items():
        if name in own_state:
          if isinstance(param, nn.Parameter): param = param.data
          try: own_state[name].copy_(param)
          except Exception:
            if name.find('tail') == -1:
              raise RuntimeError('While copying the parameter named {}, '
                                 'whose dimensions in the model are {} and '
                                 'whose dimensions in the checkpoint are {}.'
                                 .format(name, own_state[name].size(), param.size()))
        elif strict:
          if name.find('tail') == -1: raise KeyError(f'unexpected key "{name}" in state_dict')


@register('edsr-baseline')
def make_edsr_baseline(
  n_resblocks=16,
  n_feats=63,
  res_scale=1,
  scale=2,
  no_upsampling=True,
  rgb_range=1
):
  args = Namespace()
  args.n_resblocks = n_resblocks
  args.n_feats = n_feats
  args.res_scale = res_scale

  args.scale = [scale]
  args.no_upsampling = no_upsampling

  args.rgb_range = rgb_range
  args.n_colors = 3
  return EDSR(args)


@register('edsr')
def make_edsr(
  n_resblocks=32,
  n_feats=256,
  res_scale=0.1,
  scale=2,
  no_upsampling=False,
  rgb_range=1
):
  args = Namespace()
  args.n_resblocks = n_resblocks
  args.n_feats = n_feats
  args.res_scale = res_scale

  args.scale = [scale]
  args.no_upsampling = no_upsampling

  args.rgb_range = rgb_range
  args.n_colors = 3
  return EDSR(args)
