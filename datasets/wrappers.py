import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from datasets import register
from utils import make_coord, to_pixel_samples

N = 5
def round_to(v:int,to:int): return to*(v//to)

@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
      self.dataset = dataset
      self.inp_size = inp_size
      self.augment = augment
      self.sample_q = sample_q

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
      img_lr, img_hr = self.dataset[idx]

      s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
      if self.inp_size is None:
        h_lr, w_lr = img_lr.shape[-2:]
        crop_hr = img_hr = img_hr[:, :round_to(h_lr * s,N), :round_to(w_lr * s, N)]
        crop_lr = img_lr
      else:
        w_lr = self.inp_size
        x0 = random.randint(0, img_lr.shape[-2] - w_lr)
        y0 = random.randint(0, img_lr.shape[-1] - w_lr)
        crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
        w_hr = w_lr * s
        x1 = x0 * s
        y1 = y0 * s
        crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

      if self.augment:
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
          if hflip: x = x.flip(-2)
          if vflip: x = x.flip(-1)
          if dflip: x = x.transpose(-2, -1)
          return x

        crop_lr = augment(crop_lr)
        crop_hr = augment(crop_hr)

      hr_coord = make_coord(crop_hr.shape[-2]//N,crop_hr.shape[-1]//N)
      hr_rgb = crop_hr.flatten(1).t()
      #hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

      if self.sample_q is not None:
          sample_lst = np.random.choice(
              len(hr_coord), self.sample_q, replace=False)
          hr_coord = hr_coord[sample_lst]
          hr_rgb = hr_rgb[sample_lst]

      cell = torch.ones_like(hr_coord)
      cell[:, 0] *= 2 / (crop_hr.shape[-2] / N)
      cell[:, 1] *= 2 / (crop_hr.shape[-1] / N)

      return {
        'inp': crop_lr,
        'coord': hr_coord,
        'cell': cell,
        'gt': hr_rgb,
        'height': crop_hr.shape[-2],
        'width': crop_hr.shape[-1],
      }


def resize_fn(img, size):
    return transforms.ToTensor()(transforms.Resize(size, Image.BICUBIC)(transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(
      self, dataset, inp_size=None, scale_min=1, scale_max=None, augment=False, sample_q=None
    ):
      self.dataset = dataset
      self.inp_size = inp_size
      self.scale_min = scale_min
      if scale_max is None: scale_max = scale_min
      self.scale_max = scale_max
      self.augment = augment
      self.sample_q = sample_q

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
      img = self.dataset[idx]
      s = random.uniform(self.scale_min, self.scale_max)

      if self.inp_size is None:
          h_lr = math.floor(img.shape[-2] / s + 1e-9)
          w_lr = math.floor(img.shape[-1] / s + 1e-9)
          img = img[:, :round_to(round(h_lr * s), N), :round_to(round(w_lr * s), N)] # assume round int
          img_down = resize_fn(img, (h_lr, w_lr))
          crop_lr, crop_hr = img_down, img
      else:
          w_lr = self.inp_size
          w_hr = round_to(round(w_lr * s), N)
          x0 = random.randint(0, img.shape[-2] - w_hr)
          y0 = random.randint(0, img.shape[-1] - w_hr)
          crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
          crop_lr = resize_fn(crop_hr, w_lr)

      if self.augment:
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
          if hflip: x = x.flip(-2)
          if vflip: x = x.flip(-1)
          if dflip: x = x.transpose(-2, -1)
          return x

        crop_lr = augment(crop_lr)
        crop_hr = augment(crop_hr)

      hr_coord = make_coord(crop_hr.shape[-2]//N,crop_hr.shape[-1]//N)
      # add in a bit of noise to try to normalize it
      hr_coord = hr_coord + torch.randn_like(hr_coord) * 1e-3

      hr_rgb = crop_hr.reshape(3, crop_hr.shape[1]//N, N, crop_hr.shape[2]//N, N)\
        .permute(0, 2, 4, 1, 3)\
        .flatten(0, 2)\
        .flatten(1)\
        .t()

      #hr_coord = make_coord(crop_hr.shape[-2],crop_hr.shape[-1],flatten=False).movedim(-1,0)
      #hr_rgb = F.unfold(crop_hr[None], N).squeeze(0).t()
      #kernel_cutoff = kc = (N-1)//2
      #hr_coord = hr_coord[:, kc:-kc,kc:-kc].flatten(-2).t()

      #hr_coord, hr_rgb = to_pixel_samples(img_hr)

      if self.sample_q is not None:
        sample_lst = np.random.choice(hr_coord.shape[0], self.sample_q, replace=False)
        hr_coord = hr_coord[sample_lst]
        hr_rgb = hr_rgb[sample_lst]

      cell = torch.ones_like(hr_coord)
      cell[:, 0] *= 2 / (crop_hr.shape[-2] / N)
      cell[:, 1] *= 2 / (crop_hr.shape[-1] / N)

      return {
        'inp': crop_lr,
        'coord': hr_coord,
        'cell': cell,
        'gt': hr_rgb
      }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
      img_lr, img_hr = self.dataset[idx]
      p = idx / (len(self.dataset) - 1)
      w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
      img_hr = resize_fn(img_hr, w_hr)

      if self.augment:
        if random.random() < 0.5:
          img_lr = img_lr.flip(-1)
          img_hr = img_hr.flip(-1)

      if self.gt_resize is not None: img_hr = resize_fn(img_hr, self.gt_resize)

      hr_coord, hr_rgb = to_pixel_samples(img_hr)

      if self.sample_q is not None:
        sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
        hr_coord = hr_coord[sample_lst]
        hr_rgb = hr_rgb[sample_lst]

      cell = torch.ones_like(hr_coord)
      cell[:, 0] *= 2 / img_hr.shape[-2]
      cell[:, 1] *= 2 / img_hr.shape[-1]

      return {
        'inp': img_lr,
        'coord': hr_coord,
        'cell': cell,
        'gt': hr_rgb,
      }
