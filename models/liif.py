import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('liif')
class LIIF(nn.Module):
    def __init__(
      self,
      encoder_spec,
      imnet_spec,
      local_ensemble=True,
      feat_unfold=True,
      cell_decode=True,
    ):
      super().__init__()
      self.local_ensemble = local_ensemble
      self.feat_unfold = feat_unfold
      self.cell_decode = cell_decode

      self.encoder = models.make(encoder_spec)
      # Just a standard MLP usually, altho I added skip connections for fun.
      self.imnet = models.make(imnet_spec)

    def gen_feat(self, inp):
      self.feat = self.encoder(inp)
      return self.feat

    def query_rgb(self, coord, cell=None):
      feat = self.feat

      # Concatenate items in a 3x3 window
      if self.feat_unfold:
        f0, f1, f2, f3 = feat.shape
        feat = F.unfold(feat, 3, padding=1).reshape(f0, f1 * 9, f2, f3)

      vx_lst, vy_lst, eps_shift = [0], [0], 0
      if self.local_ensemble:
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

      hw = torch.tensor(feat.shape[-2:], device=coord.device, dtype=torch.float)

      # field radius (global: [-1, 1])
      rx, ry = [2/r/2 for r in feat.shape[2:]]

      feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda()\
        .permute(2, 0, 1)[None]\
        .expand(feat.shape[0], 2, *feat.shape[-2:])

      if self.cell_decode:
        rel_cell = cell.clone()
        rel_cell[..., :] *= hw

      preds = []
      areas = []
      for vx in vx_lst:
        for vy in vy_lst:
          coord_ = coord.clone()
          coord_[:, :, 0] += vx * rx + eps_shift
          coord_[:, :, 1] += vy * ry + eps_shift
          coord_.clamp_(min=-1 + 1e-6, max=1 - 1e-6)
          sample_coord = coord_.flip(-1).unsqueeze(1)
          q_feat = F.grid_sample(
            feat,
            sample_coord,
            mode='nearest',
            align_corners=False,
          )[:, :, 0, :].permute(0, 2, 1)
          q_coord = F.grid_sample(
            feat_coord,
            sample_coord,
            mode='nearest',
            align_corners=False,
          )[:, :, 0, :].permute(0, 2, 1)
          rel_coord = coord - q_coord
          rel_coord[..., :] *= hw
          inp = torch.cat([q_feat, rel_coord], dim=-1)

          if self.cell_decode: inp = torch.cat([inp, rel_cell], dim=-1)

          preds.append(self.imnet(inp))
          areas.append(torch.abs(rel_coord.prod(dim=-1, keepdim=True)) + 1e-9)

      areas = torch.stack(areas, dim=0)
      areas = areas/areas.sum(dim=0, keepdim=True)
      if self.local_ensemble: areas[[0,1,2,3]] = areas[[3,2,1,0]]
      preds = torch.stack(preds, dim=0)
      return (preds * areas).sum(dim=0)

    def query_rgb2(self, coord, cell):
      feat = self.feat

      f0, f1, f2, f3 = feat.shape
      feat = F.unfold(feat, 3, padding=1).reshape(f0, f1 * 9, f2, f3)

      offsets = torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float,device=coord.device)
      rad = torch.tensor([2/r/2 for r in feat.shape[2:]],device=coord.device,dtype=torch.float)

      feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda()\
        .permute(2, 0, 1).unsqueeze(0)\
        .expand(feat.shape[0], 2, *feat.shape[-2:])

      if self.cell_decode:
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]

      # TODO maybe this shouldn't be a long a batch dim but along another dimension?
      # where is the bug coming from?
      ensemble_coords = coord[None] + (offsets * rad[None])[:, None, None, :]

      ensemble_coords.clamp_(min=-1+1e-6, max=1-1e-6)
      sample_coords = ensemble_coords.reshape(-1, *coord.shape[1:]).flip(-1).unsqueeze(1)
      q_feat = F.grid_sample(
        feat.repeat_interleave(4, dim=0,output_size=sample_coords.shape[0]),
        sample_coords,
        mode='nearest',
        align_corners=False,
      )[:, :, 0, :].permute(0, 2, 1).reshape(4, *coord.shape[:-1], -1)
      q_coord = F.grid_sample(
        feat_coord.repeat_interleave(4, dim=0,output_size=sample_coords.shape[0]),
        sample_coords,
        mode='nearest',
        align_corners=False,
      )[:, :, 0, :].permute(0, 2, 1).reshape(4, *coord.shape)

      rel_coord = coord[None] - q_coord
      rel_coord[..., 0] *= feat.shape[-2]
      rel_coord[..., 1] *= feat.shape[-1]
      inp = torch.cat([q_feat, rel_coord], dim=-1)

      if self.cell_decode: inp = torch.cat([inp, rel_cell[None].expand_as(rel_coord)], dim=-1)

      pred = self.imnet(inp)
      areas = torch.abs(rel_coord.prod(dim=-1, keepdim=True)) + 1e-9

      tot_area = areas.sum(dim=0, keepdim=True)
      areas = areas/tot_area
      # Why does this exist?
      if self.local_ensemble:
        areas[[0,3]] = areas[[3,0]]
        areas[[1,2]] = areas[[2,1]]
      return (pred * areas).sum(dim=0)

    def query_rgb_bezier(self, coord, cell):
      ...
    def forward(self, inp, coord, cell):
      self.gen_feat(inp)
      return self.query_rgb(coord, cell)
