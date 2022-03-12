import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

def fat_tanh(v, eps=1e-2): return v.tanh() * (1 + eps)

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

    def query_rgb(self, coord, cell):
      feat = self.feat

      assert(self.feat_unfold)
      f0, f1, f2, f3 = feat.shape
      feat = F.unfold(feat, 3, padding=1).reshape(f0, f1 * 9, f2, f3)

      assert(self.local_ensemble)
      offsets = torch.tensor([
        [-1,-1],[-1, 1],[1,-1],[1,1],
      ], dtype=torch.float, device=coord.device)
      rad = torch.tensor([2/r/2 for r in feat.shape[2:]],device=coord.device,dtype=torch.float)

      feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda()\
        .permute(2, 0, 1)[None]\
        .expand(feat.shape[0], 2, *feat.shape[-2:])

      if self.cell_decode:
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]

      ensemble_coords = coord[None] + (offsets * rad[None])[:, None, None, :] + 1e-8

      ensemble_coords = ensemble_coords.clamp_(min=-1+1e-10, max=1-1e-10)
      sample_coords = ensemble_coords.permute(1,0,2,3).flip(-1)
      q_feat = F.grid_sample(
        torch.cat([feat, feat_coord], dim=1),
        sample_coords,
        mode='bilinear',
        align_corners=False,
      )
      q_feat = q_feat.permute(2,0,3,1)
      q_feat, q_coord = q_feat[..., :-2], q_feat[..., -2:]

      rel_coord = coord[None] - q_coord
      rel_coord[..., 0] *= feat.shape[-2]
      rel_coord[..., 1] *= feat.shape[-1]
      inp = torch.cat([q_feat, rel_coord], dim=-1)
      if self.cell_decode: inp = torch.cat([inp, rel_cell[None].expand_as(rel_coord)], dim=-1)

      areas = rel_coord.prod(dim=-1, keepdim=True).abs() + 1e-9
      areas = areas/areas.sum(dim=0, keepdim=True)
      # Why does this exist?
      if self.local_ensemble: areas[[0,1,2,3]] = areas[[3,2,1,0]]


      return fat_tanh(self.imnet(inp)).mul(areas).sum(dim=0)
    def forward(self, inp, coord, cell):
      self.gen_feat(inp)
      return self.query_rgb(coord, cell)
