import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

def fat_tanh(v, eps:float=1e-2): return v.tanh() * (1 + eps)

# Conditionally zeros out the last components of a vector
class StructuredDropout(nn.Module):
  def __init__(
    self,
    # chance of turning off bits.
    p=0.5,
    # the minimum number of features to always retain
    lower_bound:int=1,
    eval_size=None,
    zero_pad:bool = False,

    step:int=1,
  ):
    assert(p >= 0)
    assert(p <= 1)
    assert(step >= 1)
    assert(lower_bound > 0), "Cannot use 0 or below as lower bound"

    super().__init__()
    self.p = p
    self.step = step
    self.lower_bound = lower_bound
    self.eval_size = eval_size

    self.zero_pad = zero_pad

  def forward(self, x):
    p = self.p
    upper = x.shape[-1]

    if not self.training:
      esz = self.eval_size
      if esz is None or esz >= upper: return x

      cut = (upper/esz) * x[..., :esz]
      return cut if not self.zero_pad else F.pad(cut, (0,upper-esz))

    cutoff = self.cutoff(upper)
    if cutoff is None: return x

    cut = (upper/cutoff) * x[..., :cutoff]
    return cut if not self.zero_pad else F.pad(cut, (0, upper-cutoff))

  def set_latent_budget(self, ls:int): self.eval_size = ls
  def cutoff(self, upper):
    if random.random() > self.p or self.lower_bound >= upper: return None
    return random.choice(range(self.lower_bound, upper, self.step))
  # Apply the linear layer that precedes x more cheaply.
  def pre_apply_linear(self, lin, x, output_features:int):
    cutoff = self.cutoff(output_features) if self.training else self.eval_size

    if cutoff is None: return lin(x)

    bias = None if lin.bias is None else lin.bias[:cutoff]
    return F.linear(x, lin.weight[:cutoff], bias) * output_features/cutoff

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
      self.feat_dropout = StructuredDropout(p=0.5, zero_pad=True)
      # Just a standard MLP usually, altho I added skip connections for fun.
      self.imnet = models.make(imnet_spec)

    # TODO maybe want to constrain this, so that when encoding we can quantize it better.
    def gen_feat(self, inp): return self.encoder(inp).tanh()

    @torch.jit.export
    def query_rgb(self, feat, coord, cell):
      if self.feat_unfold:
        f0, f1, f2, f3 = feat.shape
        feat = F.unfold(feat, 3, padding=1).reshape(f0, f1 * 9, f2, f3)
      eps = 0.
      if self.local_ensemble:
        offsets = torch.tensor([
          [-1,-1],[-1, 1],[1,-1],[1,1],
        ], dtype=torch.float, device=coord.device)
        eps = 1e-6
      else:
        offsets = torch.tensor([[0,0]], dtype=torch.float, device=coord.device)
      rad = torch.tensor([2/r/2 for r in feat.shape[2:]],device=coord.device,dtype=torch.float)

      feat_coord = make_coord(feat.shape[-2], feat.shape[-1], flatten=False)\
        .to(feat.device)\
        .permute(2, 0, 1)[None]\
        .expand(feat.shape[0], 2, feat.shape[-2], feat.shape[-1])

      #if self.cell_decode:
      assert(self.cell_decode)
      rel_cell = cell.clone()
      rel_cell[:, :, 0] *= feat.shape[-2]
      rel_cell[:, :, 1] *= feat.shape[-1]

      ensemble_coords = coord[None] + (offsets * rad[None])[:, None, None, :] + eps

      ensemble_coords = ensemble_coords.clamp_(min=-1+1e-10, max=1-1e-10)
      sample_coords = ensemble_coords.permute(1,0,2,3).flip(-1)
      q_feat = F.grid_sample(
        torch.cat([feat, feat_coord], dim=1),
        sample_coords,
        # necessary to use nearest, somehow works better than bilinear
        mode='nearest',
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
      return fat_tanh(self.imnet(inp).mul(areas).sum(dim=0))
    def forward(self, inp, coord, cell):
      return self.query_rgb(self.gen_feat(inp), coord, cell)
