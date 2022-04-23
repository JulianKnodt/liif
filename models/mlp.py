import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from .ctrl import StructuredDropout

from models import register

mlp_init_kinds = {
    None,
    "zero",
    "kaiming",
    "siren",
    "xavier",
}

@register("mlp")
class MLP(nn.Module):
  "MLP which uses triangular layers for efficiency"
  def __init__(
    self,
    in_features:int=571,
    out_features:int=3,
    hidden_sizes=[256] * 3,
    # instead of outputting a single color, output multiple colors
    bias:bool=True,

    activation=nn.LeakyReLU(inplace=True),
    init="xavier",
    dropout = StructuredDropout(p=0.5,lower_bound=3),
  ):
    assert init in mlp_init_kinds, "Must use init kind"

    super().__init__()

    self.layers = nn.ModuleList([
      nn.Linear(hidden_size, hidden_sizes[i+1], bias=bias)
      for i, hidden_size in enumerate(hidden_sizes[:-1])
    ])

    assert(isinstance(dropout, StructuredDropout))
    self.dropout = dropout


    self.init = nn.Linear(in_features, hidden_sizes[0], bias=bias)
    self.out = nn.Linear(hidden_sizes[-1], out_features, bias=bias)

    self.act = activation

    if init is None: return

    weights = [self.init.weight, self.out.weight, *[l.weight for l in self.layers]]
    biases = [self.init.bias, self.out.bias, *[l.bias for l in self.layers]]

    if init == "zero":
      for t in weights: nn.init.zeros_(t)
      for t in biases: nn.init.zeros_(t)
    elif init == "siren":
      for t in weights:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(t)
        a = math.sqrt(6 / fan_in)
        nn.init._no_grad_uniform_(t, -a, a)
      for t in biases: nn.init.zeros_(t)
    elif init == "kaiming":
      for t in weights: nn.init.kaiming_normal_(t, mode="fan_out")
      for t in biases: nn.init.zeros_(t)

  def set_latent_budget(self,ls:int): self.dropout.set_latent_budget(ls)
  def forward(self, p):
    flat = p.reshape(-1, p.shape[-1])

    x, cutoff = self.dropout.pre_apply_linear(self.init, flat, self.init.out_features)

    for i, layer in enumerate(self.layers):
      x = self.act(x)
      x, cutoff = self.dropout.pre_apply_linear(layer, x, layer.out_features, cutoff)

    out_size = self.out.out_features

    return F.linear(x, self.out.weight[:, :x.shape[-1]], self.out.bias)\
      .reshape(p.shape[:-1] + (out_size,))
