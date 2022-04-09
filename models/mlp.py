import torch.nn as nn
import torch.nn.functional as F
import torch
import math

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
  "MLP with skip connections and fourier encoding"
  def __init__(
    self,
    hidden_list=[256] * 3,
    in_size=571,
    # instead of outputting a single color, output multiple colors
    out=3,
    activation=nn.LeakyReLU(inplace=True),
    init="xavier",
    bias=True,

    skip=2,

    linear=nn.Linear, #nn.Linear,
  ):
    assert init in mlp_init_kinds, "Must use init kind"
    super(MLP, self).__init__()
    self.in_size = in_size
    num_layers = len(hidden_list)

    self.skip = skip

    hidden_layers = [
      linear(sz + (in_size if i % skip == 0 else 0), hidden_list[i+1], bias=bias)
      for i, sz in enumerate(hidden_list[:-1])
    ]

    self.init = linear(in_size,hidden_list[0],bias=bias)
    self.layers = nn.ModuleList(hidden_layers)
    self.out = linear(hidden_list[-1], out, bias=bias)
    self.activation = activation

    if init is None: return

    weights = [self.init.weight, self.out.weight, *[l.weight for l in self.layers]]
    biases = [self.init.bias, self.out.bias, *[l.bias for l in self.layers]]

    if init == "zero":
      for t in weights: nn.init.zeros_(t)
      for t in biases: nn.init.zeros_(t)
    elif init == "xavier":
      for t in weights: nn.init.xavier_uniform_(t)
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
  def forward(self, p):
    batches = p.shape[:-1]
    init = p.reshape(-1, p.shape[-1])

    x = self.init(init)
    for i, layer in enumerate(self.layers):
      if i % self.skip == 0: x = torch.cat([init, x], dim=-1)
      x = layer(self.activation(x))
    out_size = self.out.out_features
    return self.out(self.activation(x)).reshape(batches + (out_size,))

