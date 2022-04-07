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

    linear=nn.Linear, #nn.Linear,
  ):
    assert init in mlp_init_kinds, "Must use init kind"
    super(MLP, self).__init__()
    self.in_size = in_size
    num_layers = len(hidden_list)
    hidden_size = max(hidden_list)

    hidden_layers = [
      linear(hidden_size, hidden_size, bias=bias,)
      for i in range(num_layers)
    ]

    self.init = linear(in_size, hidden_size,bias=bias)
    self.layers = nn.ModuleList(hidden_layers)
    self.out = linear(hidden_size, out, bias=bias)
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
      x = layer(self.activation(x))
    out_size = self.out.out_features
    return self.out(self.activation(x)).reshape(batches + (out_size,))

