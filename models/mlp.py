import torch.nn as nn
import torch

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
    in_size=580,
    out=3,
    skip=3,
    activation=nn.LeakyReLU(inplace=True),
    init=None,
  ):
    assert init in mlp_init_kinds, "Must use init kind"
    super(MLP, self).__init__()
    self.in_size = in_size
    num_layers = len(hidden_list)
    hidden_size = max(hidden_list)

    self.skip = skip
    skip_size = hidden_size + in_size

    hidden_layers = [
        nn.Linear(
            skip_size if (i % skip) == 0 and i != num_layers - 1 else hidden_size, hidden_size,
        )
        for i in range(num_layers)
    ]

    self.init = nn.Linear(in_size, hidden_size)
    self.layers = nn.ModuleList(hidden_layers)
    self.out = nn.Linear(hidden_size, out)
    weights = [self.init.weight, self.out.weight, *[l.weight for l in self.layers]]
    biases = [self.init.bias, self.out.bias, *[l.bias for l in self.layers]]

    if init is None: ...
    elif init == "zero":
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
    self.activation = activation

  def forward(self, p):
    batches = p.shape[:-1]
    init = p.reshape(-1, p.shape[-1])

    x = self.init(init)
    for i, layer in enumerate(self.layers):
      if i != len(self.layers) - 1 and (i % self.skip) == 0:
        x = torch.cat([x, init], dim=-1)
      x = layer(self.activation(x))
    out_size = self.out.out_features
    return self.out(self.activation(x)).reshape(batches + (out_size,))
