import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from .ctrl import StructuredDropout, MLP

from models import register

@register("mlp")
def mlp(*args, **kwargs): return MLP(*args, **kwargs)
