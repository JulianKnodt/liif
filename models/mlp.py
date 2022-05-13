from .ctrl import MLP

from models import register

@register("mlp")
def mlp(*args, **kwargs): return MLP(*args, skip=2, **kwargs)
