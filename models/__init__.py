import torch
import torch.nn as nn
from typing import Optional

MODEL_REGISTRY = {}
LOSS_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def register_loss(name):
    def decorator(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator

from . import sudoku
from . import resnet
from . import aggregators
# RAVEN authors' model wrapper is registered in a new module
from . import raven
from . import mazes
from . import mazes_big
