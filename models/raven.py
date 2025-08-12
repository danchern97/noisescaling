import torch
import torch.nn as nn

from . import register_model, register_loss, MODEL_REGISTRY

# Ensure upstream RAVEN modules are importable regardless of CWD by adjusting sys.path
import sys
from pathlib import Path
_this_file = Path(__file__).resolve()
_arc_root = _this_file.parents[2]
_raven_src_dir = _arc_root / 'RAVEN' / 'src'
if str(_raven_src_dir) not in sys.path:
    sys.path.insert(0, str(_raven_src_dir))

# Import the upstream RAVEN ResNet18 model wrapper
try:
    from model.resnet18 import Resnet18_MLP as UpstreamResnet18MLP  # type: ignore
except Exception:
    UpstreamResnet18MLP = None


@register_loss("raven_cross_entropy")
def raven_cross_entropy(predictions, targets, **kwargs):
    return nn.functional.cross_entropy(predictions, targets)


class _ArgsShim:
    """
    Minimal shim to satisfy the upstream Resnet18_MLP constructor signature.
    """
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, meta_alpha=0.0, meta_beta=0.0):
        self.model = "Resnet18_MLP"
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.meta_alpha = meta_alpha
        self.meta_beta = meta_beta


@register_model("raven_resnet18")
class RavenResNet18(nn.Module):
    """
    Wrapper that adapts authors' Resnet18_MLP to the noisescaling training interface.

    Expects inputs as a tuple: (images, embedding, indicator)
      - images: Tensor (B, 16, H, W)
      - embedding: Tensor (B, 6, 300)
      - indicator: Tensor (B, 1)

    Returns classification logits of shape (B, 8).
    """
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, meta_alpha=0.0, meta_beta=0.0, scaler=None, aggregator=None, **kwargs):
        super().__init__()
        if UpstreamResnet18MLP is None:
            raise ImportError("RAVEN upstream model code is not available. Ensure the RAVEN repo exists and is importable.")

        args = _ArgsShim(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, meta_alpha=meta_alpha, meta_beta=meta_beta)
        self.upstream = UpstreamResnet18MLP(args)

    def forward(self, inputs):
        images, embedding, indicator = inputs
        # Upstream expects NHWC flattened to (B*?, 16, 224, 224) internally; it handles reshaping.
        # It returns a tuple: (pred, meta_target_pred, meta_struct_pred)
        pred, _, _ = self.upstream(images, embedding, indicator)
        return pred

# Alias to match authors' model name in configs if desired
MODEL_REGISTRY["Resnet18_MLP"] = RavenResNet18


