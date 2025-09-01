import torch
import torch.nn as nn

from . import register_model, register_loss, MODEL_REGISTRY
from .scalers import StaticScaler
from .aggregators import MeanAggregator

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
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, meta_alpha=0.0, meta_beta=0.0, scaler=None, aggregator=None, injection_point=None, **kwargs):
        super().__init__()
        if UpstreamResnet18MLP is None:
            raise ImportError("RAVEN upstream model code is not available. Ensure the RAVEN repo exists and is importable.")

        args = _ArgsShim(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, meta_alpha=meta_alpha, meta_beta=meta_beta)
        self.upstream = UpstreamResnet18MLP(args)
        self.scaler = scaler
        self.aggregator = aggregator
        # Default injection point is 'embedding' to preserve current behavior
        self.injection_point = injection_point or 'embedding'

    def forward(self, inputs):
        images, embedding, indicator = inputs
        images = images.contiguous()
        embedding = embedding.contiguous()
        indicator = indicator.contiguous()

        # If no scaler/aggregator provided, run base model
        if self.scaler is None or self.aggregator is None:
            pred, _, _ = self.upstream(images, embedding, indicator)
            return pred

        ip = self.injection_point

        if ip == 'embedding':
            # Scale symbolic embedding before tree fusion
            embeddings_reps = self.scaler(embedding)  # (B, n_reps, 6, 300)
            logits_list = []
            for i in range(embeddings_reps.shape[1]):
                emb_i = embeddings_reps[:, i].contiguous()
                pred, _, _ = self.upstream(images, emb_i, indicator)
                logits_list.append(pred)
            logits_stack = torch.stack(logits_list, dim=1)
            return self.aggregator(logits_stack)

        elif ip == 'image':
            # Scale raw 16-channel panels before ResNet
            images_reps = self.scaler(images)  # (B, n_reps, 16, H, W)
            logits_list = []
            for i in range(images_reps.shape[1]):
                img_i = images_reps[:, i].contiguous()
                pred, _, _ = self.upstream(img_i, embedding, indicator)
                logits_list.append(pred)
            logits_stack = torch.stack(logits_list, dim=1)
            return self.aggregator(logits_stack)

        elif ip == 'pre_tree':
            # Scale visual features before tree fusion
            # Extract visual features once
            features = self.upstream.resnet18(images)  # (B, 512)
            features_reps = self.scaler(features)      # (B, n_reps, 512)
            logits_list = []
            for i in range(features_reps.shape[1]):
                feat_i = features_reps[:, i].contiguous()
                feat_tree_i = self.upstream.fc_tree_net(feat_i.view(-1, 1, 512), embedding, indicator)
                final_i = feat_i + 1.0 * feat_tree_i
                out_i = self.upstream.mlp(final_i)
                pred_i = out_i[:, 0:8]
                logits_list.append(pred_i)
            logits_stack = torch.stack(logits_list, dim=1)
            return self.aggregator(logits_stack)

        elif ip == 'post_tree':
            # Scale fused features before MLP
            features = self.upstream.resnet18(images)             # (B, 512)
            features_tree = self.upstream.fc_tree_net(features.view(-1, 1, 512), embedding, indicator)
            final_features = features + 1.0 * features_tree       # (B, 512)
            final_reps = self.scaler(final_features)              # (B, n_reps, 512)
            logits_list = []
            for i in range(final_reps.shape[1]):
                fin_i = final_reps[:, i].contiguous()
                out_i = self.upstream.mlp(fin_i)
                pred_i = out_i[:, 0:8]
                logits_list.append(pred_i)
            logits_stack = torch.stack(logits_list, dim=1)
            return self.aggregator(logits_stack)

        else:
            # Fallback to base behavior
            pred, _, _ = self.upstream(images, embedding, indicator)
            return pred

# Alias to match authors' model name in configs if desired
MODEL_REGISTRY["Resnet18_MLP"] = RavenResNet18


# -------------------------
# Scaler for RAVEN embedding
# -------------------------

class _EmbeddingTransformBlock(nn.Module):
    """
    Small MLP that transforms each of the 6 vectors (dim=300) independently and returns the same shape.
    Input:  (B, 6, 300)
    Output: (B, 6, 300)
    """
    def __init__(self, input_dim: int = 300, hidden_dim: int = 300, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x_flat = x.reshape(b * n, d)
        y_flat = self.net(x_flat)
        return y_flat.reshape(b, n, d)


@register_model("RavenEmbeddingStaticScaler")
class RavenEmbeddingStaticScaler(StaticScaler):
    """
    Static scaler that produces multiple transformed versions of the 6x300 symbolic embedding
    used by the RAVEN authors. Mirrors the Sudoku static scaler pattern.
    """
    def __init__(self, n_transforms: int = 1, hidden_dim: int = 300, dropout: float = 0.1, **kwargs):
        transformations = [
            _EmbeddingTransformBlock(input_dim=300, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(max(0, n_transforms - 1))
        ]
        transformations.append(nn.Identity())
        super().__init__(transformations)


class _FeatureTransformBlock(nn.Module):
    """
    MLP for 512-d features, preserves shape (B, 512).
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("RavenFeatureStaticScaler")
class RavenFeatureStaticScaler(StaticScaler):
    """
    Static scaler for 512-d features (pre_tree or post_tree injection).
    """
    def __init__(self, n_transforms: int = 1, hidden_dim: int = 512, dropout: float = 0.1, **kwargs):
        transformations = [
            _FeatureTransformBlock(input_dim=512, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(max(0, n_transforms - 1))
        ]
        transformations.append(nn.Identity())
        super().__init__(transformations)


class _ImageTransformBlock(nn.Module):
    """
    Lightweight conv block that preserves (B, 16, H, W).
    """
    def __init__(self, channels: int = 16, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("RavenImageStaticScaler")
class RavenImageStaticScaler(StaticScaler):
    """
    Static scaler for raw 16-channel RAVEN panels (image injection).
    """
    def __init__(self, n_transforms: int = 1, channels: int = 16, dropout: float = 0.0, **kwargs):
        transformations = [
            _ImageTransformBlock(channels=channels, dropout=dropout)
            for _ in range(max(0, n_transforms - 1))
        ]
        transformations.append(nn.Identity())
        super().__init__(transformations)


# -----------------------------
# Aggregator for RAVEN logits
# -----------------------------

@register_model("RavenMeanAggregator")
class RavenMeanAggregator(MeanAggregator):
    def __init__(self, aggregate_dim: int = 1, **kwargs):
        super().__init__(aggregate_dim=aggregate_dim, **kwargs)
