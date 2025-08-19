import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from . import register_model
import math 
import torch.nn.functional as F

class Aggregator(nn.Module, ABC):
    """
    Abstract base class for modules that aggregate multiple representations
    into a single representation.
    """

    def __init__(self, aggregate_dim : int = 1, **kwargs):
        self.aggregate_dim = aggregate_dim
        super().__init__()

    @abstractmethod
    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Takes a tensor of multiple representations and aggregates them.

        Args:
            representations (torch.Tensor): A tensor with of shape (B, n_reps, *).

        Returns:
            torch.Tensor: A single aggregated tensor, e.g., of shape (B, *).
        """
        raise NotImplementedError("This is an abstract class")

@register_model("MeanAggregator")
class MeanAggregator(Aggregator):
    """
    A concrete implementation of Aggregator that combines multiple
    representations by taking their mean along the `self.aggregate_dim` dimension.
    """

    def __init__(self, aggregate_dim : int = 1, **kwargs):
        super().__init__(aggregate_dim, **kwargs)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean of the representations along the `self.aggregate_dim` dimension.

        Args:
            representations (torch.Tensor): A tensor of shape (B, n_reps, ...).

        Returns:
            torch.Tensor: A tensor of shape (B, *).
        """
        return torch.mean(representations, dim=self.aggregate_dim) 
    
@register_model("MaxAggregator")
class MaxAggregator(Aggregator):
    """
    A concrete implementation of Aggregator that combines multiple
    representations by taking their maximum along the `self.aggregate_dim` dimension.
    """

    def __init__(self, aggregate_dim : int = 1, **kwargs):
        super().__init__(aggregate_dim, **kwargs)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Computes the maximum of the representations along the `self.aggregate_dim` dimension.
        """
        return torch.max(representations, dim=self.aggregate_dim)[0]
    
@register_model("MaxLNAggregator")
class MaxLNAggregator(Aggregator):
    """
    A concrete implementation of Aggregator that combines multiple
    representations by taking their maximum along the `self.aggregate_dim` dimension with a Layer Normalization.
    """

    def __init__(self, aggregate_dim : int = 1, aggregate_size : int = 1, **kwargs):
        super().__init__(aggregate_dim, **kwargs)
        self.ln = nn.LayerNorm(aggregate_size)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Computes the maximum of the representations along the `self.aggregate_dim` dimension.
        """
        return self.ln(torch.max(representations, dim=self.aggregate_dim)[0])
    
@register_model("MaxBNAggregator")
class MaxBNAggregator(Aggregator):
    """
    A concrete implementation of Aggregator that combines multiple
    representations by taking their maximum along the `self.aggregate_dim` dimension with a Batch Normalization.
    """

    def __init__(self, aggregate_dim : int = 1, aggregate_size : int = 1, **kwargs):
        super().__init__(aggregate_dim, **kwargs)
        self.bn = nn.BatchNorm1d(aggregate_size)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Computes the maximum of the representations along the `self.aggregate_dim` dimension.
        """
        return self.bn(torch.max(representations, dim=self.aggregate_dim)[0])

@register_model("AttentionAggregator")
class AttentionAggregator(Aggregator):
    """
    A concrete implementation of Aggregator that combines multiple
    representations by using a self-attention mechanism.
    A learnable query vector attends to the input representations (keys and values)
    to produce a single aggregated representation.
    """

    def __init__(self, in_features: int, aggregate_dim : int = 1, n_heads: int = 1, **kwargs):
        super().__init__(aggregate_dim, **kwargs)
        if aggregate_dim != 1:
            raise ValueError("AttentionAggregator only supports aggregation along dim 1.")
        
        self.in_features = in_features
        self.n_heads = n_heads
        
        self.query = nn.Parameter(torch.randn(1, self.in_features))
        self.attention = nn.MultiheadAttention(
            embed_dim=self.in_features,
            num_heads=self.n_heads,
            batch_first=True
        )

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Aggregates representations using self-attention.

        Args:
            representations (torch.Tensor): A tensor of shape (B, n_reps, d).

        Returns:
            torch.Tensor: A tensor of shape (B, d).
        """
        batch_size = representations.size(0)
        
        # Shape: (B, 1, d)
        query_expanded = self.query.expand(batch_size, -1, -1)
        
        # key and value are the input representations
        # query is the learnable query vector
        # output shape: (B, 1, d)
        attn_output, _ = self.attention(query_expanded, representations, representations)
        
        # Squeeze to get (B, d)
        return attn_output.squeeze(1)

@register_model("CrossAttentionAggregator")
class CrossAttentionAggregator(Aggregator):
    """
    Aggregates representations using a form of self-attention where the
    query is the mean of all representations.
    
    This aggregator has no learnable parameters itself.
    """

    def __init__(self, aggregate_dim : int = 1, **kwargs):
        super().__init__(aggregate_dim, **kwargs)

    def forward(self, representations: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Aggregates representations by computing a weighted average, where weights
        are determined by attention scores. The query is the mean of all input
        representations.

        Args:
            representations (torch.Tensor): A tensor of shape (B, n_reps, feature_dim).
            dim (int): The dimension to aggregate along. Must be 1 for this implementation.

        Returns:
            torch.Tensor: The aggregated tensor of shape (B, feature_dim).
        """
        if dim != 1:
            raise ValueError(f"CrossAttentionAggregator only supports aggregation along dim=1, but got dim={dim}")

        # Ensure the input has three dimensions (Batch, Reps, Features)
        if representations.dim() != 3:
            raise ValueError(f"Input tensor must have 3 dimensions (B, n_reps, feature_dim), but got shape {representations.shape}")

        _batch_size, _n_reps, feature_dim = representations.shape

        # 1. Define Key, Query, and Value based on the JAX logic
        # Key and Value are the individual representations.
        key = representations   # Shape: (B, n_reps, feature_dim)
        value = representations # Shape: (B, n_reps, feature_dim)

        # The Query is the global summary (mean) of all representations.
        query = torch.mean(key, dim=1, keepdim=True) # Shape: (B, 1, feature_dim)

        # 2. Calculate scaled dot-product attention scores
        # (B, 1, feature_dim) @ (B, feature_dim, n_reps) -> (B, 1, n_reps)
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(feature_dim)

        # 3. Compute attention weights using softmax
        # The weights will sum to 1 across the `n_reps` dimension.
        attn_weights = F.softmax(attn_scores, dim=-1) # Shape: (B, 1, n_reps)

        # 4. Compute the attended context vector (a weighted average of values)
        # (B, 1, n_reps) @ (B, n_reps, feature_dim) -> (B, 1, feature_dim)
        attended_context = torch.matmul(attn_weights, value)

        # 5. Squeeze to remove the aggregation dimension, resulting in the final tensor
        # (B, 1, feature_dim) -> (B, feature_dim)
        aggregated_representation = attended_context.squeeze(1)

        return aggregated_representation