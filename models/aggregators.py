import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from . import register_model

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