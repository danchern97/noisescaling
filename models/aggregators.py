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
        return torch.max(representations, dim=self.aggregate_dim)