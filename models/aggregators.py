import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class Aggregator(nn.Module, ABC):
    """
    Abstract base class for modules that aggregate multiple representations
    into a single representation.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, representations: torch.Tensor, dim : int = 1) -> torch.Tensor:
        """
        Takes a tensor of multiple representations and aggregates them.

        Args:
            representations (torch.Tensor): A tensor with of shape (B, n_reps, *).
            dim (int): The dimension along which to aggregate.

        Returns:
            torch.Tensor: A single aggregated tensor, e.g., of shape (B, *).
        """
        raise NotImplementedError("This is an abstract class")


class MeanAggregator(Aggregator):
    """
    A concrete implementation of Aggregator that combines multiple
    representations by taking their mean along the representation dimension.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, representations: torch.Tensor, dim : int = 1) -> torch.Tensor:
        """
        Computes the mean of the representations along the `dim` dimension.

        Args:
            representations (torch.Tensor): A tensor of shape (B, n_reps, ...).
            dim (int): The dimension along which to take the mean.

        Returns:
            torch.Tensor: A tensor of shape (B, *).
        """
        return torch.mean(representations, dim=dim) 
    
class MaxAggregator(Aggregator):
    """
    A concrete implementation of Aggregator that combines multiple
    representations by taking their maximum along the representation dimension.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, representations: torch.Tensor, dim : int = 1) -> torch.Tensor:
        """
        Computes the maximum of the representations along the `dim` dimension.
        """
        return torch.max(representations, dim=dim)