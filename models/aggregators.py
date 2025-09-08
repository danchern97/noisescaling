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
            representations (torch.Tensor):
                Either of shape (B, n_reps, d) or (B, n_reps, C, H, W).

        Returns:
            torch.Tensor: A tensor of shape (B, d).
        """
        if representations.dim() == 5:
            # Input is [B, n_reps, C, H, W] → flatten each expert
            B, n_reps, C, H, W = representations.shape
            representations = representations.view(B, n_reps, -1)  # [B, n_reps, C*H*W]
        elif representations.dim() != 3:
            raise ValueError(
                f"Expected input of shape (B, n_reps, d) or (B, n_reps, C, H, W), "
                f"but got {representations.shape}"
            )

        B, n_reps, d = representations.shape
        if d != self.in_features:
            raise ValueError(
                f"Aggregator expected in_features={self.in_features}, "
                f"but got {d} from input."
            )

        # Expand learnable query to batch
        query_expanded = self.query.expand(B, -1, -1)  # [B, 1, d]

        # Apply self-attention
        attn_output, _ = self.attention(query_expanded, representations, representations)  # [B, 1, d]

        return attn_output.squeeze(1)  # [B, d]

@register_model("LocalAttentionAggregator")
class LocalAttentionAggregator(Aggregator):
    """
    Aggregates expert representations using a self-attention mechanism where
    one expert (the identity) acts as the query for all other experts.
    This makes the aggregation context-dependent and spatially aware.
    """
    def __init__(self, embed_dim: int, n_heads: int = 4, **kwargs):
        """
        Args:
            embed_dim (int): The number of channels (C) in the input feature maps.
            n_heads (int): The number of attention heads. Must be a divisor of embed_dim.
        """
        super().__init__(**kwargs)
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads.")
            
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        
        # Standard MultiheadAttention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.n_heads,
            batch_first=True  # Crucial for our tensor shapes
        )

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Aggregates expert feature maps.

        Args:
            representations (torch.Tensor): Shape is [B, n_experts, C, H, W].
                                            The last expert ([-1]) is assumed to be the identity path.

        Returns:
            torch.Tensor: Aggregated feature map of shape [B, C, H, W].
        """
        if representations.dim() != 5:
            raise ValueError(f"Expected input of shape [B, n_experts, C, H, W], but got {representations.shape}")
        
        B, n_experts, C, H, W = representations.shape
        if C != self.embed_dim:
            raise ValueError(f"Aggregator expected embed_dim={self.embed_dim}, but got {C} channels from input.")

        # --- Key Transformation ---
        # The identity expert is our query, representing the "base" context.
        # It needs to attend to all other experts.
        query = representations[:, -1, :, :, :]  # [B, C, H, W] (The identity expert)

        # The attention layer expects sequence-like inputs: [Batch, Sequence_Length, Features].
        # We treat each pixel location as a separate item in the batch and the experts as the sequence.
        
        # 1. Permute to group by pixels: [B, H, W, n_experts, C]
        reps_permuted = representations.permute(0, 3, 4, 1, 2)
        
        # 2. Reshape for attention: treat each pixel as a batch element
        # Keys & Values: [ (B*H*W), n_experts, C ]
        keys_values = reps_permuted.reshape(B * H * W, n_experts, C)
        
        # 3. Reshape the query to match:
        # Query: [ (B*H*W), 1, C ]
        query_permuted = query.permute(0, 2, 3, 1) # [B, H, W, C]
        query_reshaped = query_permuted.reshape(B * H * W, 1, C)

        # Apply attention: each pixel location independently attends to all experts
        attn_output, _ = self.attention(query_reshaped, keys_values, keys_values) # [ (B*H*W), 1, C ]

        # Reshape the output back to its original image format
        # 1. Remove the sequence dim: [ (B*H*W), C ]
        output_flat = attn_output.squeeze(1)
        # 2. Reshape back to image: [B, H, W, C]
        output_image = output_flat.reshape(B, H, W, C)
        # 3. Permute back to PyTorch's standard [B, C, H, W]
        final_output = output_image.permute(0, 3, 1, 2)
        
        return final_output

@register_model("ConvAggregator")
class ConvAggregator(Aggregator):
    """
    Aggregates expert representations using a 1x1 convolution.
    This learns a spatially-aware, linear combination of the expert feature maps.
    """
    def __init__(self, n_experts: int, embed_dim: int, **kwargs):
        """
        Args:
            n_experts (int): The number of incoming expert representations.
            embed_dim (int): The number of channels (C) in each expert's feature map.
        """
        super().__init__(**kwargs)
        self.n_experts = n_experts
        self.embed_dim = embed_dim

        # The 1x1 convolution is the core of this aggregator.
        # It takes (n_experts * embed_dim) input channels and reduces them
        # back down to `embed_dim` output channels.
        self.agg_conv = nn.Conv2d(
            in_channels=n_experts * embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            bias=False # Bias is often omitted in 1x1 for aggregation
        )
        
        # Optional: Add non-linearity and normalization after aggregation
        self.post_agg = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Aggregates expert feature maps.

        Args:
            representations (torch.Tensor): Shape is [B, n_experts, C, H, W].

        Returns:
            torch.Tensor: Aggregated feature map of shape [B, C, H, W].
        """
        if representations.dim() != 5:
            raise ValueError(f"Expected input of shape [B, n_experts, C, H, W], but got {representations.shape}")
        
        B, n_exp, C, H, W = representations.shape
        if n_exp != self.n_experts or C != self.embed_dim:
            raise ValueError(f"Input shape mismatch. Expected {self.n_experts} experts and {self.embed_dim} channels.")

        # Reshape to treat experts as channel groups:
        # [B, n_experts, C, H, W] -> [B, n_experts * C, H, W]
        x_reshaped = representations.view(B, n_exp * C, H, W)
        
        # Apply the 1x1 convolution to get the aggregated feature map
        aggregated = self.agg_conv(x_reshaped) # [B, C, H, W]
        
        # Apply post-aggregation layers
        final_output = self.post_agg(aggregated)
        
        return final_output

@register_model("CrossAttentionAggregator")
class CrossAttentionAggregator(Aggregator):
    """
    Aggregates representations using a form of cross-attention where the
    query is the mean of all representations. Supports feature maps.
    """

    def __init__(self, aggregate_dim: int = 1, **kwargs):
        super().__init__(aggregate_dim, **kwargs)

    def forward(self, representations: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Aggregates representations by computing attention using the mean as query.

        Supports input of shape (B, n_reps, feature_dim) or (B, n_reps, C, H, W).

        Args:
            representations (torch.Tensor): Tensor of shape
                (B, n_reps, feature_dim) or (B, n_reps, C, H, W)
            dim (int): Aggregation dimension. Must be 1.

        Returns:
            torch.Tensor: Aggregated tensor of shape (B, feature_dim)
        """
        if dim != 1:
            raise ValueError(f"CrossAttentionAggregator only supports aggregation along dim=1, but got dim={dim}")

        # Flatten spatial dimensions if needed
        if representations.dim() == 5:
            B, n_reps, C, H, W = representations.shape
            representations = representations.view(B, n_reps, -1)  # [B, n_reps, C*H*W]
        elif representations.dim() != 3:
            raise ValueError(f"Input tensor must have 3 or 5 dimensions, but got {representations.shape}")

        B, n_reps, feature_dim = representations.shape

        # Key and Value are the individual representations
        key = representations        # [B, n_reps, feature_dim]
        value = representations      # [B, n_reps, feature_dim]

        # Query is the mean across n_reps
        query = torch.mean(key, dim=1, keepdim=True)  # [B, 1, feature_dim]

        # Scaled dot-product attention
        attn_scores = torch.matmul(query, key.transpose(-2, -1))  # [B, 1, n_reps]
        attn_scores = attn_scores / math.sqrt(feature_dim)

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, n_reps]

        # Weighted sum of values
        attended_context = torch.matmul(attn_weights, value)  # [B, 1, feature_dim]

        # Squeeze aggregation dimension
        aggregated_representation = attended_context.squeeze(1)  # [B, feature_dim]

        return aggregated_representation