import torch
import torch.nn as nn
from typing import Optional
import itertools
import torch.nn.functional as F
from . import register_model, register_loss
from .scalers import StaticScaler
from .aggregators import Aggregator

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(*self.shape)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", no_batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU() # nn.GELU() works similarly here
        if not no_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.activation(self.conv(x))
        if hasattr(self, 'bn'):
            return self.bn(x)
        else:
            return x
    
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))

class ResidualBlock(nn.Module):
    """
    A simple wrapper that adds a residual connection around a transformation.
    It computes: output = x + transform_block(x)
    
    Args:
        transform_block (nn.Module): The module (e.g., a ConvBlock or LinearBlock)
                                     that learns the residual modification.
    """
    def __init__(self, transform_block: nn.Module):
        super().__init__()
        self.transform_block = transform_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformation and adds the input to the output.
        """
        # The 'transform_block' learns the modification to be made.
        modification = self.transform_block(x)
        
        # Add the modification to the original input.
        return x + modification
        
@register_model("SudokuCNN")
class SudokuCNN(nn.Module):
    def __init__(self, scaler: Optional[nn.Module] = None, aggregator: Optional[nn.Module] = None, scaler_inj_point: int = 3, aggregator_inj_point: int = 7, dropout: float = 0.0, n_mid_layers : int = 1, **kwargs):
        """
        A flexible CNN for Sudoku with fine-grained expert injection points within the decoder.

        Args:
            scaler (Optional[nn.Module]): A module that creates multiple representations.
            aggregator (Optional[nn.Module]): A module that aggregates multiple representations.
            scaler_inj_point (int): The index of the layer where the scaler is injected.
            aggregator_inj_point (int): The index of the layer where the aggregator is injected.
            dropout (float): The dropout rate.
            n_mid_layers (int): The number of middle layers (Conv 512 -> 512) to add
            n_dec_layers (int): The number of decoder layers (Linear 512 -> 512) to add
            **kwargs: Additional arguments for the parent class.
        """
        super(SudokuCNN, self).__init__()

        self.scaler = scaler
        self.aggregator = aggregator

        # -- Define the main model as the list of layers --
        self.layers = [
            ConvBlock(1, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 512)
        ]
        for _ in range(n_mid_layers):
            self.layers.append(ConvBlock(512, 512))

        self.layers.extend([
            ConvBlock(512, 1024),
            ConvBlock(1024, 9),
            nn.Flatten(),
            nn.Linear(9*9*9, 81*9), # Change to 512 if you want the longer decoder, but it performs worse
            nn.GELU(), # nn.ReLU() works very BADLY here
            # nn.Dropout(dropout),
            # nn.Linear(512, 81*9),
            # nn.GELU(),
            nn.LayerNorm(81 * 9),
            Reshape((-1, 9, 9, 9))
        ])
        self.layers = nn.ModuleList(self.layers)

        # -- Validate the scaler and aggregator --
        if (scaler is not None and aggregator is None) or (scaler is None and aggregator is not None):
            raise ValueError("Scaler and aggregator must be either both None or both not None")

        if scaler is not None:
            if 0 <= scaler_inj_point < len(self.layers):
                self.scaler_inj_point = scaler_inj_point
            else:
                raise ValueError(f"Invalid scaler_inj_point '{scaler_inj_point}'. Must be between 0 and {len(self.layers) - 1}")
            if scaler_inj_point < aggregator_inj_point < len(self.layers):
                self.aggregator_inj_point = aggregator_inj_point
            else:
                raise ValueError(f"Invalid aggregator_inj_point '{aggregator_inj_point}'. Must be between '{scaler_inj_point + 1}' and {len(self.layers) - 1}")
        

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        for i, layer in enumerate(self.layers):
            if self.scaler is not None and i == self.scaler_inj_point:
                x = self.scaler(x)
            if isinstance(x, torch.Tensor) and x.dim() > 4:
                x = [x[:, j] for j in range(x.shape[1])]
            if isinstance(x, list):
                x = [layer(x_repr) for x_repr in x]
            else:
                x = layer(x)
        if self.aggregator is not None:
            x = self.aggregator(torch.stack(x, dim=1))
            outputs['expert_representations'] = x
        outputs['predictions'] = x
        return outputs

    def partial_train(self):
        """
        Sets the main model layers to eval mode to avoid distribution shift, and the scaler and aggregator to train mode.
        """
        self.layers.eval()
        self.scaler.train()
        self.aggregator.train()


@register_model("SudokuStaticScaler")
class SudokuStaticScaler(StaticScaler):
    def __init__(self, n_transforms: int = 1, layer_type: str = 'conv', dim: int = 256, use_residual: bool = False, **kwargs):
        """
        A flexible static scaler that can create either convolutional or linear transformations.

        Args:
            n_transforms (int): The total number of representations to create.
            layer_type (str): The type of layer to use. Must be 'conv' or 'linear'.
            dim (int): The feature dimension. For 'conv', this is the number of channels.
                       For 'linear', this is the number of input/output features.
            use_residual (bool): Whether to use a residual block around the transformation.
            **kwargs: Additional arguments for the parent class.
        """
        # --- Validate the layer_type argument ---
        if layer_type not in ['conv', 'linear']:
            raise ValueError(f"Invalid layer_type '{layer_type}'. Must be 'conv' or 'linear'.")

        transformations = []
        dropout_p = kwargs.pop('dropout', 0.0)  # default 0%, can pass via scaler args

        for _ in range(n_transforms - 1):
            if layer_type == 'conv':
                block = [
                    ConvBlock(in_channels=dim, out_channels=dim),
                ]
                
            else:  # layer_type == 'linear'
                block = [
                    LinearBlock(in_features=dim, out_features=dim),
                ]
            if use_residual:
                block = [ResidualBlock(layer) for layer in block]
            if dropout_p > 0:
                block.append(nn.Dropout2d(p=dropout_p) if layer_type == 'conv' else nn.Dropout(p=dropout_p))
                   
            transformations.append(nn.Sequential(*block))

        # Always include the original, unmodified representation
        transformations.append(nn.Identity())

        super().__init__(transformations, **kwargs)

@register_model("WeightedMeanAggregator")
class WeightedMeanAggregator(nn.Module):
    """
    Aggregates expert representations using a learnable weighted average.
    Initialized to strongly favor the last expert (assumed to be the identity path).
    """
    def __init__(self, n_experts: int, initial_identity_bias: float = 10.0):
        super().__init__()
        # We create learnable logits for numerical stability
        self.logits = nn.Parameter(torch.zeros(n_experts))
        
        # Initialize logits to favor the last expert
        with torch.no_grad():
            self.logits[-1] = initial_identity_bias

    def forward(self, expert_reps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_reps (torch.Tensor): Tensor of shape (B, n_reps, D)
        
        Returns:
            torch.Tensor: Aggregated tensor of shape (B, D)
        """
        # Convert logits to weights that sum to 1
        # (1, n_reps, 1) to allow for broadcasting
        weights = F.softmax(self.logits, dim=0)[None, :, None]
        
        # Perform the weighted sum
        # (B, n_reps, D) * (1, n_reps, 1) -> (B, n_reps, D)
        # Then sum along the n_reps dimension
        aggregated = (expert_reps * weights).sum(dim=1)
        
        return aggregated
        
@register_model("SudokuMLPAggregator")
class SudokuMLPAggregator(Aggregator):
    def __init__(self, n_transforms : int = 1, **kwargs):
        super(SudokuMLPAggregator, self).__init__()
        self.aggregator = nn.Sequential(
            nn.Linear(9*9*9*n_transforms, 9*9*9),
            nn.GELU(),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1) # (B, 9*9*9*n_transforms)
        return self.aggregator(x)

@register_loss("cross_entropy_loss")
def cross_entropy_loss(predictions, targets, **kwargs):
    """
    Calculates a cross-entropy loss between the predictions and the targets.
    """
    return nn.functional.cross_entropy(predictions, targets)

@register_loss("cross_entropy_experts_loss")
def cross_entropy_experts_loss(expert_predictions, **kwargs):
    """
    Calculates a cross-entropy loss between each pair of expert predictions.
    Used to encourage diversity among the expert predictions.
    """
    loss = 0.0
    ids_combinations = list(itertools.combinations(range(len(expert_predictions)), 2))
    for i, j in ids_combinations:
        loss += nn.functional.cross_entropy(expert_predictions[i], expert_predictions[j])
    return loss / len(ids_combinations)


@register_loss("orthonormality_loss")
def orthonormality_loss(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    **kwargs) -> torch.Tensor:
    """
    Calculates an orthonormality loss based on the provided formula to encourage
    diversity among the feature dimensions within each expert representation.
    
    This loss is designed to work with a training loop that may or may not provide
    expert representations. If they are not provided, the loss is zero.

    The formula implemented is:
        L = (1/d) * || (1/N_L) * H^T * H - I ||_F^2
    
    where H is a latent representation, d is its feature dimension, and N_L is
    the number of elements.

    Args:
        predictions (torch.Tensor): The main model predictions. This argument is ignored
                                    but required for a consistent function signature.
        targets (torch.Tensor): The ground truth targets. This argument is also ignored.
        **kwargs: Must contain 'expert_reps' (torch.Tensor, optional), a tensor of 
                  expert representations of shape (B, n_reps, ...).

    Returns:
        torch.Tensor: A single scalar loss value, appropriately scaled.
    """
    expert_reps = kwargs.get('expert_reps', None)
    
    # If no experts are present or there's only one (nothing to enforce diversity on), return 0
    if expert_reps is None or expert_reps.shape[1] <= 1:
        return torch.tensor(0.0, device=predictions.device, dtype=torch.float32)

    B, n_reps = expert_reps.shape[0], expert_reps.shape[1]
    
    # Flatten spatial or other dimensions, keeping batch and feature dimensions
    # e.g., (B, n_reps, C, H, W) -> (B, n_reps, C, H*W)
    if expert_reps.dim() > 3:
        reps = expert_reps.flatten(start_dim=3) 
    else:
        reps = expert_reps

    # Here, 'd' is the feature dimension and 'N_L' is the number of elements.
    # For a conv layer (B, n_reps, C, N_L_flat), d=C and N_L=N_L_flat.
    # We permute to get (B, n_reps, N_L_flat, C) to treat C as the feature dim.
    # H_batch has shape (B * n_reps, N_L, d)
    H_batch = reps.permute(0, 1, 3, 2).reshape(B * n_reps, reps.shape[3], reps.shape[2])
    
    N_L = H_batch.shape[1]
    d = H_batch.shape[2]
    
    # Prevent calculation if N_L is 1, as covariance is not meaningful
    if N_L <= 1:
        return torch.tensor(0.0, device=predictions.device, dtype=torch.float32)

    # Calculate the unnormalized feature covariance matrix: H^T * H
    # (B * n_reps, d, N_L) @ (B * n_reps, N_L, d) -> (B * n_reps, d, d)
    covariance = torch.matmul(H_batch.transpose(1, 2), H_batch)

    # Normalize by the number of elements (N_L)
    covariance = covariance / N_L

    # Create the target identity matrix
    identity = torch.eye(d, device=covariance.device).expand_as(covariance)

    # Calculate the squared Frobenius norm of the difference, normalized by d.
    # F.mse_loss(A, B) = (1/n) * ||A - B||_F^2, where n is the number of elements in A.
    # Here n = d*d. The formula asks for (1/d) * ||...||_F^2.
    # So, we multiply mse_loss by d to get the desired scaling.
    loss = d * F.mse_loss(covariance, identity)
    
    return loss
