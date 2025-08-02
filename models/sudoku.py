import torch
import torch.nn as nn
from typing import Optional
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
        self.activation = nn.ReLU()
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
       

@register_model("SudokuCNN")
class SudokuCNN(nn.Module):
    def __init__(self, scaler: Optional[nn.Module] = None, aggregator: Optional[nn.Module] = None, injection_point: str = 'pre_dec', **kwargs):
        """
        A flexible CNN for Sudoku with fine-grained expert injection points within the decoder.

        Args:
            scaler (Optional[nn.Module]): A module that creates multiple representations.
            aggregator (Optional[nn.Module]): A module that aggregates multiple representations.
            **kwargs: Must contain `injection_point` to control the expert location.
        """
        super(SudokuCNN, self).__init__()

        self.scaler = scaler
        self.aggregator = aggregator

        # --- Define the valid injection points within the decoder ---
        self.allowed_injection_points: List[str] = [
            '0',  # After encoder
            '1',  # After first mid layer
            '2',  # After second mid layer
            '3',  # After third mid layer
            '4',  # After fourth mid layer
        ]
        
        # --- Get the injection point from kwargs, with validation ---
        self.injection_point = injection_point

        if self.scaler is not None and self.injection_point not in self.allowed_injection_points:
            raise ValueError(
                f"Invalid injection_point '{self.injection_point}'. "
                f"With a scaler provided, it must be one of {self.allowed_injection_points}"
            )


        # --- Define the core architectural blocks ---
        self.enc = nn.Sequential(
            ConvBlock(1, 128),
            ConvBlock(128, 256)
        )

        self.mid_1 = ConvBlock(256, 512)
        self.mid_2 = ConvBlock(512, 512)
        self.mid_3 = ConvBlock(512, 1024)
        self.mid_4 = ConvBlock(1024, 9)

        self.flatten = nn.Flatten()

        # --- Break up the decoder from nn.Sequential to allow for injection ---
        self.dec = nn.Sequential(
            LinearBlock(9*9*9, 512),
            LinearBlock(512, 81 * 9),
            nn.LayerNorm(81 * 9),
            Reshape((-1, 9, 9, 9)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the data flow with precise injection points."""
        
        x_enc = self.enc(x)

        # --- Case 1: No experts. Return a dictionary with only predictions. ---
        if self.scaler is None or self.aggregator is None:
            
            x = self.flatten(self.mid_4(self.mid_3(self.mid_2(self.mid_1(x_enc)))))
            final_preds = self.dec(x)
            
            return {'predictions': final_preds}

        # --- Case 2: Experts are used. ---
        
        # Start with the tensor at the point *before* the injection
        if self.injection_point == '0':
            x = x_enc
        elif self.injection_point == '1':
            x = self.mid_1(x_enc)
        elif self.injection_point == '2':
            x = self.mid_2(self.mid_1(x_enc))
        elif self.injection_point == '3':
            x = self.mid_3(self.mid_2(self.mid_1(x_enc)))
        elif self.injection_point == '4':
            x = self.mid_4(self.mid_3(self.mid_2(self.mid_1(x_enc))))
        # Apply the scaler to get the expert representations
        # This is the tensor we need to pass to the loss function.
        expert_reps = self.scaler(x)
        
        # --- Continue the forward pass from the injection point ---
        if self.injection_point == '0':
            x = self.aggregator(torch.stack([self.flatten(self.mid_4(self.mid_3(self.mid_2(self.mid_1(expert_reps[:, i]))))) for i in range(expert_reps.shape[1])], dim=1)) # (B, 1024)
        elif self.injection_point == '1':
            x = self.aggregator(torch.stack([self.flatten(self.mid_4(self.mid_3(self.mid_2(expert_reps[:, i])))) for i in range(expert_reps.shape[1])], dim=1))
        elif self.injection_point == '2':
            x = self.aggregator(torch.stack([self.flatten(self.mid_4(self.mid_3(expert_reps[:, i]))) for i in range(expert_reps.shape[1])], dim=1))
        elif self.injection_point == '3':
            x = self.aggregator(torch.stack([self.flatten(self.mid_4(expert_reps[:, i])) for i in range(expert_reps.shape[1])], dim=1))
        elif self.injection_point == '4':
            x = self.aggregator(torch.stack([self.flatten(expert_reps[:, i]) for i in range(expert_reps.shape[1])], dim=1))

        final_preds = self.dec(x)

        return {
            'predictions': final_preds,
            'expert_representations': expert_reps # Pass the scaler output
        }


@register_model("SudokuStaticScaler")
class SudokuStaticScaler(StaticScaler):
    def __init__(self, n_transforms: int = 1, layer_type: str = 'conv', dim: int = 256, **kwargs):
        """
        A flexible static scaler that can create either convolutional or linear transformations.

        Args:
            n_transforms (int): The total number of representations to create.
            layer_type (str): The type of layer to use. Must be 'conv' or 'linear'.
            dim (int): The feature dimension. For 'conv', this is the number of channels.
                       For 'linear', this is the number of input/output features.
            **kwargs: Additional arguments for the parent class.
        """
        # --- Validate the layer_type argument ---
        if layer_type not in ['conv', 'linear']:
            raise ValueError(f"Invalid layer_type '{layer_type}'. Must be 'conv' or 'linear'.")

        transformations = []
        for _ in range(n_transforms - 1):
            if layer_type == 'conv':
                # Creates a transformation for convolutional feature maps
                transformations.append(ConvBlock(in_channels=dim, out_channels=dim))
            else: # layer_type == 'linear'
                # Creates a transformation for flat vectors
                transformations.append(LinearBlock(in_features=dim, out_features=dim))
        
        # Always include the original, unmodified representation
        transformations.append(nn.Identity())
        
        super().__init__(transformations, **kwargs)

@register_model("SudokuMLPAggregator")
class SudokuMLPAggregator(Aggregator):
    def __init__(self, n_transforms : int = 1, **kwargs):
        super(SudokuMLPAggregator, self).__init__()
        self.aggregator = nn.Sequential(
            nn.Linear(9*9*9*n_transforms, 9*9*9),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1) # (B, 9*9*9*n_transforms)
        return self.aggregator(x)

@register_loss("sudoku_loss")
def get_loss_sudoku(predictions, targets, **kwargs):
    return nn.functional.cross_entropy(predictions, targets)


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