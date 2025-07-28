import torch
import torch.nn as nn
from typing import Optional
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
    def __init__(self, scaler : Optional[nn.Module] = None, aggregator : Optional[nn.Module] = None, **kwargs):
        super(SudokuCNN, self).__init__()
        self.scaler = scaler
        self.aggregator = aggregator
        self.enc = nn.Sequential(
            ConvBlock(1, 128),
            ConvBlock(128, 256)
        )
        self.mid = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 1024),
            ConvBlock(1024, 9),
            nn.Flatten(),
        )
        self.dec = nn.Sequential(
            LinearBlock(9*9*9, 512),
            # nn.Dropout(0.2),
            LinearBlock(512, 81 * 9),
            nn.LayerNorm(81 * 9),
            Reshape((-1, 9, 9, 9)),
        )

    def forward(self, x):
        if self.scaler is not None and self.aggregator is not None:
            x = self.enc(x) # (B, 256, 9, 9)
            x = self.scaler(x) # (B, n_reps, 256, 9, 9)
            x = self.aggregator(torch.stack([self.mid(x[:, i]) for i in range(x.shape[1])], dim=1)) # (B, 1024)
            x = self.dec(x) # (B, 9, 9, 9)
            return x
        else:
            return self.dec(self.mid(self.enc(x)))

@register_model("SudokuStaticScaler")
class SudokuStaticScaler(StaticScaler):
    def __init__(self, n_transforms : int = 1, **kwargs):
        transformations = [
            ConvBlock(256, 256)
            for _ in range(n_transforms-1)
        ]
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