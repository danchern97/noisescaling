import torch
import torch.nn as nn
from typing import Optional
from . import register_model, register_loss

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
        

# TODO: Add ability to plug in Scaler & Aggregator.
@register_model("SudokuCNN")
class SudokuCNN(nn.Module):
    def __init__(self, scaler : Optional[nn.Module] = None, aggregator : Optional[nn.Module] = None):
        super(SudokuCNN, self).__init__()
        self.scaler = scaler
        self.aggregator = aggregator
        module_list = [
            ConvBlock(1, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 1024),
            ConvBlock(1024, 9),
            # ConvBlock(9, 1),
            nn.Flatten(),
            LinearBlock(9*9*9, 512),
            nn.Dropout(0.2),
            LinearBlock(512, 81 * 9),
            nn.LayerNorm(81 * 9),
            Reshape((-1, 9, 9, 9)),
        ]
        if scaler is not None and aggregator is not None:
            module_list.insert(1, scaler)
            module_list.insert(-3, aggregator)
            self.encoder = nn.Sequential(*module_list[:1])
            self.mid = nn.Sequential(*module_list[1:-3])
            self.decoder = nn.Sequential(*module_list[-2:])
        else:
            self.model = nn.Sequential(*module_list)

    def forward(self, x):
        if self.scaler is not None and self.aggregator is not None:
            x = self.encoder(x)
            x = self.aggregator(torch.stack([self.mid(x[:, i]) for i in range(x.shape[1])], dim=1))
            x = self.decoder(x)
            return x
        else:
            return self.model(x)
    
@register_loss("sudoku_loss")
def get_loss_sudoku(predictions, targets, **kwargs):
    return nn.functional.cross_entropy(predictions, targets)