import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model, register_loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


@register_model("MazeAutoencoder")
class MazeAutoencoder(nn.Module):
    """
    Simple convolutional autoencoder that maps a 1x45x45 maze to a 1x45x45 solved path mask.

    Forward returns a dict with 'predictions' for compatibility with training loop.
    """
    def __init__(self, latent_dim: int = 128, **kwargs):
        super().__init__()

        # Encoder: downsample to capture global structure
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),
            nn.MaxPool2d(2, ceil_mode=True),  # 45 -> 23
            ConvBlock(32, 64),
            nn.MaxPool2d(2, ceil_mode=True),  # 23 -> 12
            ConvBlock(64, 128),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),  # 12 -> 6
            ConvBlock(128, 128),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, latent_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Decoder: upsample back to 45x45
        self.decoder = nn.Sequential(
            ConvBlock(latent_dim, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 6 -> 12
            ConvBlock(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 12 -> 24
            ConvBlock(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        # Final resize to ensure exact 45x45 output, then sigmoid
        self.output_resize = nn.Upsample(size=(45, 45), mode='bilinear', align_corners=False)

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        y = self.decoder(z)
        y = self.output_resize(y)
        return {'predictions': y}

    # Train loop expects a .compile() method; provide a no-op for compatibility
    def compile(self):
        return None


@register_loss("maze_bce")
def maze_bce_loss(predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
    return F.binary_cross_entropy_with_logits(predictions, targets)


