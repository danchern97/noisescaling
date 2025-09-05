import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from . import register_model, register_loss
from .scalers import StaticScaler


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=True, dropout_p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p) if dropout_p and dropout_p > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p) if dropout_p and dropout_p > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


@register_model("MazesBigAutoencoder")
class MazesBigAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        base_channels: int = 64,
        dropout: float = 0.3,
        scaler: Optional[nn.Module] = None,
        aggregator: Optional[nn.Module] = None,
        injection_point: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.scaler = scaler
        self.aggregator = aggregator
        self.injection_point = injection_point
        c1, c2, c3, c4 = base_channels, base_channels*2, base_channels*4, base_channels*8

        # Encoder
        self.enc1 = ConvBlock(1, c1, dropout_p=dropout)
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)  # 91 -> 46
        self.enc2 = ConvBlock(c1, c2, dropout_p=dropout)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)  # 46 -> 23
        self.enc3 = ConvBlock(c2, c3, dropout_p=dropout)
        self.pool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # 23 -> 12
        self.enc4 = ConvBlock(c3, c4, dropout_p=dropout)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c4, latent_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 12 -> 24
        self.dec3 = ConvBlock(latent_dim + c3, c3, dropout_p=dropout)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 24 -> 48
        self.dec2 = ConvBlock(c3 + c2, c2, dropout_p=dropout)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 48 -> 96
        self.dec1 = ConvBlock(c2 + c1, c1, dropout_p=dropout)
        self.head = nn.Conv2d(c1, 1, kernel_size=3, padding=1)
        self.output_resize = nn.Upsample(size=(91, 91), mode='bilinear', align_corners=False)

    def _forward_from_injection(self, injection_point: str, tensors):
        e1, e2, e3, e4, z = tensors

        def decode_from_z(z_local: torch.Tensor) -> torch.Tensor:
            d3 = self.up3(z_local)
            if d3.shape[-2:] != e3.shape[-2:]:
                d3 = F.interpolate(d3, size=e3.shape[-2:], mode='bilinear', align_corners=False)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            if d2.shape[-2:] != e2.shape[-2:]:
                d2 = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            if d1.shape[-2:] != e1.shape[-2:]:
                d1 = F.interpolate(d1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            y = self.head(d1)
            y = self.output_resize(y)
            return y

        if injection_point == '0':
            src = e1
            def to_z(t):
                p1 = self.pool1(t)
                e2_ = self.enc2(p1)
                p2 = self.pool2(e2_)
                e3_ = self.enc3(p2)
                p3 = self.pool3(e3_)
                e4_ = self.enc4(p3)
                return self.bottleneck(e4_)
        elif injection_point == '1':
            src = e2
            def to_z(t):
                p2 = self.pool2(t)
                e3_ = self.enc3(p2)
                p3 = self.pool3(e3_)
                e4_ = self.enc4(p3)
                return self.bottleneck(e4_)
        elif injection_point == '2':
            src = e3
            def to_z(t):
                p3 = self.pool3(t)
                e4_ = self.enc4(p3)
                return self.bottleneck(e4_)
        elif injection_point == '3':
            src = e4
            def to_z(t):
                return self.bottleneck(t)
        elif injection_point == '4':
            src = z
            def to_z(t):
                return t
        else:
            raise ValueError(f"Invalid injection_point: {injection_point}")

        reps = self.scaler(src)
        preds = []
        for i in range(reps.shape[1]):
            z_i = to_z(reps[:, i])
            y = decode_from_z(z_i)
            preds.append(y)
        return torch.stack(preds, dim=1), reps

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        z = self.bottleneck(e4)

        normalized_ip = self.injection_point
        if isinstance(normalized_ip, str) and normalized_ip.startswith('mazesbig'):
            normalized_ip = normalized_ip.replace('mazesbig', '')

        if self.scaler is None or self.aggregator is None or normalized_ip is None:
            d3 = self.up3(z)
            if d3.shape[-2:] != e3.shape[-2:]:
                d3 = F.interpolate(d3, size=e3.shape[-2:], mode='bilinear', align_corners=False)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            if d2.shape[-2:] != e2.shape[-2:]:
                d2 = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            if d1.shape[-2:] != e1.shape[-2:]:
                d1 = F.interpolate(d1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            y = self.head(d1)
            y = self.output_resize(y)
            return {'predictions': y}

        stacked_preds, expert_reps = self._forward_from_injection(normalized_ip, (e1, e2, e3, e4, z))
        aggregated = self.aggregator(stacked_preds)
        return {'predictions': aggregated, 'expert_representations': expert_reps}

    def compile(self):
        return None

    @torch.no_grad()
    def predict_per_expert(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaler is None or self.aggregator is None or self.injection_point is None:
            out = self.forward(x)
            y = out['predictions'] if isinstance(out, dict) else out
            return y.unsqueeze(1)

        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        z = self.bottleneck(e4)

        normalized_ip = self.injection_point
        if isinstance(normalized_ip, str) and normalized_ip.startswith('mazesbig'):
            normalized_ip = normalized_ip.replace('mazesbig', '')

        stacked_preds, _ = self._forward_from_injection(normalized_ip, (e1, e2, e3, e4, z))
        return stacked_preds


class _Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


@register_model("MazesBigStaticScaler")
class MazesBigStaticScaler(StaticScaler):
    def __init__(self, n_transforms: int = 4, layer_type: str = 'conv', dim: int = 64, dropout: float = 0.2, **kwargs):
        if layer_type != 'conv':
            raise ValueError("MazesBigStaticScaler supports only 'conv' layer_type")
        transformations = []
        for _ in range(max(0, n_transforms - 1)):
            block = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout) if dropout and dropout > 0 else nn.Identity(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout) if dropout and dropout > 0 else nn.Identity(),
            )
            transformations.append(_Residual(block))
        transformations.append(nn.Identity())
        super().__init__(transformations)


