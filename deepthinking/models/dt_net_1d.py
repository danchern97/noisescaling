import torch
from torch import nn

from .blocks import BasicBlock1D as BasicBlock
import torch.nn.functional as F

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, width=64,
                 num_classes=2, group_norm=False): # Set num_classes to 2 to match DTNet
        super().__init__()

        self.in_planes = width

        # Initial projection layer
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(4, width) if group_norm else nn.Identity()

        # Residual layers
        self.layers = nn.ModuleList()
        for i in range(len(num_blocks)):
            # Each "layer" is a sequence of blocks
            layer = self._make_layer(block, width, num_blocks[i], stride=1, group_norm=group_norm)
            self.layers.append(layer)

        # --- KEY CHANGE 1: Replace Pooling and FC with a Conv Head ---
        # This preserves the sequence dimension and outputs a class score for each position.
        # This makes it a Seq2Seq model.
        self.head = nn.Conv1d(width, num_classes, kernel_size=1, stride=1, padding=0, bias=True)


    def _make_layer(self, block, planes, num_blocks, stride, group_norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd, group_norm))
            # The original code had a bug here, it didn't update in_planes for the next block
            # self.in_planes = planes * block.expansion # This should be outside the loop if width is constant
        # For a simple ResNet where width doesn't increase, self.in_planes stays `width`
        return nn.Sequential(*layers)


    # --- KEY CHANGE 2: Adapt the forward pass ---
    def forward(self, x, iters_to_do, **kwargs):
        # A standard ResNet is not iterative, so it will just do one deep pass.
        # `iters_to_do` will be used to format the output for the training/testing code.

        # 1. Perform the single forward pass
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        
        # 2. Get per-position logits from the head
        final_logits = self.head(out) # Shape: (batch, num_classes, seq_len)

        # 3. Handle training vs. evaluation outputs
        if self.training:
            # The training loop needs two outputs. The second is the "interim thought",
            # which for a non-iterative model can just be None or the final output itself.
            # The first output is just the result of the single pass.
            # The training loop's logic for `alpha` will handle this correctly.
            # If alpha=1 (progressive loss), it will use this `final_logits`.
            # If alpha=0 (max_iter loss), it will also use this `final_logits`.
            return final_logits, None

        # During evaluation/testing, the code expects an output for each "iteration".
        # We will just repeat our single result `iters_to_do` times.
        else:
            # The DTNet returns shape (batch, iters, classes, seq_len). We mimic that.
            # Unsqueeze to add the "iters" dimension
            final_logits_unsqueezed = final_logits.unsqueeze(1) # Shape: (batch, 1, classes, seq_len)
            # Repeat along the "iters" dimension
            all_outputs = final_logits_unsqueezed.repeat(1, iters_to_do, 1, 1) # Shape: (batch, iters, classes, seq_len)
            return all_outputs


class DTNet1D(nn.Module):
    def __init__(self, block, num_blocks, width, recall,
                 group_norm=False, noise_type='none', noise_block_idx=-1, **kwargs):
        super().__init__()
        self.width = int(width)
        self.recall = recall
        self.group_norm = group_norm
        self.noise_type = noise_type
        self.noise_block_idx = noise_block_idx

        proj_conv = nn.Conv1d(1, width, kernel_size=3, stride=1, padding=1, bias=False)
        conv_recall = nn.Conv1d(width + 1, width, kernel_size=3, stride=1, padding=1, bias=False)

        if self.recall:
            self.recall_proj = nn.Sequential(conv_recall, nn.ReLU())
        else:
            self.recall_proj = nn.Identity()

        self.res_blocks = nn.ModuleList()
        for i in range(len(num_blocks)):
            layer = self._make_layer(block, width, num_blocks[i], stride=1)
            self.res_blocks.append(layer)

        self.head = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1, bias=False), nn.ReLU(),
            nn.Conv1d(width, int(width / 2), kernel_size=3, padding=1, bias=False), nn.ReLU(),
            nn.Conv1d(int(width / 2), 2, kernel_size=3, padding=1, bias=False)
        )

        self.projection = nn.Sequential(proj_conv, nn.ReLU())

        # --- Noise components ---
        if noise_type == 'gaussian':
            self.noise_mlp = nn.Sequential(
                nn.Conv1d(width, width, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(width, width, kernel_size=1)
            )
        elif noise_type == 'conditioned':
            self.noise_mlp = nn.Sequential(
                nn.Conv1d(width, width, kernel_size=1),
                nn.ReLU()
            )
            self.condition_mlp = nn.Conv1d(width, 2 * width, kernel_size=1)  # outputs mean and log_std

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def add_noise(self, x):
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(x)
            return x + self.noise_mlp(noise)

        elif self.noise_type == 'conditioned':
            stats = self.condition_mlp(x)
            mean, log_std = stats.chunk(2, dim=1)
            noise = torch.randn_like(mean)
            z = mean + torch.exp(log_std) * noise
            return x + self.noise_mlp(z)

        return x  # if noise_type is 'none'

    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2))).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
                interim_thought = self.recall_proj(interim_thought)  # Apply the conv to reduce back to width

            for idx, layer in enumerate(self.res_blocks):
                interim_thought = layer(interim_thought)
                if idx == self.noise_block_idx and self.noise_type != 'none':
                    interim_thought = self.add_noise(interim_thought)

            out = self.head(interim_thought)
            all_outputs[:, i] = out

        if self.training:
            return out, interim_thought

        return all_outputs


def dt_net_1d(width, **kwargs):
    return DTNet1D(BasicBlock, [2], width, recall=False)


def dt_net_recall_1d(width, **kwargs):
    return DTNet1D(BasicBlock, [2], width, recall=True)


def dt_net_gn_1d(width, **kwargs):
    return DTNet1D(BasicBlock, [2], width, recall=False, group_norm=True)

def dt_net_recall_gn_1d(width, **kwargs):
    return DTNet1D(BasicBlock, [2], width, recall=True, group_norm=True)


def resnet_1d(width, **kwargs):
    return ResNet1D(BasicBlock, [2,2,2,2], in_channels=1, width=width, num_classes=2, group_norm=True)
