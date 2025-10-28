"""
Improved 3D Attention U-Net for lung nodule segmentation (LUNA16)
Includes:
 - Attention gates
 - Residual connections
 - Instance normalization (stable for small batches)
 - DropBlock (Dropout3D)
 - Refinement conv head
"""

import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet
from monai.networks.layers import Norm
from typing import Tuple


# ======================================================
# Residual Conv Block
# ======================================================
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm='instance'):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels) if norm == 'instance' else nn.BatchNorm3d(out_channels)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels) if norm == 'instance' else nn.BatchNorm3d(out_channels)
        self.act2 = nn.PReLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        out = self.act2(out)
        return out


# ======================================================
# Main Model: Attention ResUNet 3D
# ======================================================
class LungNoduleAttentionResUNet3D(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
                 strides: Tuple[int, ...] = (2, 2, 2, 2),
                 dropout: float = 0.2,
                 use_refine: bool = True,
                 use_residual: bool = True):
        super().__init__()

        # MONAI AttentionUnet without 'norm' argument
        self.unet = AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            # norm=Norm.INSTANCE,  # Убираем, MONAI больше не принимает norm
        )

        # optional residual refinement head
        self.use_residual = use_residual
        self.residual_block = ResidualConvBlock(out_channels, out_channels) if use_residual else nn.Identity()

        # spatial dropout
        self.drop_block = nn.Dropout3d(p=dropout)

        # optional refinement convolution
        self.refine = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1)
        ) if use_refine else nn.Identity()

        self.final_act = nn.Identity()  # keep logits, activation handled in loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unet(x)
        x = self.residual_block(x)
        x = self.drop_block(x)
        x = self.refine(x)
        x = self.final_act(x)
        return x


# ======================================================
# Utility functions
# ======================================================
def create_model(device: str = 'cuda',
                 pretrained_path: str = None,
                 **kwargs) -> nn.Module:
    model = LungNoduleAttentionResUNet3D(**kwargs)
    if pretrained_path is not None:
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ======================================================
# Debug / self-test
# ======================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(device=device,
                         in_channels=1,
                         out_channels=1,
                         channels=(16, 32, 64, 128, 256),
                         dropout=0.2)
    print(f"Model parameters: {count_parameters(model):,}")
    x = torch.randn(1, 1, 64, 128, 128).to(device)
    with torch.no_grad():
        y = model(x)
    print("Input:", tuple(x.shape), "Output:", tuple(y.shape))
