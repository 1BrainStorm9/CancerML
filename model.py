"""
3D U-Net модель для сегментации узелков в КТ-снимках лёгких.
Использует MONAI для реализации архитектуры.
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from typing import Tuple


class LungNodule3DUNet(nn.Module):
    """3D U-Net для сегментации узелков лёгких."""
    
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
                 strides: Tuple[int, ...] = (2, 2, 2, 2),
                 num_res_units: int = 2,
                 dropout: float = 0.0):
        """
        Args:
            in_channels: количество входных каналов
            out_channels: количество выходных каналов
            channels: количество каналов на каждом уровне
            strides: шаги для downsampling
            num_res_units: количество residual units
            dropout: вероятность dropout
        """
        super().__init__()
        
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=Norm.BATCH,
            dropout=dropout,
            act='PRELU'
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: входной тензор (B, C, D, H, W)
            
        Returns:
            output: предсказанная маска (B, C, D, H, W)
        """
        return self.unet(x)


class DiceBCELoss(nn.Module):
    """Комбинированный Dice + BCE Loss."""
    
    def __init__(self, 
                 dice_weight: float = 0.5,
                 bce_weight: float = 0.5,
                 smooth: float = 1e-6):
        """
        Args:
            dice_weight: вес для Dice loss
            bce_weight: вес для BCE loss
            smooth: сглаживание для Dice
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Вычисление Dice loss.
        
        Args:
            pred: предсказания (B, C, D, H, W) - логиты
            target: ground truth (B, C, D, H, W)
            
        Returns:
            loss: Dice loss
        """
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Комбинированный loss.
        
        Args:
            pred: предсказания (логиты)
            target: ground truth
            
        Returns:
            combined_loss
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


def create_model(device: str = 'cuda',
                pretrained_path: str = None) -> nn.Module:
    """
    Создание и инициализация модели.
    
    Args:
        device: устройство для модели
        pretrained_path: путь к предобученным весам (опционально)
        
    Returns:
        model: инициализированная модель
    """
    model = LungNodule3DUNet(
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1
    )
    
    if pretrained_path is not None:
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Подсчёт количества обучаемых параметров.
    
    Args:
        model: модель
        
    Returns:
        num_params: количество параметров
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Тест модели
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = create_model(device=device)
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Тестовый forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 64, 256, 256).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Тест loss
    criterion = DiceBCELoss()
    target = torch.randint(0, 2, (batch_size, 1, 64, 256, 256)).float().to(device)
    
    loss = criterion(output, target)
    print(f"\nLoss: {loss.item():.4f}")