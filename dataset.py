"""
PyTorch Dataset для загрузки обработанных КТ-снимков и масок узелков.
Включает аугментации данных.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import monai
from monai.transforms import (
    Compose, RandRotate90d, RandFlipd, RandGaussianNoised,
    RandScaleIntensityd, RandShiftIntensityd, EnsureChannelFirstd,
    ToTensord, RandAffined
)


class LUNA16Dataset(Dataset):
    """Dataset для обработанных LUNA16 сканов."""
    
    def __init__(self,
                 data_dir: str,
                 file_list: List[str],
                 augmentation: bool = True,
                 cache: bool = False):
        """
        Args:
            data_dir: директория с обработанными данными
            file_list: список файлов для загрузки
            augmentation: применять ли аугментации
            cache: кэшировать ли данные в памяти
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.masks_dir = self.data_dir / 'masks'
        self.file_list = file_list
        self.augmentation = augmentation
        
        # Кэш для данных
        self.cache = cache
        self.cached_data = {} if cache else None
        
        # Аугментации
        self.transforms = self._get_transforms() if augmentation else None
        
    def _get_transforms(self):
        """Создание пайплайна аугментаций."""
        return Compose([
            # Случайное вращение на 90 градусов
            RandRotate90d(
                keys=['image', 'mask'],
                prob=0.5,
                spatial_axes=(0, 1)
            ),
            # Случайное отражение
            RandFlipd(
                keys=['image', 'mask'],
                prob=0.5,
                spatial_axis=0
            ),
            RandFlipd(
                keys=['image', 'mask'],
                prob=0.5,
                spatial_axis=1
            ),
            RandFlipd(
                keys=['image', 'mask'],
                prob=0.5,
                spatial_axis=2
            ),
            # Гауссов шум (только для изображения)
            RandGaussianNoised(
                keys=['image'],
                prob=0.3,
                mean=0.0,
                std=0.05
            ),
            # Случайное масштабирование интенсивности
            RandScaleIntensityd(
                keys=['image'],
                factors=0.1,
                prob=0.3
            ),
            # Случайный сдвиг интенсивности
            RandShiftIntensityd(
                keys=['image'],
                offsets=0.1,
                prob=0.3
            ),
            # Случайная аффинная трансформация
            RandAffined(
                keys=['image', 'mask'],
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=['bilinear', 'nearest'],
                padding_mode='border'
            )
        ])
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Загрузка одного примера.
        
        Returns:
            dict с ключами 'image' и 'mask'
        """
        filename = self.file_list[idx]
        
        # Проверяем кэш
        if self.cache and filename in self.cached_data:
            image, mask = self.cached_data[filename]
        else:
            # Загружаем данные
            image_path = self.images_dir / filename
            mask_path = self.masks_dir / filename
            
            image = np.load(image_path)  # (z, y, x)
            mask = np.load(mask_path)    # (z, y, x)
            
            # Кэшируем если нужно
            if self.cache:
                self.cached_data[filename] = (image.copy(), mask.copy())
        
        # Добавляем канальное измерение
        image = image[np.newaxis, ...]  # (1, z, y, x)
        mask = mask[np.newaxis, ...]    # (1, z, y, x)
        
        # Создаём словарь для аугментаций
        data_dict = {
            'image': image.astype(np.float32),
            'mask': mask.astype(np.float32)
        }
        
        # Применяем аугментации
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        
        # Конвертируем в torch tensors
        return {
            'image': torch.from_numpy(data_dict['image']).float(),
            'mask': torch.from_numpy(data_dict['mask']).float()
        }


def get_dataloaders(data_dir: str,
                   train_split: float = 0.8,
                   batch_size: int = 2,
                   num_workers: int = 4,
                   cache_train: bool = False,
                   cache_val: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Создание DataLoader'ов для обучения и валидации.
    
    Args:
        data_dir: директория с обработанными данными
        train_split: доля данных для обучения
        batch_size: размер батча
        num_workers: количество рабочих процессов
        cache_train: кэшировать ли train данные
        cache_val: кэшировать ли validation данные
        
    Returns:
        train_loader, val_loader
    """
    # Читаем список файлов
    file_list_path = Path(data_dir) / 'processed_files.txt'
    with open(file_list_path, 'r') as f:
        all_files = [line.strip() for line in f.readlines()]
    
    # Разделяем на train и validation
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    split_idx = int(len(all_files) * train_split)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Train samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Создаём datasets
    train_dataset = LUNA16Dataset(
        data_dir=data_dir,
        file_list=train_files,
        augmentation=True,
        cache=cache_train
    )
    
    val_dataset = LUNA16Dataset(
        data_dir=data_dir,
        file_list=val_files,
        augmentation=False,
        cache=cache_val
    )
    
    # Создаём dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # для валидации обычно batch_size=1
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Тест dataset
    train_loader, val_loader = get_dataloaders(
        data_dir='data/LUNA16/processed',
        train_split=0.8,
        batch_size=2,
        num_workers=2
    )
    
    print("\nTesting dataloader...")
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        print(f"Mask unique values: {torch.unique(batch['mask'])}")
        break