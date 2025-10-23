import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Tuple, Dict, List
from monai.transforms import (
    Compose, RandFlipd, RandGaussianNoised,
    RandScaleIntensityd, RandShiftIntensityd, RandAffined, Resize
)

# ======================================================
# Вспомогательная функция
# ======================================================
def split_positive_negative(data_dir: str):
    """Разделяет файлы на позитивные (с узелками) и негативные (без узелков)."""
    IMAGES_POS_DIR = Path(data_dir) / "images/positive"
    IMAGES_NEG_DIR = Path(data_dir) / "images/negative"

    positive_files = [f.name for f in IMAGES_POS_DIR.glob("*.npy")]
    negative_files = [f.name for f in IMAGES_NEG_DIR.glob("*.npy")]

    print(f"✅ Найдено {len(positive_files)} позитивных и {len(negative_files)} негативных образцов.")
    return positive_files, negative_files


# ======================================================
# Класс датасета
# ======================================================
class LUNA16Dataset(Dataset):
    """Dataset для обработанных LUNA16 сканов с аугментациями."""

    def __init__(self, data_dir: str, file_list: List[str],
                 augmentation: bool = True, cache: bool = False):
        self.data_dir = Path(data_dir)
        self.IMAGES_POS_DIR = self.data_dir / "images/positive"
        self.IMAGES_NEG_DIR = self.data_dir / "images/negative"
        self.MASKS_POS_DIR = self.data_dir / "masks/positive"
        self.MASKS_NEG_DIR = self.data_dir / "masks/negative"

        self.file_list = file_list
        self.augmentation = augmentation
        self.cache = cache
        self.cached_data = {} if cache else None

        self.transforms = self._get_transforms() if augmentation else None

        # Фиксированная целевая форма
        self.target_shape = (64, 256, 256)
        self.resizer = Resize(self.target_shape, mode="nearest")

    def _get_transforms(self):
        return Compose([
            RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.05),
            RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.3),
            RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.3),
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

    def _normalize_shape(self, array: np.ndarray) -> np.ndarray:
        """Приводим массив к форме (64, 256, 256) с помощью MONAI Resize."""
        array = array.astype(np.float32)
        # Добавляем временный канал для Resize
        array = array[np.newaxis, ...]  # (1,D,H,W)
        if array.shape[1:] != self.target_shape:
            array = self.resizer(array)
        # Убираем канал, чтобы получить (D,H,W)
        array = array[0]
        return array.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename = self.file_list[idx]

        if self.cache and filename in self.cached_data:
            image, mask = self.cached_data[filename]
        else:
            if (self.IMAGES_POS_DIR / filename).exists():
                image_path = self.IMAGES_POS_DIR / filename
                mask_path = self.MASKS_POS_DIR / filename
            else:
                image_path = self.IMAGES_NEG_DIR / filename
                mask_path = self.MASKS_NEG_DIR / filename

            image = np.load(image_path)
            mask = np.load(mask_path)

            # Приведение формы к (D,H,W)
            image = self._normalize_shape(image)
            mask = self._normalize_shape(mask)
            mask = (mask > 0).astype(np.uint8)

            if self.cache:
                self.cached_data[filename] = (image.copy(), mask.copy())

        # --- Добавляем канал перед трансформациями ---
        data_dict = {'image': image[np.newaxis, ...], 'mask': mask[np.newaxis, ...]}  # (1,D,H,W)

        # Применяем аугментации
        if self.transforms is not None:
            try:
                data_dict = self.transforms(data_dict)
            except IndexError:
                print(f"Warning: skipping augmentation for {filename} due to small shape {image.shape}")

        # --- В результате получаем (1,D,H,W) ---
        image = data_dict['image'].astype(np.float32)
        mask = data_dict['mask'].astype(np.float32)

        return {
            'image': torch.as_tensor(image, dtype=torch.float32),
            'mask': torch.as_tensor(mask, dtype=torch.float32)
        }


# ======================================================
# Создание DataLoader'ов
# ======================================================
def get_dataloaders(data_dir: str,
                    train_split: float = 0.8,
                    batch_size: int = 2,
                    num_workers: int = 4,
                    cache_train: bool = False,
                    cache_val: bool = False) -> Tuple[DataLoader, DataLoader]:

    positive_files, negative_files = split_positive_negative(data_dir)

    all_files = positive_files + negative_files
    np.random.shuffle(all_files)

    split_idx = int(len(all_files) * train_split)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Train samples: {len(train_files)}, Validation samples: {len(val_files)}")

    train_dataset = LUNA16Dataset(data_dir, train_files, augmentation=True, cache=cache_train)
    val_dataset = LUNA16Dataset(data_dir, val_files, augmentation=False, cache=cache_val)

    # === WeightedRandomSampler для баланса классов ===
    labels = [1 if "positive" in str(f) else 0 for f in train_dataset.file_list]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        num_workers = 0
        pin_memory = False
        persistent_workers = False
    else:
        pin_memory = True
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    # ======================================================
    # Тестируем первый батч
    # ======================================================
    print("\nTesting dataloader...")
    sample = next(iter(train_loader))
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"Mask unique values: {torch.unique(sample['mask'])}")

    # ======================================================
    # Проверка баланса по всем батчам
    # ======================================================
    pos_count = 0
    neg_count = 0
    for batch in train_loader:
        masks = batch['mask']
        for m in masks:
            if m.sum() > 0:
                pos_count += 1
            else:
                neg_count += 1
    total = pos_count + neg_count
    print(f"\nБаланс в train_loader по всем батчам:")
    print(f"Позитивные: {pos_count} ({pos_count/total*100:.2f}%)")
    print(f"Негативные: {neg_count} ({neg_count/total*100:.2f}%)")

    return train_loader, val_loader


# ======================================================
# Тестирование
# ======================================================
if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders(
        data_dir='data/LUNA16/processed',
        train_split=0.8,
        batch_size=2,
        num_workers=2
    )
