import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
import torch


from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandFlipd,
    RandAffined,
    RandCropByPosNegLabeld,
    NormalizeIntensityd,
    EnsureTyped,
)


class Luna16Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=True, crop_shape=(64, 256, 256), cache=False):
        super().__init__()
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.npy")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.npy")))
        assert len(self.image_paths) == len(self.mask_paths), "Images and masks mismatch!"
        self.augment = augment
        self.crop_shape = crop_shape
        self.cache = cache

        self.transforms = self._build_transforms()

        if cache:
            print("⚙️ Caching dataset in memory...")
            self.data = []
            for img_p, msk_p in zip(self.image_paths, self.mask_paths):
                self.data.append({
                    "image": np.load(img_p).astype(np.float32),
                    "mask": np.load(msk_p).astype(np.uint8)
                })
            print(f"✅ Cached {len(self.data)} samples in RAM.")

    def _build_transforms(self):
        base = [
            EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]

        if self.augment:
            base.extend([
                RandCropByPosNegLabeld(
                    keys=["image", "mask"],
                    label_key="mask",
                    spatial_size=self.crop_shape,
                    pos=1,
                    neg=1,
                    num_samples=1,
                    allow_smaller=True
                ),
                RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
                RandAffined(
                    keys=["image", "mask"],
                    prob=0.3,
                    rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest")
                ),
            ])
        base.append(EnsureTyped(keys=["image", "mask"]))
        return Compose(base)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cache:
            data = self.data[idx].copy()
        else:
            img = np.load(self.image_paths[idx]).astype(np.float32)
            msk = np.load(self.mask_paths[idx]).astype(np.uint8)
            data = {"image": img, "mask": msk}

        data = self.transforms(data)
        
        if isinstance(data, list):
            data = data[0]
        
        return data["image"], data["mask"]


def get_dataloaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                    batch_size=1, crop_shape=(64, 256, 256),
                    num_workers=4, augment=True):
    train_ds = Luna16Dataset(train_img_dir, train_mask_dir, augment=augment, crop_shape=crop_shape)
    val_ds = Luna16Dataset(val_img_dir, val_mask_dir, augment=False, crop_shape=crop_shape)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(
        train_img_dir="data/LUNA16/processed/images/positive",
        train_mask_dir="data/LUNA16/processed/masks/positive",
        val_img_dir="data/LUNA16/processed/images/negative",
        val_mask_dir="data/LUNA16/processed/masks/negative",
        batch_size=1,
        crop_shape=(64, 256, 256),
        augment=True,
    )

    # Статистика датасета ДО трансформаций
    print("\n" + "="*60)
    print("📊 Статистика ИСХОДНОГО датасета:")
    print("="*60)
    print(f"✅ Положительные КТ (с узелками): {len(train_loader.dataset)} файлов")
    print(f"❌ Отрицательные КТ (без узелков): {len(val_loader.dataset)} файлов")
    print(f"📦 Всего КТ: {len(train_loader.dataset) + len(val_loader.dataset)} файлов")
    print("="*60 + "\n")

    # Подсчет положительных/отрицательных ПОСЛЕ трансформаций
    print("⏳ Анализ датасета после трансформаций (это может занять время)...\n")
    
    def analyze_dataset(loader, name):
        positive_count = 0
        negative_count = 0
        total_nodule_voxels = 0
        
        for img, mask in loader:
            has_nodules = (mask.sum() > 0).item()
            if has_nodules:
                positive_count += 1
                total_nodule_voxels += mask.sum().item()
            else:
                negative_count += 1
        
        print(f"📈 {name}:")
        print(f"  ✅ Положительные патчи (с узелками): {positive_count}")
        print(f"  ❌ Отрицательные патчи (без узелков): {negative_count}")
        print(f"  📊 Соотношение pos/neg: {positive_count}/{negative_count} = {positive_count/(negative_count+1e-6):.2f}")
        if positive_count > 0:
            print(f"  🎯 Среднее вокселей узелков на патч: {total_nodule_voxels/positive_count:.1f}")
        print()
        
        return positive_count, negative_count
    
    train_pos, train_neg = analyze_dataset(train_loader, "Train (positive КТ)")
    val_pos, val_neg = analyze_dataset(val_loader, "Val (negative КТ)")
    
    print("="*60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА ПОСЛЕ ТРАНСФОРМАЦИЙ:")
    print("="*60)
    print(f"✅ Всего положительных патчей: {train_pos + val_pos}")
    print(f"❌ Всего отрицательных патчей: {train_neg + val_neg}")
    print(f"📦 Всего патчей: {train_pos + val_pos + train_neg + val_neg}")
    print(f"⚖️  Баланс классов: {(train_pos + val_pos)/(train_pos + val_pos + train_neg + val_neg)*100:.1f}% positive")
    print("="*60 + "\n")

    # Пример батча
    print("🔍 Пример батча из train:")
    for img, mask in train_loader:
        print(f"  Image shape: {img.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Unique mask values: {torch.unique(mask).tolist()}")
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
        
        has_nodules = (mask.sum() > 0).item()
        print(f"  Содержит узелки: {'Да ✓' if has_nodules else 'Нет ✗'}")
        if has_nodules:
            print(f"  Количество вокселей с узелками: {mask.sum().item():.0f}")
        break