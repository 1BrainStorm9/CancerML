# preprocess.py
import os
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom
from tqdm import tqdm

# --- Параметры ---
INPUT_DIR = Path('data/LUNA16/raw')
ANNOTATIONS_PATH = Path('data/LUNA16/annotations.csv')
OUTPUT_DIR = Path('data/LUNA16/processed')

TARGET_SHAPE = (64, 256, 256)   # (D,H,W)
HU_MIN = -1000
HU_MAX = 400
TARGET_SPACING = (1.0, 1.0, 1.0)
NODULE_RADIUS_MM = 5.0          # радиус узелка для маски

# --- Директории ---
POS_DIR = OUTPUT_DIR / 'positive'
NEG_DIR = OUTPUT_DIR / 'negative'
IMAGES_POS_DIR = OUTPUT_DIR / 'images/positive'
IMAGES_NEG_DIR = OUTPUT_DIR / 'images/negative'
MASKS_POS_DIR = OUTPUT_DIR / 'masks/positive'
MASKS_NEG_DIR = OUTPUT_DIR / 'masks/negative'

for d in [POS_DIR, NEG_DIR, IMAGES_POS_DIR, IMAGES_NEG_DIR, MASKS_POS_DIR, MASKS_NEG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==========================
# Функции обработки
# ==========================
def load_itk_image(filename):
    itk_image = sitk.ReadImage(str(filename))
    image_array = sitk.GetArrayFromImage(itk_image)  # (D,H,W)
    spacing = np.array(itk_image.GetSpacing())[::-1]  # z,y,x
    origin = np.array(itk_image.GetOrigin())[::-1]
    return image_array, origin, spacing


def resample_image(volume, original_spacing, new_spacing):
    scale = original_spacing / np.array(new_spacing)
    volume_resampled = zoom(volume, scale, order=1)
    return volume_resampled, scale


def clip_and_normalize(volume, hu_min, hu_max):
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


def create_nodule_mask(shape, center_voxel, radius_vox):
    """Создание сферической маски узелка."""
    mask = np.zeros(shape, dtype=np.uint8)
    z0, y0, x0 = center_voxel
    zmin, zmax = max(0, z0 - radius_vox), min(shape[0] - 1, z0 + radius_vox)
    ymin, ymax = max(0, y0 - radius_vox), min(shape[1] - 1, y0 + radius_vox)
    xmin, xmax = max(0, x0 - radius_vox), min(shape[2] - 1, x0 + radius_vox)
    
    for zz in range(zmin, zmax + 1):
        for yy in range(ymin, ymax + 1):
            for xx in range(xmin, xmax + 1):
                if (zz - z0) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2 <= radius_vox ** 2:
                    mask[zz, yy, xx] = 1
    return mask


def process_case(case_id, annotations_df):
    mhd_file = INPUT_DIR / f"{case_id}.mhd"
    if not mhd_file.exists():
        print(f"⚠ Файл {mhd_file} не найден")
        return []

    # 1. Загрузка и ресэмплинг
    volume, origin, spacing = load_itk_image(mhd_file)
    volume_resampled, scale = resample_image(volume, spacing, TARGET_SPACING)
    volume_resized = zoom(volume_resampled, np.array(TARGET_SHAPE) / np.array(volume_resampled.shape), order=1)

    # 2. Клиппинг и нормализация
    volume_norm = clip_and_normalize(volume_resized, HU_MIN, HU_MAX)

    # 3. Определяем позитив/негатив
    is_positive = case_id in annotations_df['seriesuid'].values
    img_dir = IMAGES_POS_DIR if is_positive else IMAGES_NEG_DIR
    mask_dir = MASKS_POS_DIR if is_positive else MASKS_NEG_DIR

    # 4. Сохраняем изображение
    np.save(img_dir / f"{case_id}.npy", volume_norm)

    # 5. Создаем маску
    mask = np.zeros(TARGET_SHAPE, dtype=np.uint8)
    coords_new = []
    if is_positive:
        series_ann = annotations_df[annotations_df['seriesuid'] == case_id]
        radius_vox = max(1, int(NODULE_RADIUS_MM / TARGET_SPACING[0]))
        for _, row in series_ann.iterrows():
            # пересчет координат в воксели
            z = int((row['coordZ'] - origin[0]) / spacing[0] * scale[0])
            y = int((row['coordY'] - origin[1]) / spacing[1] * scale[1])
            x = int((row['coordX'] - origin[2]) / spacing[2] * scale[2])
            z = np.clip(z, 0, TARGET_SHAPE[0] - 1)
            y = np.clip(y, 0, TARGET_SHAPE[1] - 1)
            x = np.clip(x, 0, TARGET_SHAPE[2] - 1)
            mask += create_nodule_mask(TARGET_SHAPE, (z, y, x), radius_vox)
            coords_new.append([case_id, z, y, x])

        mask = np.clip(mask, 0, 1)

    np.save(mask_dir / f"{case_id}.npy", mask)
    return coords_new


# ==========================
# Main
# ==========================
def main():
    annotations_df = pd.read_csv(ANNOTATIONS_PATH) if ANNOTATIONS_PATH.exists() else pd.DataFrame(columns=['seriesuid','coordZ','coordY','coordX'])
    all_coords = []

    mhd_files = sorted(INPUT_DIR.glob("*.mhd"))
    if not mhd_files:
        print("❌ Нет .mhd файлов для обработки!")
        return

    for mhd_file in tqdm(mhd_files, desc="Processing scans"):
        case_id = mhd_file.stem
        coords = process_case(case_id, annotations_df)
        all_coords.extend(coords)

    # Сохраняем пересчитанные координаты узелков
    df_coords = pd.DataFrame(all_coords, columns=['seriesuid', 'coordZ', 'coordY', 'coordX'])
    df_coords.to_csv(POS_DIR / 'annotations_rescaled.csv', index=False)

    # Статистика
    pos_count = len(list(IMAGES_POS_DIR.glob("*.npy")))
    neg_count = len(list(IMAGES_NEG_DIR.glob("*.npy")))
    print(f"\n✅ Конвертация завершена")
    print(f"Позитивные КТ: {pos_count}")
    print(f"Негативные КТ: {neg_count}")
    print(f"Изображения позитивные: {IMAGES_POS_DIR}")
    print(f"Изображения негативные: {IMAGES_NEG_DIR}")
    print(f"Маски позитивные: {MASKS_POS_DIR}")
    print(f"Маски негативные: {MASKS_NEG_DIR}")


if __name__ == '__main__':
    main()
