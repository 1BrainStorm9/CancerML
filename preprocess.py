import os
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
from pydicom.dataset import FileDataset
from scipy.ndimage import zoom
from tqdm import tqdm

# --- Параметры ---
INPUT_DIR = Path('data/LUNA16/raw')
ANNOTATIONS_PATH = Path('data/LUNA16/annotations.csv')
OUTPUT_DIR = Path('data/LUNA16/processed')
TARGET_SHAPE = (64, 256, 256)
HU_MIN = -1000
HU_MAX = 400
TARGET_SPACING = (1.0, 1.0, 1.0)

# Папки
POS_DIR = OUTPUT_DIR / 'positive'
NEG_DIR = OUTPUT_DIR / 'negative'
IMAGES_POS_DIR = OUTPUT_DIR / 'images/positive'
IMAGES_NEG_DIR = OUTPUT_DIR / 'images/negative'
MASKS_POS_DIR = OUTPUT_DIR / 'masks/positive'
MASKS_NEG_DIR = OUTPUT_DIR / 'masks/negative'

for d in [POS_DIR, NEG_DIR, IMAGES_POS_DIR, IMAGES_NEG_DIR, MASKS_POS_DIR, MASKS_NEG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# --- Утилиты ---
def load_itk_image(filename):
    itk_image = sitk.ReadImage(str(filename))
    image_array = sitk.GetArrayFromImage(itk_image)  # (D,H,W)
    spacing = np.array(itk_image.GetSpacing())[::-1]
    origin = np.array(itk_image.GetOrigin())[::-1]
    return image_array, origin, spacing


def resample_image(volume, original_spacing, new_spacing):
    scale = original_spacing / np.array(new_spacing)
    new_shape = (volume.shape * scale).astype(int)
    resampled = zoom(volume, scale, order=1)
    return resampled, scale


def clip_and_normalize(volume, hu_min, hu_max):
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


def save_dicom(volume, output_dir, series_uid):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, slice_data in enumerate(volume):
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Modality = 'CT'
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.PatientName = "LUNA16"
        ds.PatientID = "0001"
        ds.ContentDate = str(datetime.date.today()).replace('-', '')
        ds.Rows, ds.Columns = slice_data.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.InstanceNumber = i + 1
        ds.PixelData = slice_data.astype(np.int16).tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(output_dir / f"slice_{i:03d}.dcm", write_like_original=False)


def process_case(case_id, annotations):
    mhd_file = INPUT_DIR / f"{case_id}.mhd"
    if not mhd_file.exists():
        print(f"⚠ Файл {mhd_file} не найден")
        return []

    volume, origin, spacing = load_itk_image(mhd_file)
    volume_resampled, scale = resample_image(volume, spacing, TARGET_SPACING)
    volume_resized = zoom(volume_resampled, np.array(TARGET_SHAPE) / np.array(volume_resampled.shape), order=1)

    # Клиппинг и нормализация
    volume_norm = clip_and_normalize(volume_resized, HU_MIN, HU_MAX)

    # Определяем позитив/негатив
    is_positive = case_id in annotations['seriesuid'].values

    dicom_dir = POS_DIR / case_id if is_positive else NEG_DIR / case_id
    save_dicom(volume_resized, dicom_dir, case_id)

    # Сохраняем изображение в соответствующую папку
    img_dir = IMAGES_POS_DIR if is_positive else IMAGES_NEG_DIR
    np.save(img_dir / f"{case_id}.npy", volume_norm)

    # Создание маски узелков
    coords_new = []
    mask_dir = MASKS_POS_DIR if is_positive else MASKS_NEG_DIR
    mask = np.zeros(TARGET_SHAPE, dtype=np.uint8)
    if is_positive:
        series_ann = annotations[annotations['seriesuid'] == case_id]
        for _, row in series_ann.iterrows():
            z = int((row['coordZ'] - origin[0]) / spacing[0] * scale[0])
            y = int((row['coordY'] - origin[1]) / spacing[1] * scale[1])
            x = int((row['coordX'] - origin[2]) / spacing[2] * scale[2])
            z = np.clip(z, 0, TARGET_SHAPE[0]-1)
            y = np.clip(y, 0, TARGET_SHAPE[1]-1)
            x = np.clip(x, 0, TARGET_SHAPE[2]-1)
            mask[z, y, x] = 1
            coords_new.append([case_id, z, y, x])
    # Сохраняем маску
    np.save(mask_dir / f"{case_id}.npy", mask)

    return coords_new


def split_positive_negative():
    positive_files = [f.name for f in IMAGES_POS_DIR.glob("*.npy")]
    negative_files = [f.name for f in IMAGES_NEG_DIR.glob("*.npy")]
    print(f"Найдено {len(positive_files)} позитивных и {len(negative_files)} негативных образцов.")
    return positive_files, negative_files


def main():
    annotations = pd.read_csv(ANNOTATIONS_PATH) if ANNOTATIONS_PATH.exists() else pd.DataFrame(columns=['seriesuid','coordZ','coordY','coordX'])
    all_coords = []

    mhd_files = list(INPUT_DIR.glob("*.mhd"))
    if not mhd_files:
        print("❌ Нет .mhd файлов для обработки!")
        return

    for f in tqdm(mhd_files, desc="Processing scans"):
        case_id = f.stem
        coords = process_case(case_id, annotations)
        all_coords.extend(coords)

    # Сохраняем пересчитанные аннотации для позитивных КТ
    df_coords = pd.DataFrame(all_coords, columns=['seriesuid', 'coordZ', 'coordY', 'coordX'])
    df_coords.to_csv(POS_DIR / 'annotations_rescaled.csv', index=False)

    # Статистика
    pos_count = len(list(IMAGES_POS_DIR.glob("*.npy")))
    neg_count = len(list(IMAGES_NEG_DIR.glob("*.npy")))
    print(f"\n✅ Конвертация завершена")
    print(f"Позитивные КТ: {pos_count}")
    print(f"Негативные КТ: {neg_count}")
    print(f"Изображения позитивные сохранены в: {IMAGES_POS_DIR}")
    print(f"Изображения негативные сохранены в: {IMAGES_NEG_DIR}")
    print(f"Маски позитивные сохранены в: {MASKS_POS_DIR}")
    print(f"Маски негативные сохранены в: {MASKS_NEG_DIR}")


if __name__ == '__main__':
    main()
