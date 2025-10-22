"""
Предобработка LUNA16 датасета:
- Загрузка .mhd/.raw файлов
- Ресэмплинг к spacing 1x1x1 мм
- Клиппинг по HU диапазону [-1000, 400]
- Нормализация интенсивностей
- Изменение размера к 256x256x64
- Создание масок узелков
- Применение масок лёгких
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from typing import Dict, List, Tuple
import utils


class LUNA16Preprocessor:
    """Класс для предобработки LUNA16 датасета."""
    
    def __init__(self,
                 raw_data_dir: str,
                 annotations_path: str,
                 lung_masks_dir: str,
                 output_dir: str,
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 target_size: Tuple[int, int, int] = (64, 256, 256),  # (z, y, x)
                 hu_min: float = -1000.0,
                 hu_max: float = 400.0):
        """
        Args:
            raw_data_dir: директория с .mhd/.raw файлами
            annotations_path: путь к annotations.csv
            lung_masks_dir: директория с масками лёгких
            output_dir: директория для сохранения обработанных данных
            target_spacing: целевой spacing (x, y, z) в мм
            target_size: целевой размер (z, y, x)
            hu_min: минимальное значение HU
            hu_max: максимальное значение HU
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.annotations_path = annotations_path
        self.lung_masks_dir = Path(lung_masks_dir)
        self.output_dir = Path(output_dir)
        self.target_spacing = np.array(target_spacing)  # (x, y, z)
        self.target_size = target_size  # (z, y, x)
        self.hu_min = hu_min
        self.hu_max = hu_max
        
        # Создаём выходные директории
        self.output_images_dir = self.output_dir / 'images'
        self.output_masks_dir = self.output_dir / 'masks'
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем аннотации
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> pd.DataFrame:
        """Загрузка и группировка аннотаций по серии."""
        df = pd.read_csv(self.annotations_path)
        return df
    
    def _resample_image(self, image: sitk.Image, 
                       new_spacing: np.ndarray) -> sitk.Image:
        """
        Ресэмплинг изображения к новому spacing.
        
        Args:
            image: SimpleITK изображение
            new_spacing: новый spacing (x, y, z)
            
        Returns:
            resampled_image: ресэмплированное изображение
        """
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        
        # Вычисляем новый размер
        new_size = (original_size * original_spacing / new_spacing).astype(int)
        new_size = [int(s) for s in new_size]
        
        # Настраиваем resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(-1000)  # HU воздуха
        resampler.SetInterpolator(sitk.sitkLinear)
        
        return resampler.Execute(image)
    
    def _resize_image(self, image: sitk.Image, 
                     target_size: Tuple[int, int, int]) -> sitk.Image:
        """
        Изменение размера изображения к целевому размеру.
        
        Args:
            image: SimpleITK изображение
            target_size: целевой размер (x, y, z)
            
        Returns:
            resized_image: изображение нового размера
        """
        original_size = image.GetSize()
        original_spacing = np.array(image.GetSpacing())
        
        # Вычисляем новый spacing
        new_spacing = (np.array(original_size) * original_spacing / 
                      np.array(target_size))
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetSize(target_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(-1000)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        return resampler.Execute(image)
    
    def _normalize_hu(self, image_array: np.ndarray) -> np.ndarray:
        """
        Нормализация HU значений.
        
        Args:
            image_array: массив с HU значениями
            
        Returns:
            normalized: нормализованный массив [0, 1]
        """
        # Клиппинг
        image_array = np.clip(image_array, self.hu_min, self.hu_max)
        
        # Нормализация к [0, 1]
        normalized = (image_array - self.hu_min) / (self.hu_max - self.hu_min)
        
        return normalized.astype(np.float32)
    
    def _update_annotations_for_new_params(self,
                                          nodules: List[Dict],
                                          original_origin: np.ndarray,
                                          original_spacing: np.ndarray,
                                          new_origin: np.ndarray,
                                          new_spacing: np.ndarray) -> List[Dict]:
        """
        Обновление координат аннотаций после изменения spacing и размера.
        
        Args:
            nodules: список аннотаций узелков
            original_origin: исходный origin (x, y, z)
            original_spacing: исходный spacing (x, y, z)
            new_origin: новый origin (x, y, z)
            new_spacing: новый spacing (x, y, z)
            
        Returns:
            updated_nodules: обновлённые аннотации
        """
        updated_nodules = []
        
        for nodule in nodules:
            # Исходные мировые координаты
            world_coord = np.array([
                nodule['coordX'],
                nodule['coordY'],
                nodule['coordZ']
            ])
            
            # Конвертируем в voxel координаты исходного изображения
            voxel_coord = utils.world_to_voxel_coords(
                world_coord, original_origin, original_spacing
            )
            
            # Конвертируем обратно в мировые с новыми параметрами
            new_world_coord = utils.voxel_to_world_coords(
                voxel_coord, new_origin, new_spacing
            )
            
            updated_nodule = nodule.copy()
            updated_nodule['coordX'] = new_world_coord[0]
            updated_nodule['coordY'] = new_world_coord[1]
            updated_nodule['coordZ'] = new_world_coord[2]
            
            updated_nodules.append(updated_nodule)
        
        return updated_nodules
    
    def process_single_scan(self, series_uid: str) -> bool:
        """
        Обработка одного КТ скана.
        
        Args:
            series_uid: идентификатор серии
            
        Returns:
            success: True если обработка успешна
        """
        try:
            # Пути к файлам
            mhd_path = self.raw_data_dir / f"{series_uid}.mhd"
            lung_mask_path = self.lung_masks_dir / f"{series_uid}.mhd"
            
            if not mhd_path.exists():
                print(f"Warning: {mhd_path} not found")
                return False
            
            # Загружаем изображение
            itk_image = sitk.ReadImage(str(mhd_path))
            original_origin = np.array(itk_image.GetOrigin())
            original_spacing = np.array(itk_image.GetSpacing())
            
            # Ресэмплинг к целевому spacing
            resampled_image = self._resample_image(itk_image, self.target_spacing)
            
            # Изменение размера к целевому размеру (x, y, z)
            target_size_xyz = (self.target_size[2], self.target_size[1], 
                              self.target_size[0])
            resized_image = self._resize_image(resampled_image, target_size_xyz)
            
            # Получаем новые параметры
            new_origin = np.array(resized_image.GetOrigin())
            new_spacing = np.array(resized_image.GetSpacing())
            
            # Конвертируем в numpy массив
            image_array = sitk.GetArrayFromImage(resized_image)  # (z, y, x)
            
            # Применяем маску лёгких если доступна
            if lung_mask_path.exists():
                lung_mask_itk = sitk.ReadImage(str(lung_mask_path))
                lung_mask_resampled = self._resample_image(
                    lung_mask_itk, self.target_spacing
                )
                lung_mask_resized = self._resize_image(
                    lung_mask_resampled, target_size_xyz
                )
                lung_mask_array = sitk.GetArrayFromImage(lung_mask_resized)
                
                # Применяем маску
                image_array = utils.apply_lung_mask(image_array, lung_mask_array)
            
            # Нормализация
            normalized_image = self._normalize_hu(image_array)
            
            # Получаем аннотации для этой серии
            series_annotations = self.annotations[
                self.annotations['seriesuid'] == series_uid
            ]
            
            # Создаём список узелков
            nodules = []
            for _, row in series_annotations.iterrows():
                nodules.append({
                    'coordX': row['coordX'],
                    'coordY': row['coordY'],
                    'coordZ': row['coordZ'],
                    'diameter_mm': row['diameter_mm']
                })
            
            # Обновляем координаты узелков
            updated_nodules = self._update_annotations_for_new_params(
                nodules, original_origin, original_spacing,
                new_origin, new_spacing
            )
            
            # Создаём маску узелков
            nodule_mask = utils.create_combined_nodule_mask(
                self.target_size, updated_nodules, new_origin, new_spacing
            )
            
            # Сохраняем обработанное изображение
            output_image_path = self.output_images_dir / f"{series_uid}.npy"
            np.save(output_image_path, normalized_image)
            
            # Сохраняем маску
            output_mask_path = self.output_masks_dir / f"{series_uid}.npy"
            np.save(output_mask_path, nodule_mask)
            
            # Сохраняем метаданные
            metadata = {
                'origin': new_origin.tolist(),
                'spacing': new_spacing.tolist(),
                'nodules': updated_nodules
            }
            metadata_path = self.output_dir / f"{series_uid}_metadata.npy"
            np.save(metadata_path, metadata)
            
            return True
            
        except Exception as e:
            print(f"Error processing {series_uid}: {str(e)}")
            return False
    
    def process_all(self):
        """Обработка всех сканов в датасете."""
        unique_series = self.annotations['seriesuid'].unique()
        
        print(f"Processing {len(unique_series)} scans...")
        
        successful = 0
        for series_uid in tqdm(unique_series):
            if self.process_single_scan(series_uid):
                successful += 1
        
        print(f"\nProcessing complete: {successful}/{len(unique_series)} successful")
        
        # Сохраняем список обработанных файлов
        processed_list = [
            f"{series_uid}.npy" 
            for series_uid in unique_series 
            if (self.output_images_dir / f"{series_uid}.npy").exists()
        ]
        
        list_path = self.output_dir / 'processed_files.txt'
        with open(list_path, 'w') as f:
            f.write('\n'.join(processed_list))
        
        print(f"Processed files list saved to: {list_path}")


if __name__ == '__main__':
    # Пример использования
    preprocessor = LUNA16Preprocessor(
        raw_data_dir='data/LUNA16/raw',
        annotations_path='data/LUNA16/annotations.csv',
        lung_masks_dir='data/LUNA16/seg-lungs-LUNA16',
        output_dir='data/LUNA16/processed',
        target_spacing=(1.0, 1.0, 1.0),
        target_size=(64, 256, 256),
        hu_min=-1000.0,
        hu_max=400.0
    )
    
    preprocessor.process_all()