"""
Скрипт для инференса: получение и сохранение предсказанных масок узелков.
Также извлекает координаты обнаруженных узелков.
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from scipy import ndimage
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import create_model
import utils


class NoduleInferencer:
    """Класс для инференса и извлечения узелков."""
    
    def __init__(self,
                 checkpoint_path: str,
                 output_dir: str,
                 device: str = 'cuda',
                 threshold: float = 0.5,
                 min_nodule_size: int = 10):
        """
        Args:
            checkpoint_path: путь к чекпоинту модели
            output_dir: директория для сохранения результатов
            device: устройство
            threshold: порог для бинаризации предсказаний
            min_nodule_size: минимальный размер узелка в вокселях
        """
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.device = device
        self.threshold = threshold
        self.min_nodule_size = min_nodule_size
        
        # Создаём директории
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir = self.output_dir / 'predicted_masks'
        self.visualizations_dir = self.output_dir / 'visualizations'
        self.masks_dir.mkdir(exist_ok=True)
        self.visualizations_dir.mkdir(exist_ok=True)
        
        # Загрузка модели
        self.model = self._load_model()
        
    def _load_model(self) -> torch.nn.Module:
        """Загрузка модели из чекпоинта."""
        print(f"Loading model from {self.checkpoint_path}")
        
        model = create_model(device=self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Предсказание маски для изображения.
        
        Args:
            image: входное изображение (D, H, W)
            
        Returns:
            mask: предсказанная маска (D, H, W)
        """
        # Подготовка данных
        image_tensor = torch.from_numpy(image[np.newaxis, np.newaxis, ...]).float()
        image_tensor = image_tensor.to(self.device)
        
        # Предсказание
        output = self.model(image_tensor)
        pred = torch.sigmoid(output) > self.threshold
        
        # Конвертация обратно в numpy
        mask = pred.cpu().numpy()[0, 0, ...]
        
        return mask.astype(np.uint8)
    
    def extract_nodule_candidates(self, 
                                 mask: np.ndarray,
                                 origin: np.ndarray,
                                 spacing: np.ndarray) -> List[Dict]:
        """
        Извлечение кандидатов узелков из маски.
        
        Args:
            mask: бинарная маска (D, H, W)
            origin: origin изображения (x, y, z)
            spacing: spacing (x, y, z)
            
        Returns:
            nodules: список словарей с информацией об узелках
        """
        # Connected components analysis
        labeled_mask, num_features = ndimage.label(mask)
        
        nodules = []
        
        for label in range(1, num_features + 1):
            component = (labeled_mask == label)
            size = component.sum()
            
            # Фильтруем маленькие компоненты
            if size < self.min_nodule_size:
                continue
            
            # Вычисляем центр масс в voxel координатах
            center_voxel = ndimage.center_of_mass(component)
            center_voxel = np.array([center_voxel[2], center_voxel[1], center_voxel[0]])  # (x, y, z)
            
            # Конвертируем в мировые координаты
            center_world = utils.voxel_to_world_coords(center_voxel, origin, spacing)
            
            # Оцениваем диаметр
            bbox = utils.compute_bbox_from_mask(component, margin=(0, 0, 0))
            if bbox is not None:
                z_size = (bbox[1] - bbox[0]) * spacing[2]
                y_size = (bbox[3] - bbox[2]) * spacing[1]
                x_size = (bbox[5] - bbox[4]) * spacing[0]
                diameter = np.mean([z_size, y_size, x_size])
            else:
                diameter = 0.0
            
            nodules.append({
                'center_world': center_world.tolist(),
                'center_voxel': center_voxel.tolist(),
                'diameter_mm': float(diameter),
                'volume_voxels': int(size),
                'bbox': bbox
            })
        
        return nodules
    
    def visualize_prediction(self,
                           image: np.ndarray,
                           mask_gt: np.ndarray,
                           mask_pred: np.ndarray,
                           nodules: List[Dict],
                           filename: str):
        """
        Визуализация предсказания.
        
        Args:
            image: исходное изображение (D, H, W)
            mask_gt: ground truth маска (D, H, W)
            mask_pred: предсказанная маска (D, H, W)
            nodules: список обнаруженных узелков
            filename: имя файла для сохранения
        """
        # Выбираем срез с максимальным перекрытием
        overlap = mask_gt * mask_pred
        slice_sums = overlap.sum(axis=(1, 2))
        
        if slice_sums.max() > 0:
            best_slice = slice_sums.argmax()
        else:
            # Если нет перекрытия, выбираем срез с максимальной маской GT
            gt_sums = mask_gt.sum(axis=(1, 2))
            best_slice = gt_sums.argmax() if gt_sums.max() > 0 else image.shape[0] // 2
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Исходное изображение
        axes[0].imshow(image[best_slice], cmap='gray')
        axes[0].set_title('CT Slice')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(image[best_slice], cmap='gray')
        axes[1].imshow(mask_gt[best_slice], cmap='Reds', alpha=0.5)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Предсказание
        axes[2].imshow(image[best_slice], cmap='gray')
        axes[2].imshow(mask_pred[best_slice], cmap='Greens', alpha=0.5)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Наложение
        axes[3].imshow(image[best_slice], cmap='gray')
        axes[3].imshow(mask_gt[best_slice], cmap='Reds', alpha=0.3)
        axes[3].imshow(mask_pred[best_slice], cmap='Greens', alpha=0.3)
        axes[3].set_title('Overlay (Red=GT, Green=Pred)')
        axes[3].axis('off')
        
        # Добавляем информацию об узелках
        info_text = f"Detected nodules: {len(nodules)}\n"
        info_text += f"Slice: {best_slice}/{image.shape[0]}"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=12)
        
        plt.tight_layout()
        
        save_path = self.visualizations_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def infer_single(self,
                    image_path: str,
                    mask_path: str,
                    metadata_path: str,
                    series_uid: str):
        """
        Инференс на одном примере.
        
        Args:
            image_path: путь к изображению
            mask_path: путь к GT маске
            metadata_path: путь к метаданным
            series_uid: ID серии
        """
        # Загрузка данных
        image = np.load(image_path)
        mask_gt = np.load(mask_path)
        metadata = np.load(metadata_path, allow_pickle=True).item()
        
        origin = np.array(metadata['origin'])
        spacing = np.array(metadata['spacing'])
        
        # Предсказание
        mask_pred = self.predict(image)
        
        # Извлечение узелков
        nodules = self.extract_nodule_candidates(mask_pred, origin, spacing)
        
        # Сохранение предсказанной маски
        output_mask_path = self.masks_dir / f"{series_uid}_pred.npy"
        np.save(output_mask_path, mask_pred)
        
        # Сохранение информации об узелках
        nodules_path = self.output_dir / f"{series_uid}_nodules.npy"
        np.save(nodules_path, nodules)
        
        # Визуализация
        vis_filename = f"{series_uid}_prediction.png"
        self.visualize_prediction(image, mask_gt, mask_pred, nodules, vis_filename)
        
        return nodules
    
    def infer_dataset(self, data_dir: str):
        """
        Инференс на всём датасете.
        
        Args:
            data_dir: директория с обработанными данными
        """
        data_dir = Path(data_dir)
        images_dir = data_dir / 'images'
        masks_dir = data_dir / 'masks'
        
        # Читаем список файлов
        file_list_path = data_dir / 'processed_files.txt'
        with open(file_list_path, 'r') as f:
            all_files = [line.strip() for line in f.readlines()]
        
        # Берём validation set (последние 20%)
        split_idx = int(len(all_files) * 0.8)
        val_files = all_files[split_idx:]
        
        print(f"Running inference on {len(val_files)} samples...")
        
        all_nodules = {}
        
        for filename in tqdm(val_files):
            series_uid = filename.replace('.npy', '')
            
            image_path = images_dir / filename
            mask_path = masks_dir / filename
            metadata_path = data_dir / f"{series_uid}_metadata.npy"
            
            if not all([image_path.exists(), mask_path.exists(), metadata_path.exists()]):
                print(f"Warning: Missing files for {series_uid}")
                continue
            
            try:
                nodules = self.infer_single(
                    str(image_path),
                    str(mask_path),
                    str(metadata_path),
                    series_uid
                )
                all_nodules[series_uid] = nodules
                
            except Exception as e:
                print(f"Error processing {series_uid}: {str(e)}")
        
        # Сохраняем сводку
        summary_path = self.output_dir / 'inference_summary.npy'
        np.save(summary_path, all_nodules)
        
        # Печатаем статистику
        self._print_inference_summary(all_nodules)
    
    def _print_inference_summary(self, all_nodules: Dict):
        """Печать сводки инференса."""
        total_scans = len(all_nodules)
        total_nodules = sum(len(nodules) for nodules in all_nodules.values())
        
        print("\n" + "="*60)
        print("INFERENCE SUMMARY")
        print("="*60)
        print(f"Total scans processed: {total_scans}")
        print(f"Total nodules detected: {total_nodules}")
        print(f"Average nodules per scan: {total_nodules/total_scans:.2f}")
        
        # Распределение по размерам
        all_diameters = []
        for nodules in all_nodules.values():
            all_diameters.extend([n['diameter_mm'] for n in nodules])
        
        if all_diameters:
            print(f"\nNodule diameter statistics (mm):")
            print(f"  Mean: {np.mean(all_diameters):.2f}")
            print(f"  Std: {np.std(all_diameters):.2f}")
            print(f"  Min: {np.min(all_diameters):.2f}")
            print(f"  Max: {np.max(all_diameters):.2f}")
        
        print("="*60 + "\n")


if __name__ == '__main__':
    inferencer = NoduleInferencer(
        checkpoint_path='experiments/luna16_unet/checkpoints/best_model.pth',
        output_dir='experiments/luna16_unet/inference',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        threshold=0.5,
        min_nodule_size=10
    )
    
    inferencer.infer_dataset(data_dir='data/LUNA16/processed')