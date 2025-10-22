"""
Вспомогательные функции для работы с LUNA16 датасетом.
Включает загрузку .mhd/.raw файлов, конвертацию координат,
генерацию масок узелков, расчёт метрик.
"""

import os
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import binary_dilation, generate_binary_structure
from pathlib import Path
from typing import Tuple, List, Dict, Optional


def load_mhd_image(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загрузка .mhd/.raw файла с помощью SimpleITK.
    
    Args:
        filepath: путь к .mhd файлу
        
    Returns:
        image: numpy массив с интенсивностями (z, y, x)
        origin: мировые координаты начала (x, y, z)
        spacing: расстояние между вокселями (x, y, z)
    """
    itk_image = sitk.ReadImage(filepath)
    image = sitk.GetArrayFromImage(itk_image)  # (z, y, x)
    origin = np.array(itk_image.GetOrigin())  # (x, y, z)
    spacing = np.array(itk_image.GetSpacing())  # (x, y, z)
    
    return image, origin, spacing


def save_mhd_image(image: np.ndarray, filepath: str, origin: np.ndarray, 
                   spacing: np.ndarray):
    """
    Сохранение массива как .mhd/.raw файл.
    
    Args:
        image: numpy массив (z, y, x)
        filepath: путь для сохранения
        origin: мировые координаты начала (x, y, z)
        spacing: расстояние между вокселями (x, y, z)
    """
    itk_image = sitk.GetImageFromArray(image)
    itk_image.SetOrigin(origin.tolist())
    itk_image.SetSpacing(spacing.tolist())
    sitk.WriteImage(itk_image, filepath)


def world_to_voxel_coords(world_coords: np.ndarray, origin: np.ndarray, 
                          spacing: np.ndarray) -> np.ndarray:
    """
    Конвертация мировых координат в voxel координаты.
    
    Args:
        world_coords: мировые координаты (x, y, z) или (N, 3)
        origin: начало координат (x, y, z)
        spacing: spacing (x, y, z)
        
    Returns:
        voxel_coords: voxel координаты (x, y, z) или (N, 3)
    """
    voxel_coords = (world_coords - origin) / spacing
    return voxel_coords


def voxel_to_world_coords(voxel_coords: np.ndarray, origin: np.ndarray,
                         spacing: np.ndarray) -> np.ndarray:
    """
    Конвертация voxel координат в мировые координаты.
    
    Args:
        voxel_coords: voxel координаты (x, y, z) или (N, 3)
        origin: начало координат (x, y, z)
        spacing: spacing (x, y, z)
        
    Returns:
        world_coords: мировые координаты (x, y, z) или (N, 3)
    """
    world_coords = voxel_coords * spacing + origin
    return world_coords


def generate_nodule_mask(shape: Tuple[int, int, int], 
                        center_voxel: np.ndarray,
                        diameter_mm: float,
                        spacing: np.ndarray) -> np.ndarray:
    """
    Генерация сферической маски узелка.
    
    Args:
        shape: размер маски (z, y, x)
        center_voxel: центр узелка в voxel координатах (x, y, z)
        diameter_mm: диаметр узелка в мм
        spacing: spacing (x, y, z)
        
    Returns:
        mask: бинарная маска (z, y, x)
    """
    mask = np.zeros(shape, dtype=np.float32)
    
    # Радиус в вокселях для каждого измерения
    radius_voxels = (diameter_mm / 2.0) / spacing  # (x, y, z)
    
    # Создаём сетку координат
    z_coords, y_coords, x_coords = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
    
    # Центр в формате (x, y, z) -> преобразуем в (z, y, x) для индексации
    center_z = center_voxel[2]
    center_y = center_voxel[1]
    center_x = center_voxel[0]
    
    # Вычисляем расстояние с учётом анизотропного spacing
    distance_sq = ((x_coords - center_x) / radius_voxels[0]) ** 2 + \
                  ((y_coords - center_y) / radius_voxels[1]) ** 2 + \
                  ((z_coords - center_z) / radius_voxels[2]) ** 2
    
    mask[distance_sq <= 1.0] = 1.0
    
    return mask


def create_combined_nodule_mask(shape: Tuple[int, int, int],
                               nodules: List[Dict],
                               origin: np.ndarray,
                               spacing: np.ndarray) -> np.ndarray:
    """
    Создание комбинированной маски для всех узелков пациента.
    
    Args:
        shape: размер маски (z, y, x)
        nodules: список словарей с информацией об узелках
                 каждый содержит 'coordX', 'coordY', 'coordZ', 'diameter_mm'
        origin: origin изображения (x, y, z)
        spacing: spacing (x, y, z)
        
    Returns:
        combined_mask: бинарная маска всех узелков (z, y, x)
    """
    combined_mask = np.zeros(shape, dtype=np.float32)
    
    for nodule in nodules:
        world_coord = np.array([
            nodule['coordX'],
            nodule['coordY'],
            nodule['coordZ']
        ])
        
        voxel_coord = world_to_voxel_coords(world_coord, origin, spacing)
        
        # Проверка, что узелок в пределах изображения
        if (0 <= voxel_coord[0] < shape[2] and
            0 <= voxel_coord[1] < shape[1] and
            0 <= voxel_coord[2] < shape[0]):
            
            nodule_mask = generate_nodule_mask(
                shape, voxel_coord, nodule['diameter_mm'], spacing
            )
            combined_mask = np.maximum(combined_mask, nodule_mask)
    
    return combined_mask


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, 
                    smooth: float = 1e-6) -> float:
    """
    Вычисление Dice coefficient.
    
    Args:
        pred: предсказанная маска (B, C, D, H, W) или (D, H, W)
        target: ground truth маска
        smooth: smoothing factor
        
    Returns:
        dice: Dice coefficient
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def sensitivity_score(pred: torch.Tensor, target: torch.Tensor,
                     smooth: float = 1e-6) -> float:
    """
    Вычисление Sensitivity (Recall, True Positive Rate).
    
    Args:
        pred: предсказанная маска
        target: ground truth маска
        smooth: smoothing factor
        
    Returns:
        sensitivity: чувствительность
    """
    pred = pred.flatten()
    target = target.flatten()
    
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    
    sensitivity = (true_positive + smooth) / (actual_positive + smooth)
    return sensitivity.item()


def precision_score(pred: torch.Tensor, target: torch.Tensor,
                   smooth: float = 1e-6) -> float:
    """
    Вычисление Precision (Positive Predictive Value).
    
    Args:
        pred: предсказанная маска
        target: ground truth маска
        smooth: smoothing factor
        
    Returns:
        precision: точность
    """
    pred = pred.flatten()
    target = target.flatten()
    
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision.item()


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, metrics: Dict,
                   filepath: str, scheduler=None):
    """
    Сохранение чекпоинта модели.
    
    Args:
        model: модель
        optimizer: оптимизатор
        epoch: номер эпохи
        loss: значение loss
        metrics: словарь с метриками
        filepath: путь для сохранения
        scheduler: планировщик lr (опционально)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler=None) -> int:
    """
    Загрузка чекпоинта модели.
    
    Args:
        filepath: путь к чекпоинту
        model: модель
        optimizer: оптимизатор (опционально)
        scheduler: планировщик lr (опционально)
        
    Returns:
        epoch: номер эпохи из чекпоинта
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from: {filepath}, epoch: {epoch}")
    
    return epoch


def apply_lung_mask(image: np.ndarray, lung_mask: np.ndarray,
                   margin: int = 10) -> np.ndarray:
    """
    Применение маски лёгких с небольшим расширением границ.
    
    Args:
        image: КТ изображение (z, y, x)
        lung_mask: маска лёгких (z, y, x)
        margin: количество вокселей для расширения маски
        
    Returns:
        masked_image: изображение с применённой маской
    """
    # Расширяем маску лёгких
    struct = generate_binary_structure(3, 1)
    dilated_mask = binary_dilation(lung_mask > 0, structure=struct, 
                                   iterations=margin)
    
    # Применяем маску (всё вне лёгких -> -1000 HU)
    masked_image = np.where(dilated_mask, image, -1000)
    
    return masked_image


def compute_bbox_from_mask(mask: np.ndarray, 
                          margin: Tuple[int, int, int] = (5, 5, 5)) -> Tuple:
    """
    Вычисление bounding box из маски с отступами.
    
    Args:
        mask: бинарная маска (z, y, x)
        margin: отступы (z, y, x)
        
    Returns:
        bbox: (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    nonzero = np.nonzero(mask)
    
    if len(nonzero[0]) == 0:
        return None
    
    z_min, z_max = nonzero[0].min(), nonzero[0].max()
    y_min, y_max = nonzero[1].min(), nonzero[1].max()
    x_min, x_max = nonzero[2].min(), nonzero[2].max()
    
    # Добавляем отступы
    z_min = max(0, z_min - margin[0])
    z_max = min(mask.shape[0], z_max + margin[0] + 1)
    y_min = max(0, y_min - margin[1])
    y_max = min(mask.shape[1], y_max + margin[1] + 1)
    x_min = max(0, x_min - margin[2])
    x_max = min(mask.shape[2], x_max + margin[2] + 1)
    
    return (z_min, z_max, y_min, y_max, x_min, x_max)