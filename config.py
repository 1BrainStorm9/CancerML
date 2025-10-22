"""
Файл конфигурации для проекта LUNA16 сегментации узелков.
Все основные параметры собраны в одном месте для удобства настройки.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    """Конфигурация данных."""
    
    # Пути к данным
    raw_data_dir: str = 'data/LUNA16/raw'
    annotations_path: str = 'data/LUNA16/annotations.csv'
    lung_masks_dir: str = 'data/LUNA16/seg-lungs-LUNA16'
    processed_dir: str = 'data/LUNA16/processed'
    
    # Параметры предобработки
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (x, y, z) в мм
    target_size: Tuple[int, int, int] = (64, 256, 256)  # (z, y, x) в вокселях
    hu_min: float = -1000.0  # Минимальное HU значение
    hu_max: float = 400.0    # Максимальное HU значение
    
    # Разделение данных
    train_split: float = 0.8
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Конфигурация модели."""
    
    # Архитектура U-Net
    in_channels: int = 1
    out_channels: int = 1
    channels: Tuple[int, ...] = (32, 64, 128, 256, 512)
    strides: Tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2
    dropout: float = 0.1
    
    # Loss функция
    dice_weight: float = 0.5
    bce_weight: float = 0.5
    smooth: float = 1e-6


@dataclass
class TrainingConfig:
    """Конфигурация обучения."""
    
    # Гиперпараметры
    num_epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Оптимизация
    optimizer: str = 'adamw'  # 'adam' или 'adamw'
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    min_lr: float = 1e-6
    
    # Mixed precision
    mixed_precision: bool = True
    
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    
    # Чекпоинты
    checkpoint_interval: int = 5
    save_best_only: bool = False
    
    # Early stopping
    use_early_stopping: bool = False
    patience: int = 20
    min_delta: float = 0.001
    
    # Устройство
    device: str = 'cuda'  # 'cuda' или 'cpu'


@dataclass
class AugmentationConfig:
    """Конфигурация аугментаций."""
    
    # Вероятности аугментаций
    rotate_prob: float = 0.5
    flip_prob: float = 0.5
    noise_prob: float = 0.3
    scale_intensity_prob: float = 0.3
    shift_intensity_prob: float = 0.3
    affine_prob: float = 0.3
    
    # Параметры аугментаций
    noise_std: float = 0.05
    intensity_scale_factor: float = 0.1
    intensity_shift_offset: float = 0.1
    affine_rotate_range: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    affine_scale_range: Tuple[float, float, float] = (0.1, 0.1, 0.1)


@dataclass
class InferenceConfig:
    """Конфигурация инференса."""
    
    # Пути
    checkpoint_path: str = 'experiments/luna16_unet/checkpoints/best_model.pth'
    output_dir: str = 'experiments/luna16_unet/inference'
    
    # Параметры
    threshold: float = 0.5
    min_nodule_size: int = 10  # Минимальный размер узелка в вокселях
    
    # Постобработка
    use_morphology: bool = False
    morphology_iterations: int = 2
    
    # Визуализация
    save_visualizations: bool = True
    visualization_format: str = 'png'  # 'png' или 'jpg'
    
    # Устройство
    device: str = 'cuda'


@dataclass
class ProjectConfig:
    """Общая конфигурация проекта."""
    
    # Название проекта
    project_name: str = 'luna16_nodule_segmentation'
    experiment_name: str = 'unet_baseline'
    
    # Директории
    output_dir: str = 'experiments/luna16_unet'
    
    # Компоненты конфигурации
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Воспроизводимость
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """Создание директорий после инициализации."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.processed_dir).mkdir(parents=True, exist_ok=True)
    
    def print_config(self):
        """Печать конфигурации."""
        print("\n" + "="*70)
        print(f"PROJECT CONFIGURATION: {self.project_name}")
        print("="*70)
        
        print("\n[DATA]")
        print(f"  Processed dir: {self.data.processed_dir}")
        print(f"  Target spacing: {self.data.target_spacing} mm")
        print(f"  Target size: {self.data.target_size}")
        print(f"  HU range: [{self.data.hu_min}, {self.data.hu_max}]")
        print(f"  Train split: {self.data.train_split}")
        
        print("\n[MODEL]")
        print(f"  Architecture: 3D U-Net")
        print(f"  Channels: {self.model.channels}")
        print(f"  Dropout: {self.model.dropout}")
        print(f"  Loss: Dice ({self.model.dice_weight}) + BCE ({self.model.bce_weight})")
        
        print("\n[TRAINING]")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Optimizer: {self.training.optimizer}")
        print(f"  Scheduler: {self.training.scheduler}")
        print(f"  Mixed precision: {self.training.mixed_precision}")
        print(f"  Device: {self.training.device}")
        
        print("\n[AUGMENTATION]")
        print(f"  Rotate prob: {self.augmentation.rotate_prob}")
        print(f"  Flip prob: {self.augmentation.flip_prob}")
        print(f"  Noise prob: {self.augmentation.noise_prob}")
        print(f"  Affine prob: {self.augmentation.affine_prob}")
        
        print("\n[INFERENCE]")
        print(f"  Threshold: {self.inference.threshold}")
        print(f"  Min nodule size: {self.inference.min_nodule_size} voxels")
        print(f"  Checkpoint: {self.inference.checkpoint_path}")
        
        print("="*70 + "\n")


# Создание конфигурации по умолчанию
default_config = ProjectConfig()


# Примеры кастомных конфигураций
def get_small_model_config() -> ProjectConfig:
    """Конфигурация для обучения на GPU с малой памятью."""
    config = ProjectConfig()
    config.experiment_name = 'unet_small'
    config.model.channels = (16, 32, 64, 128, 256)
    config.training.batch_size = 1
    config.data.target_size = (48, 192, 192)
    return config


def get_large_model_config() -> ProjectConfig:
    """Конфигурация для обучения на мощном GPU."""
    config = ProjectConfig()
    config.experiment_name = 'unet_large'
    config.model.channels = (48, 96, 192, 384, 768)
    config.training.batch_size = 4
    config.training.num_epochs = 150
    return config


def get_fast_training_config() -> ProjectConfig:
    """Конфигурация для быстрого обучения (для тестирования)."""
    config = ProjectConfig()
    config.experiment_name = 'unet_fast'
    config.training.num_epochs = 10
    config.training.batch_size = 4
    config.training.checkpoint_interval = 2
    config.data.target_size = (32, 128, 128)
    return config


if __name__ == '__main__':
    # Печать конфигурации по умолчанию
    config = default_config
    config.print_config()
    
    # Примеры кастомных конфигураций
    print("\n\nAVAILABLE CONFIGURATIONS:")
    print("-" * 70)
    
    configs = {
        'default': default_config,
        'small': get_small_model_config(),
        'large': get_large_model_config(),
        'fast': get_fast_training_config()
    }
    
    for name, cfg in configs.items():
        print(f"\n{name.upper()}:")
        print(f"  Batch size: {cfg.training.batch_size}")
        print(f"  Model channels: {cfg.model.channels}")
        print(f"  Target size: {cfg.data.target_size}")
        print(f"  Epochs: {cfg.training.num_epochs}")