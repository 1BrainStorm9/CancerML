"""
Скрипт для детальной валидации модели на тестовых данных.
Вычисляет метрики для каждого пациента и общие статистики.
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from model import create_model
from dataset import LUNA16Dataset
from torch.utils.data import DataLoader
import utils


class Validator:
    """Класс для валидации модели."""
    
    def __init__(self,
                 data_dir: str,
                 checkpoint_path: str,
                 output_dir: str,
                 device: str = 'cuda',
                 threshold: float = 0.5):
        """
        Args:
            data_dir: директория с обработанными данными
            checkpoint_path: путь к чекпоинту модели
            output_dir: директория для сохранения результатов
            device: устройство
            threshold: порог для бинаризации предсказаний
        """
        self.data_dir = Path(data_dir)
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.device = device
        self.threshold = threshold
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
    def validate_single(self, image: torch.Tensor, 
                       mask: torch.Tensor) -> Dict[str, float]:
        """
        Валидация на одном примере.
        
        Args:
            image: входное изображение (1, 1, D, H, W)
            mask: ground truth маска (1, 1, D, H, W)
            
        Returns:
            metrics: словарь с метриками
        """
        image = image.to(self.device)
        mask = mask.to(self.device)
        
        # Предсказание
        output = self.model(image)
        pred = torch.sigmoid(output) > self.threshold
        
        # Вычисление метрик
        metrics = {
            'dice': utils.dice_coefficient(pred.float(), mask),
            'sensitivity': utils.sensitivity_score(pred.float(), mask),
            'precision': utils.precision_score(pred.float(), mask)
        }
        
        # Recall (то же что sensitivity)
        metrics['recall'] = metrics['sensitivity']
        
        return metrics, pred.cpu()
    
    def validate_dataset(self, file_list: List[str]) -> pd.DataFrame:
        """
        Валидация на списке файлов.
        
        Args:
            file_list: список файлов для валидации
            
        Returns:
            results_df: DataFrame с результатами
        """
        dataset = LUNA16Dataset(
            data_dir=str(self.data_dir),
            file_list=file_list,
            augmentation=False,
            cache=False
        )
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        
        results = []
        
        print(f"Validating on {len(file_list)} samples...")
        
        for idx, batch in enumerate(tqdm(loader)):
            filename = file_list[idx]
            
            image = batch['image']
            mask = batch['mask']
            
            metrics, pred = self.validate_single(image, mask)
            
            results.append({
                'filename': filename,
                'dice': metrics['dice'],
                'sensitivity': metrics['sensitivity'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            })
        
        results_df = pd.DataFrame(results)
        
        # Сохраняем результаты
        results_path = self.output_dir / 'validation_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        
        return results_df
    
    def print_statistics(self, results_df: pd.DataFrame):
        """Печать статистики."""
        print("\n" + "="*60)
        print("VALIDATION STATISTICS")
        print("="*60)
        
        print(f"\nNumber of samples: {len(results_df)}")
        print("\nMetrics (mean ± std):")
        
        for metric in ['dice', 'sensitivity', 'precision', 'recall']:
            mean = results_df[metric].mean()
            std = results_df[metric].std()
            print(f"  {metric.capitalize():12s}: {mean:.4f} ± {std:.4f}")
        
        print("\nBest performances:")
        for metric in ['dice', 'sensitivity', 'precision']:
            best_idx = results_df[metric].idxmax()
            best_file = results_df.loc[best_idx, 'filename']
            best_value = results_df.loc[best_idx, metric]
            print(f"  {metric.capitalize():12s}: {best_value:.4f} ({best_file})")
        
        print("\nWorst performances:")
        for metric in ['dice', 'sensitivity', 'precision']:
            worst_idx = results_df[metric].idxmin()
            worst_file = results_df.loc[worst_idx, 'filename']
            worst_value = results_df.loc[worst_idx, metric]
            print(f"  {metric.capitalize():12s}: {worst_value:.4f} ({worst_file})")
        
        print("="*60 + "\n")
    
    def plot_metrics_distribution(self, results_df: pd.DataFrame):
        """Визуализация распределения метрик."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Validation Metrics Distribution', fontsize=16)
        
        metrics = ['dice', 'sensitivity', 'precision', 'recall']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            ax.hist(results_df[metric], bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(results_df[metric].mean(), color='red', 
                      linestyle='--', linewidth=2, label='Mean')
            ax.set_xlabel(metric.capitalize())
            ax.set_ylabel('Count')
            ax.set_title(f'{metric.capitalize()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'metrics_distribution.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Metrics distribution plot saved to {plot_path}")
        plt.close()
    
    def run_validation(self):
        """Запуск полной валидации."""
        # Читаем список файлов
        file_list_path = self.data_dir / 'processed_files.txt'
        with open(file_list_path, 'r') as f:
            all_files = [line.strip() for line in f.readlines()]
        
        # Используем последние 20% для валидации
        split_idx = int(len(all_files) * 0.8)
        val_files = all_files[split_idx:]
        
        print(f"Validation files: {len(val_files)}")
        
        # Валидация
        results_df = self.validate_dataset(val_files)
        
        # Статистика
        self.print_statistics(results_df)
        
        # Визуализация
        self.plot_metrics_distribution(results_df)


if __name__ == '__main__':
    validator = Validator(
        data_dir='data/LUNA16/processed',
        checkpoint_path='experiments/luna16_unet/checkpoints/best_model.pth',
        output_dir='experiments/luna16_unet/validation',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        threshold=0.5
    )
    
    validator.run_validation()