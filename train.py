"""
Скрипт для обучения 3D U-Net модели на LUNA16 датасете.
Включает mixed precision training, cosine annealing scheduler,
логирование и сохранение чекпоинтов.
"""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

from model import create_model, DiceBCELoss, count_parameters
from dataset import get_dataloaders
import utils


class Trainer:
    """Класс для обучения модели."""
    
    def __init__(self,
                 data_dir: str,
                 output_dir: str,
                 num_epochs: int = 100,
                 batch_size: int = 2,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 num_workers: int = 4,
                 device: str = 'cuda',
                 mixed_precision: bool = True,
                 checkpoint_interval: int = 5):
        """
        Args:
            data_dir: директория с обработанными данными
            output_dir: директория для сохранения результатов
            num_epochs: количество эпох
            batch_size: размер батча
            learning_rate: learning rate
            weight_decay: weight decay для AdamW
            num_workers: количество workers для DataLoader
            device: устройство для обучения
            mixed_precision: использовать ли mixed precision
            checkpoint_interval: интервал сохранения чекпоинтов
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.device = device
        self.mixed_precision = mixed_precision
        self.checkpoint_interval = checkpoint_interval
        
        # Создаём директории
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Логирование
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = self.output_dir / f'logs_{timestamp}'
        self.writer = SummaryWriter(self.log_dir)
        
        # Инициализация компонентов
        self._setup()
        
    def _setup(self):
        """Инициализация модели, оптимизатора, загрузчиков данных."""
        print("Setting up training...")
        
        # Проверка CUDA и автоматическая настройка
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("\nWARNING: CUDA requested but not available. Switching to CPU.")
            print("Training on CPU will be VERY slow (hours -> days).")
            print("Consider using Google Colab or other GPU platforms.\n")
            self.device = 'cpu'
            self.mixed_precision = False  # Mixed precision не работает на CPU
        
        # Отключаем pin_memory для CPU
        use_pin_memory = (self.device == 'cuda')
        
        # DataLoaders
        self.train_loader, self.val_loader = get_dataloaders(
            data_dir=self.data_dir,
            train_split=0.8,
            batch_size=self.batch_size,
            num_workers=self.num_workers if self.device == 'cuda' else 0,  # CPU: num_workers=0
            cache_train=False,
            cache_val=False
        )
        
        # Обновляем pin_memory в DataLoader
        if not use_pin_memory:
            for loader in [self.train_loader, self.val_loader]:
                loader.pin_memory = False
        
        # Модель
        self.model = create_model(device=self.device)
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Loss
        self.criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Метрики
        self.best_val_dice = 0.0
        self.start_epoch = 0
        
    def train_epoch(self, epoch: int) -> float:
        """
        Обучение одной эпохи.
        
        Args:
            epoch: номер эпохи
            
        Returns:
            avg_loss: средний loss за эпоху
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass с mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Обновление прогресс-бара
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Логирование
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """
        Валидация модели.
        
        Args:
            epoch: номер эпохи
            
        Returns:
            metrics: словарь с метриками
        """
        self.model.eval()
        
        total_dice = 0.0
        total_sensitivity = 0.0
        total_precision = 0.0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
            else:
                outputs = self.model(images)
            
            # Применяем sigmoid и threshold
            preds = torch.sigmoid(outputs) > 0.5
            
            # Вычисляем метрики
            dice = utils.dice_coefficient(preds.float(), masks)
            sensitivity = utils.sensitivity_score(preds.float(), masks)
            precision = utils.precision_score(preds.float(), masks)
            
            total_dice += dice
            total_sensitivity += sensitivity
            total_precision += precision
            
            pbar.set_postfix({
                'dice': dice,
                'sens': sensitivity,
                'prec': precision
            })
        
        # Средние метрики
        metrics = {
            'dice': total_dice / len(self.val_loader),
            'sensitivity': total_sensitivity / len(self.val_loader),
            'precision': total_precision / len(self.val_loader)
        }
        
        # Логирование
        for key, value in metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Сохранение чекпоинта."""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        utils.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=0.0,
            metrics=metrics,
            filepath=str(checkpoint_path),
            scheduler=self.scheduler
        )
        
        # Сохраняем лучшую модель
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            utils.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=0.0,
                metrics=metrics,
                filepath=str(best_path),
                scheduler=self.scheduler
            )
            print(f"New best model saved! Dice: {metrics['dice']:.4f}")
    
    def train(self):
        """Основной цикл обучения."""
        print("\n" + "="*50)
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print("="*50 + "\n")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # Обучение
            train_loss = self.train_epoch(epoch)
            
            # Валидация
            val_metrics = self.validate(epoch)
            
            # Scheduler step
            self.scheduler.step()
            
            # Логирование
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Dice: {val_metrics['dice']:.4f}")
            print(f"Val Sensitivity: {val_metrics['sensitivity']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}\n")
            
            # Сохранение чекпоинта
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
            
            if (epoch + 1) % self.checkpoint_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        print("="*50 + "\n")
        
        self.writer.close()


if __name__ == '__main__':
    # Параметры обучения
    trainer = Trainer(
        data_dir='data/LUNA16/processed',
        output_dir='experiments/luna16_unet',
        num_epochs=100,
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=1e-5,
        num_workers=4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision=True,
        checkpoint_interval=5
    )
    
    trainer.train()