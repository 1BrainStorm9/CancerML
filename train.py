"""
train.py — обучение 3D U-Net на вашем LUNA16-like датасете
с проверкой размеров тензоров и подсчетом метрик
по позитивным и негативным элементам маски.
"""

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from model import create_model, DiceBCELoss, count_parameters
from dataset import get_dataloaders
import utils  # здесь должны быть функции dice_coefficient, sensitivity_score, precision_score, save_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, data_dir, output_dir,
                 num_epochs=20, batch_size=2, learning_rate=1e-4,
                 weight_decay=1e-5, num_workers=4,
                 mixed_precision=True, checkpoint_interval=5):

        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        self.checkpoint_interval = checkpoint_interval

        # Директории для чекпоинтов и логов
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(self.output_dir / f"logs_{timestamp}")

        # Настройка компонентов
        self._setup()

    def _setup(self):
        print("Setting up training...")

        # Загрузчики данных с балансом классов
        self.train_loader, self.val_loader = get_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            cache_train=False,
            cache_val=False
        )
        

        # Модель
        self.model = create_model(device=DEVICE)
        print(f"Model parameters: {count_parameters(self.model):,}")

        # Loss
        self.criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)

        # Оптимизатор и scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=1e-6
        )

        # Mixed precision
        if DEVICE.type == "cpu":
            self.mixed_precision = False
            self.scaler = None
        else:
            from torch.cuda.amp import GradScaler, autocast
            self.scaler = GradScaler() if self.mixed_precision else None

        self.best_val_dice = 0.0

        # В Trainer._setup() после get_dataloaders
        pos_batches, neg_batches = 0, 0
        for batch in self.train_loader:
            if batch["mask"].sum() > 0:
                pos_batches += 1
            else:
                neg_batches += 1
        print(f"Train batches with positive masks: {pos_batches}/{pos_batches + neg_batches}")


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            # Проверка размера: (B, C, D, H, W)
            assert images.shape[1:] == (1, 64, 256, 256), f"Bad image shape: {images.shape}"

            self.optimizer.zero_grad()
            if self.mixed_precision:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            self.writer.add_scalar("Train/Loss", loss.item(), epoch * len(self.train_loader) + batch_idx)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_dice = 0.0
        total_sens = 0.0
        total_prec = 0.0

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            outputs = self.model(images)
            preds = torch.sigmoid(outputs) > 0.5

            # Подсчет по позитивным и негативным элементам
            dice = utils.dice_coefficient(preds.float(), masks)
            sens = utils.sensitivity_score(preds.float(), masks)
            prec = utils.precision_score(preds.float(), masks)

            total_dice += dice
            total_sens += sens
            total_prec += prec

        metrics = {
            "dice": total_dice / len(self.val_loader),
            "sensitivity": total_sens / len(self.val_loader),
            "precision": total_prec / len(self.val_loader)
        }

        for k, v in metrics.items():
            self.writer.add_scalar(f"Val/{k}", v, epoch)

        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        utils.save_checkpoint(self.model, self.optimizer, epoch, metrics, str(path), scheduler=self.scheduler)
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            utils.save_checkpoint(self.model, self.optimizer, epoch, metrics, str(best_path), scheduler=self.scheduler)
            print(f"Saved best model with Dice={metrics['dice']:.4f}")

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            self.scheduler.step()

            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Dice={val_metrics['dice']:.4f}")

            is_best = val_metrics["dice"] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics["dice"]

            if (epoch + 1) % self.checkpoint_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)


if __name__ == "__main__":
    trainer = Trainer(
        data_dir="data/LUNA16/processed",
        output_dir="experiments/luna16_unet",
        num_epochs=20,
        batch_size=2,
        learning_rate=1e-4,
        num_workers=4,
        mixed_precision=True,
        checkpoint_interval=5
    )
    trainer.train()
