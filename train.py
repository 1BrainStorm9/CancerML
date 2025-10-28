"""
Training script for LUNA16 3D nodule segmentation
-------------------------------------------------
Architecture: Attention-ResUNet3D (see model.py)
Loss: SoftDice + BCE
Scheduler: OneCycleLR
GPU: RTX 4070 Ti (mixed precision)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch

from dataset import get_dataloaders
from model import create_model, count_parameters


# ======================================================
# Config
# ======================================================
class CFG:
    train_img_dir = "data/LUNA16/processed/images/positive"
    train_mask_dir = "data/LUNA16/processed/masks/positive"
    val_img_dir   = "data/LUNA16/processed/images/negative"
    val_mask_dir  = "data/LUNA16/processed/masks/negative"

    crop_shape = (64, 256, 256)
    batch_size = 1
    num_epochs = 40
    lr = 1e-4
    weight_decay = 1e-5

    num_workers = 0  # для Windows лучше ставить 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)


# ======================================================
# Loss Function
# ======================================================
bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss(sigmoid=True, squared_pred=True, smooth_nr=1e-5, smooth_dr=1e-5)

def combined_loss(pred, target):
    return 0.8 * dice_loss(pred, target) + 0.2 * bce_loss(pred, target)


# ======================================================
# Train / Validate
# ======================================================
def train_one_epoch(model, loader, optimizer, scaler, epoch, scheduler=None):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for images, masks in pbar:
        images, masks = images.to(CFG.device), masks.to(CFG.device)

        optimizer.zero_grad(set_to_none=True)
        # Use new torch.amp.autocast signature
        with torch.amp.autocast(device_type=CFG.device, enabled=(CFG.device == "cuda")):
            outputs = model(images)
            loss = combined_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(model, loader, epoch):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    val_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    for images, masks in pbar:
        images, masks = images.to(CFG.device), masks.to(CFG.device)

        with torch.amp.autocast(device_type=CFG.device, enabled=(CFG.device == "cuda")):
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

        preds_list = decollate_batch(preds)
        masks_list = decollate_batch(masks)
        dice_metric(y_pred=preds_list, y=masks_list)

    dice_mean = dice_metric.aggregate().item()
    dice_metric.reset()

    avg_val_loss = val_loss / len(loader)
    return avg_val_loss, dice_mean


# ======================================================
# Main
# ======================================================
def main():
    print(f"🚀 Training on {CFG.device.upper()}")

    train_loader, val_loader = get_dataloaders(
        CFG.train_img_dir, CFG.train_mask_dir,
        CFG.val_img_dir, CFG.val_mask_dir,
        batch_size=CFG.batch_size,
        crop_shape=CFG.crop_shape,
        num_workers=CFG.num_workers,
        augment=True
    )

    model = create_model(device=CFG.device, dropout=0.2)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CFG.lr,
        epochs=CFG.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=10
    )

    # Use new GradScaler signature for mixed precision
    scaler = torch.amp.GradScaler(enabled=(CFG.device == "cuda"))
    best_dice = 0.0

    for epoch in range(CFG.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch, scheduler)
        val_loss, dice_score = validate(model, val_loader, epoch)

        print(f"📘 Epoch [{epoch+1}/{CFG.num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Dice: {dice_score:.4f}")

        # Save best checkpoint
        if dice_score > best_dice:
            best_dice = dice_score
            save_path = os.path.join(CFG.checkpoint_dir,
                                     f"best_model_epoch{epoch+1}_dice{dice_score:.4f}.pth")
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "dice": dice_score,
                        "epoch": epoch + 1}, save_path)
            print(f"💾 Saved best model to {save_path}")


# ======================================================
# Entry point
# ======================================================
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
