"""Training and evaluation script for mouth operator classification."""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import build_dataloaders
from .model import build_model


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs.detach(), targets) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(outputs, targets) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def main():
    parser = argparse.ArgumentParser(description="Train mouth operator classifier")
    parser.add_argument("--data-root", default="data/collected", help="Path with train/ and val/ subfolders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Fine-tune the backbone instead of freezing")
    parser.add_argument("--checkpoint", default="checkpoints/mouth_ops.pt", help="Where to save the best model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, idx_to_class = build_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    num_classes = len(idx_to_class)

    model = build_model(num_classes=num_classes, pretrained=True, freeze_backbone=not args.unfreeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "idx_to_class": idx_to_class, "args": vars(args)},
                ckpt_path,
            )
            print(f"  Saved checkpoint to {ckpt_path} (val_acc={val_acc:.3f})")


if __name__ == "__main__":
    main()
