"""Model definitions for mouth operator classification."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

__all__ = ["build_model", "load_checkpoint"]


def build_model(num_classes: int = 5, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Create a MobileNetV3-Small with a custom classifier head."""
    model = models.mobilenet_v3_small(pretrained=pretrained)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    # Replace final classification layer
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, map_location: Optional[str] = None) -> nn.Module:
    """Load model weights from checkpoint if path exists."""
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(ckpt, map_location=map_location)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    return model
