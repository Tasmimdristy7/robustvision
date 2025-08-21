"""
Model loading utilities for RobustVision
"""

import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Dict, List, Optional, Union
import timm
from transformers import ViTForImageClassification, ViTImageProcessor

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Supported model registry
SUPPORTED_MODELS = {
    # ResNet models
    "resnet18": {"type": "torchvision", "model": tv_models.resnet18, "pretrained": True},
    "resnet34": {"type": "torchvision", "model": tv_models.resnet34, "pretrained": True},
    "resnet50": {"type": "torchvision", "model": tv_models.resnet50, "pretrained": True},
    "resnet101": {"type": "torchvision", "model": tv_models.resnet101, "pretrained": True},
    "resnet152": {"type": "torchvision", "model": tv_models.resnet152, "pretrained": True},
    
    # Vision Transformers (timm)
    "vit_base_patch16_224": {"type": "timm", "model": "vit_base_patch16_224", "pretrained": True},
    "vit_large_patch16_224": {"type": "timm", "model": "vit_large_patch16_224", "pretrained": True},
    "vit_huge_patch14_224": {"type": "timm", "model": "vit_huge_patch14_224", "pretrained": True},
    
    # HuggingFace Vision Transformers
    "google/vit-base-patch16-224": {"type": "hf", "model": "google/vit-base-patch16-224"},
    "google/vit-large-patch16-224": {"type": "hf", "model": "google/vit-large-patch16-224"},
    "google/vit-huge-patch14-224": {"type": "hf", "model": "google/vit-huge-patch14-224"},
}

def load_model(
    model_name: str, 
    pretrained: bool = True,
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load a vision model by name
    
    Args:
        model_name: Name of the model to load
        pretrained: Whether to load pretrained weights
        num_classes: Number of classes (for custom datasets)
        device: Device to load model on
        
    Returns:
        Loaded PyTorch model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if model_name is a file path
    if os.path.exists(model_name):
        logger.info(f"Loading model from file: {model_name}")
        model = torch.load(model_name, map_location=device)
        if isinstance(model, dict) and 'state_dict' in model:
            # Load state dict
            model = load_model_from_state_dict(model['state_dict'], model.get('model_name', 'unknown'))
        return model
    
    # Check if model is in supported registry
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Use get_supported_models() to see available models.")
    
    model_info = SUPPORTED_MODELS[model_name]
    model_type = model_info["type"]
    
    logger.info(f"Loading {model_type} model: {model_name}")
    
    if model_type == "torchvision":
        model = _load_torchvision_model(model_name, pretrained, num_classes)
    elif model_type == "timm":
        model = _load_timm_model(model_name, pretrained, num_classes)
    elif model_type == "hf":
        model = _load_hf_model(model_name, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    return model

def _load_torchvision_model(model_name: str, pretrained: bool, num_classes: Optional[int]) -> nn.Module:
    """Load torchvision model"""
    model_func = SUPPORTED_MODELS[model_name]["model"]
    
    if pretrained:
        model = model_func(pretrained=True)
    else:
        model = model_func(pretrained=False)
    
    # Modify final layer if num_classes is specified
    if num_classes is not None and num_classes != 1000:  # ImageNet has 1000 classes
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    return model

def _load_timm_model(model_name: str, pretrained: bool, num_classes: Optional[int]) -> nn.Module:
    """Load timm model"""
    model = timm.create_model(
        SUPPORTED_MODELS[model_name]["model"],
        pretrained=pretrained,
        num_classes=num_classes or 1000
    )
    return model

def _load_hf_model(model_name: str, num_classes: Optional[int]) -> nn.Module:
    """Load HuggingFace model"""
    model = ViTForImageClassification.from_pretrained(
        SUPPORTED_MODELS[model_name]["model"],
        num_labels=num_classes or 1000,
        ignore_mismatched_sizes=True
    )
    return model

def load_model_from_state_dict(state_dict: Dict, model_name: str) -> nn.Module:
    """Load model from state dict"""
    # Try to infer model architecture from state dict
    if any('resnet' in key.lower() for key in state_dict.keys()):
        # ResNet-like architecture
        if 'layer4.2.conv3.weight' in state_dict:
            model = tv_models.resnet152(pretrained=False)
        elif 'layer4.1.conv3.weight' in state_dict:
            model = tv_models.resnet101(pretrained=False)
        elif 'layer4.0.conv3.weight' in state_dict:
            model = tv_models.resnet50(pretrained=False)
        elif 'layer4.0.conv2.weight' in state_dict:
            model = tv_models.resnet34(pretrained=False)
        else:
            model = tv_models.resnet18(pretrained=False)
    else:
        # Default to ResNet50
        model = tv_models.resnet50(pretrained=False)
    
    model.load_state_dict(state_dict)
    return model

def get_supported_models() -> List[str]:
    """Get list of supported model names"""
    return list(SUPPORTED_MODELS.keys())

def get_model_info(model_name: str) -> Dict[str, Union[str, int]]:
    """Get information about a specific model"""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' not supported")
    
    model_info = SUPPORTED_MODELS[model_name].copy()
    
    # Add additional info
    if model_info["type"] == "torchvision":
        # Load model to get parameter count
        model = model_info["model"](pretrained=False)
        model_info["total_parameters"] = sum(p.numel() for p in model.parameters())
        model_info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return model_info

def register_custom_model(name: str, model: nn.Module, model_type: str = "custom"):
    """Register a custom model for use with RobustVision"""
    SUPPORTED_MODELS[name] = {
        "type": model_type,
        "model": model,
        "pretrained": False
    }
    logger.info(f"Registered custom model: {name}")

def get_model_processor(model_name: str):
    """Get the appropriate processor for a model (for HuggingFace models)"""
    if model_name in SUPPORTED_MODELS and SUPPORTED_MODELS[model_name]["type"] == "hf":
        return ViTImageProcessor.from_pretrained(SUPPORTED_MODELS[model_name]["model"])
    return None 