"""
Dataset loading utilities for RobustVision
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Supported dataset registry
SUPPORTED_DATASETS = {
    # CIFAR datasets
    "cifar10": {
        "type": "torchvision",
        "dataset": torchvision.datasets.CIFAR10,
        "num_classes": 10,
        "image_size": 32,
        "default_split": "test"
    },
    "cifar100": {
        "type": "torchvision", 
        "dataset": torchvision.datasets.CIFAR100,
        "num_classes": 100,
        "image_size": 32,
        "default_split": "test"
    },
    
    # ImageNet (requires manual download)
    "imagenet": {
        "type": "imagenet",
        "num_classes": 1000,
        "image_size": 224,
        "default_split": "val"
    },
    
    # ImageNet variants
    "imagenet-a": {
        "type": "imagenet_variant",
        "num_classes": 1000,
        "image_size": 224,
        "default_split": "test"
    },
    "imagenet-c": {
        "type": "imagenet_variant", 
        "num_classes": 1000,
        "image_size": 224,
        "default_split": "test"
    },
    "imagenet-r": {
        "type": "imagenet_variant",
        "num_classes": 200,
        "image_size": 224,
        "default_split": "test"
    }
}

# Standard transforms for different datasets
TRANSFORMS = {
    "cifar10": {
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    },
    "cifar100": {
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    },
    "imagenet": {
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
}

class CustomImageDataset(Dataset):
    """Custom dataset for loading images from directory"""
    
    def __init__(self, root_dir: str, transform=None, num_samples: Optional[int] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Find all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if any(filename.lower().endswith(ext) for ext in valid_extensions):
                        self.image_paths.append(os.path.join(class_dir, filename))
                        self.labels.append(class_idx)
        
        # Limit number of samples if specified
        if num_samples and num_samples < len(self.image_paths):
            indices = np.random.choice(len(self.image_paths), num_samples, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
        
        self.num_classes = len(set(self.labels))
        logger.info(f"Loaded {len(self.image_paths)} images from {self.num_classes} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset(
    dataset_name: str,
    split: str = "test",
    root: Optional[str] = None,
    transform: Optional[transforms.Compose] = None,
    num_samples: Optional[int] = None,
    download: bool = True
) -> Dataset:
    """
    Load a dataset by name
    
    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split (train, test, val)
        root: Root directory for dataset
        transform: Custom transforms to apply
        num_samples: Number of samples to load (for testing)
        download: Whether to download dataset if not present
        
    Returns:
        Loaded PyTorch dataset
    """
    # Check if dataset_name is a directory path
    if os.path.isdir(dataset_name):
        logger.info(f"Loading custom dataset from directory: {dataset_name}")
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return CustomImageDataset(dataset_name, transform, num_samples)
    
    # Check if dataset is in supported registry
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Use get_supported_datasets() to see available datasets.")
    
    dataset_info = SUPPORTED_DATASETS[dataset_name]
    dataset_type = dataset_info["type"]
    
    logger.info(f"Loading {dataset_type} dataset: {dataset_name} ({split} split)")
    
    # Get default transform if not provided
    if transform is None:
        transform = TRANSFORMS.get(dataset_name, {}).get(split)
        if transform is None:
            # Default transform
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    if dataset_type == "torchvision":
        dataset = _load_torchvision_dataset(dataset_name, split, root, transform, download)
    elif dataset_type == "imagenet":
        dataset = _load_imagenet_dataset(dataset_name, split, root, transform)
    elif dataset_type == "imagenet_variant":
        dataset = _load_imagenet_variant_dataset(dataset_name, split, root, transform)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Limit number of samples if specified
    if num_samples and num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        logger.info(f"Limited dataset to {num_samples} samples")
    
    # Add dataset info
    dataset.num_classes = dataset_info["num_classes"]
    dataset.name = dataset_name
    
    logger.info(f"Dataset loaded successfully: {len(dataset)} samples, {dataset.num_classes} classes")
    return dataset

def _load_torchvision_dataset(
    dataset_name: str, 
    split: str, 
    root: Optional[str], 
    transform: transforms.Compose,
    download: bool
) -> Dataset:
    """Load torchvision dataset"""
    dataset_class = SUPPORTED_DATASETS[dataset_name]["dataset"]
    
    # Map split names
    split_map = {"train": True, "test": False, "val": False}
    train = split_map.get(split, False)
    
    if root is None:
        root = os.path.join(os.path.expanduser("~"), ".torchvision", "datasets")
    
    dataset = dataset_class(
        root=root,
        train=train,
        transform=transform,
        download=download
    )
    
    return dataset

def _load_imagenet_dataset(
    dataset_name: str,
    split: str,
    root: Optional[str],
    transform: transforms.Compose
) -> Dataset:
    """Load ImageNet dataset"""
    if root is None:
        root = os.path.join(os.path.expanduser("~"), ".torchvision", "datasets", "imagenet")
    
    # Map split names
    split_map = {"train": "train", "test": "val", "val": "val"}
    split = split_map.get(split, "val")
    
    dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(root, split),
        transform=transform
    )
    
    return dataset

def _load_imagenet_variant_dataset(
    dataset_name: str,
    split: str,
    root: Optional[str],
    transform: transforms.Compose
) -> Dataset:
    """Load ImageNet variant dataset (A, C, R)"""
    if root is None:
        root = os.path.join(os.path.expanduser("~"), ".torchvision", "datasets", dataset_name)
    
    if not os.path.exists(root):
        raise FileNotFoundError(
            f"Dataset {dataset_name} not found at {root}. "
            f"Please download it manually and place it in the specified directory."
        )
    
    dataset = torchvision.datasets.ImageFolder(
        root=root,
        transform=transform
    )
    
    return dataset

def get_supported_datasets() -> List[str]:
    """Get list of supported dataset names"""
    return list(SUPPORTED_DATASETS.keys())

def get_dataset_info(dataset_name: str) -> Dict[str, Union[str, int]]:
    """Get information about a specific dataset"""
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not supported")
    
    return SUPPORTED_DATASETS[dataset_name].copy()

def get_dataset_transforms(dataset_name: str, split: str = "test") -> transforms.Compose:
    """Get default transforms for a dataset"""
    return TRANSFORMS.get(dataset_name, {}).get(split)

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader for a dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    ) 