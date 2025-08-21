"""
Image corruption utilities for RobustVision
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Union, List, Optional
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance

from ..utils.logging import get_logger

logger = get_logger(__name__)

def apply_corruptions(
    data: torch.Tensor, 
    corruption_type: Union[str, List[str]], 
    severity: int = 1
) -> torch.Tensor:
    """
    Apply image corruptions to input data
    
    Args:
        data: Input tensor of shape (B, C, H, W)
        corruption_type: Type of corruption(s) to apply
        severity: Severity level (1-5)
        
    Returns:
        Corrupted tensor
    """
    if isinstance(corruption_type, str):
        corruption_type = [corruption_type]
    
    corrupted_data = data.clone()
    
    for corruption in corruption_type:
        corrupted_data = _apply_single_corruption(corrupted_data, corruption, severity)
    
    return corrupted_data

def _apply_single_corruption(
    data: torch.Tensor, 
    corruption_type: str, 
    severity: int
) -> torch.Tensor:
    """Apply a single corruption type"""
    
    # Normalize severity to [0, 1]
    severity_norm = severity / 5.0
    
    if corruption_type == "gaussian_noise":
        return _gaussian_noise(data, severity_norm)
    elif corruption_type == "shot_noise":
        return _shot_noise(data, severity_norm)
    elif corruption_type == "impulse_noise":
        return _impulse_noise(data, severity_norm)
    elif corruption_type == "defocus_blur":
        return _defocus_blur(data, severity_norm)
    elif corruption_type == "motion_blur":
        return _motion_blur(data, severity_norm)
    elif corruption_type == "zoom_blur":
        return _zoom_blur(data, severity_norm)
    elif corruption_type == "brightness":
        return _brightness(data, severity_norm)
    elif corruption_type == "contrast":
        return _contrast(data, severity_norm)
    elif corruption_type == "saturate":
        return _saturate(data, severity_norm)
    elif corruption_type == "fog":
        return _fog(data, severity_norm)
    elif corruption_type == "frost":
        return _frost(data, severity_norm)
    elif corruption_type == "snow":
        return _snow(data, severity_norm)
    elif corruption_type == "spatter":
        return _spatter(data, severity_norm)
    elif corruption_type == "speckle_noise":
        return _speckle_noise(data, severity_norm)
    elif corruption_type == "gaussian_blur":
        return _gaussian_blur(data, severity_norm)
    elif corruption_type == "jpeg_compression":
        return _jpeg_compression(data, severity_norm)
    elif corruption_type == "pixelate":
        return _pixelate(data, severity_norm)
    else:
        logger.warning(f"Unknown corruption type: {corruption_type}")
        return data

def _gaussian_noise(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply Gaussian noise"""
    std = severity * 0.5
    noise = torch.randn_like(data) * std
    return torch.clamp(data + noise, 0, 1)

def _shot_noise(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply shot noise (Poisson noise)"""
    intensity = severity * 60
    data_scaled = data * 255
    noise = torch.poisson(data_scaled * intensity) / intensity
    return torch.clamp(noise / 255, 0, 1)

def _impulse_noise(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply impulse noise (salt and pepper)"""
    prob = severity * 0.05
    mask = torch.rand_like(data) < prob
    salt = torch.rand_like(data) < 0.5
    data_corrupted = data.clone()
    data_corrupted[mask] = salt[mask].float()
    return data_corrupted

def _defocus_blur(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply defocus blur"""
    kernel_size = int(severity * 10) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create Gaussian kernel
    sigma = severity * 2.0
    kernel = _gaussian_kernel(kernel_size, sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution to each channel separately
    blurred = torch.zeros_like(data)
    for i in range(data.size(1)):  # For each channel
        blurred[:, i:i+1] = F.conv2d(data[:, i:i+1], kernel, padding=kernel_size//2)
    return blurred

def _motion_blur(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply motion blur"""
    kernel_size = int(severity * 15) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create motion blur kernel
    kernel = torch.zeros(1, 1, kernel_size, kernel_size)
    center = kernel_size // 2
    kernel[0, 0, center, :] = 1.0 / kernel_size
    
    # Apply convolution to each channel separately
    blurred = torch.zeros_like(data)
    for i in range(data.size(1)):  # For each channel
        blurred[:, i:i+1] = F.conv2d(data[:, i:i+1], kernel, padding=center)
    return blurred

def _zoom_blur(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply zoom blur"""
    # Simulate zoom blur by averaging multiple scaled versions
    scales = [1.0 + severity * i * 0.1 for i in range(5)]
    blurred = torch.zeros_like(data)
    
    for scale in scales:
        scaled = F.interpolate(data, scale_factor=scale, mode='bilinear', align_corners=False)
        if scale > 1.0:
            # Crop to original size
            h, w = data.shape[2:]
            start_h = (scaled.shape[2] - h) // 2
            start_w = (scaled.shape[3] - w) // 2
            scaled = scaled[:, :, start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad to original size
            h, w = data.shape[2:]
            pad_h = (h - scaled.shape[2]) // 2
            pad_w = (w - scaled.shape[3]) // 2
            scaled = F.pad(scaled, (pad_w, pad_w, pad_h, pad_h))
        
        blurred += scaled
    
    return blurred / len(scales)

def _brightness(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply brightness change"""
    factor = 1.0 + severity * 0.5
    return torch.clamp(data * factor, 0, 1)

def _contrast(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply contrast change"""
    factor = 1.0 + severity * 0.5
    mean = torch.mean(data, dim=[2, 3], keepdim=True)
    return torch.clamp((data - mean) * factor + mean, 0, 1)

def _saturate(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply saturation change"""
    factor = 1.0 + severity * 0.5
    # Convert to HSV, modify saturation, convert back
    # Simplified version: modify color channels
    gray = torch.mean(data, dim=1, keepdim=True)
    return torch.clamp(gray + (data - gray) * factor, 0, 1)

def _fog(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply fog effect"""
    # Create fog mask
    h, w = data.shape[2:]
    fog_mask = torch.rand(1, 1, h, w) * severity * 0.3
    fog_mask = F.interpolate(fog_mask, size=(h, w), mode='bilinear', align_corners=False)
    
    # Apply fog
    fog_color = torch.ones_like(data) * 0.5
    return torch.clamp(data * (1 - fog_mask) + fog_color * fog_mask, 0, 1)

def _frost(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply frost effect"""
    # Simulate frost by adding structured noise
    h, w = data.shape[2:]
    frost_pattern = torch.randn(1, 1, h, w) * severity * 0.2
    frost_pattern = F.interpolate(frost_pattern, size=(h, w), mode='bilinear', align_corners=False)
    
    return torch.clamp(data + frost_pattern, 0, 1)

def _snow(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply snow effect"""
    h, w = data.shape[2:]
    
    # Create snowflakes
    num_snowflakes = int(severity * 1000)
    snow_mask = torch.zeros(1, 1, h, w)
    
    for _ in range(num_snowflakes):
        x = torch.randint(0, w, (1,))
        y = torch.randint(0, h, (1,))
        size = torch.randint(1, int(severity * 5) + 2, (1,))
        
        y1, y2 = max(0, y - size), min(h, y + size)
        x1, x2 = max(0, x - size), min(w, x + size)
        
        snow_mask[0, 0, y1:y2, x1:x2] = 1.0
    
    # Apply snow
    snow_color = torch.ones_like(data)
    return torch.clamp(data * (1 - snow_mask) + snow_color * snow_mask * 0.8, 0, 1)

def _spatter(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply spatter effect"""
    # Simulate liquid spatter
    h, w = data.shape[2:]
    spatter_mask = torch.rand(1, 1, h, w) < severity * 0.1
    
    # Create spatter color (brownish)
    spatter_color = torch.tensor([0.6, 0.4, 0.2]).view(1, 3, 1, 1).repeat(1, 1, h, w)
    
    return torch.clamp(data * (1 - spatter_mask) + spatter_color * spatter_mask, 0, 1)

def _speckle_noise(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply speckle noise"""
    noise = torch.randn_like(data) * severity * 0.5
    return torch.clamp(data + data * noise, 0, 1)

def _gaussian_blur(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply Gaussian blur"""
    kernel_size = int(severity * 10) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    sigma = severity * 2.0
    kernel = _gaussian_kernel(kernel_size, sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution to each channel separately
    blurred = torch.zeros_like(data)
    for i in range(data.size(1)):  # For each channel
        blurred[:, i:i+1] = F.conv2d(data[:, i:i+1], kernel, padding=kernel_size//2)
    return blurred

def _jpeg_compression(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply JPEG compression artifacts"""
    # Simulate JPEG compression by quantizing in frequency domain
    quality = int(100 - severity * 80)  # Lower quality = more artifacts
    
    # Convert to PIL and back to simulate JPEG compression
    data_np = (data.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    compressed = []
    
    for img in data_np:
        pil_img = Image.fromarray(img)
        # Simulate compression by saving and loading
        compressed.append(np.array(pil_img) / 255.0)
    
    compressed = torch.tensor(compressed).permute(0, 3, 1, 2)
    return compressed

def _pixelate(data: torch.Tensor, severity: float) -> torch.Tensor:
    """Apply pixelation effect"""
    scale_factor = 1.0 - severity * 0.8
    if scale_factor < 0.1:
        scale_factor = 0.1
    
    # Downsample and upsample
    h, w = data.shape[2:]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    downsampled = F.interpolate(data, size=(new_h, new_w), mode='nearest')
    upsampled = F.interpolate(downsampled, size=(h, w), mode='nearest')
    
    return upsampled

def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Create Gaussian kernel"""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    return g.unsqueeze(0) * g.unsqueeze(1)

# Available corruption types
AVAILABLE_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    "defocus_blur", "motion_blur", "zoom_blur", "gaussian_blur",
    "brightness", "contrast", "saturate",
    "fog", "frost", "snow", "spatter",
    "jpeg_compression", "pixelate"
]

def get_available_corruptions() -> List[str]:
    """Get list of available corruption types"""
    return AVAILABLE_CORRUPTIONS.copy() 