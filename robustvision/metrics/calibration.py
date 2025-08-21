"""
Calibration metrics for RobustVision
"""

import numpy as np
from typing import Optional

def expected_calibration_error(
    probabilities: np.ndarray, 
    targets: np.ndarray, 
    n_bins: int = 15
) -> float:
    """
    Calculate Expected Calibration Error (ECE)
    
    Args:
        probabilities: Predicted probabilities of shape (N, C)
        targets: True labels of shape (N,)
        n_bins: Number of bins for calibration calculation
        
    Returns:
        Expected Calibration Error
    """
    # Get predicted classes and confidence
    predicted_classes = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = np.logical_and(confidence > bin_lower, confidence <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            # Calculate accuracy and confidence in this bin
            bin_accuracy = np.mean(predicted_classes[in_bin] == targets[in_bin])
            bin_confidence = np.mean(confidence[in_bin])
            
            # Add to ECE
            ece += bin_size * np.abs(bin_accuracy - bin_confidence)
    
    # Normalize by total number of samples
    ece /= len(targets)
    
    return ece

def reliability_diagram(
    probabilities: np.ndarray, 
    targets: np.ndarray, 
    n_bins: int = 15
) -> tuple:
    """
    Calculate reliability diagram data
    
    Args:
        probabilities: Predicted probabilities of shape (N, C)
        targets: True labels of shape (N,)
        n_bins: Number of bins
        
    Returns:
        Tuple of (accuracies, confidences, bin_sizes)
    """
    # Get predicted classes and confidence
    predicted_classes = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = np.logical_and(confidence > bin_lower, confidence <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            # Calculate accuracy and confidence in this bin
            bin_accuracy = np.mean(predicted_classes[in_bin] == targets[in_bin])
            bin_confidence = np.mean(confidence[in_bin])
            
            accuracies.append(bin_accuracy)
            confidences.append(bin_confidence)
            bin_sizes.append(bin_size)
        else:
            accuracies.append(0.0)
            confidences.append((bin_lower + bin_upper) / 2)
            bin_sizes.append(0)
    
    return np.array(accuracies), np.array(confidences), np.array(bin_sizes)

def maximum_calibration_error(
    probabilities: np.ndarray, 
    targets: np.ndarray, 
    n_bins: int = 15
) -> float:
    """
    Calculate Maximum Calibration Error (MCE)
    
    Args:
        probabilities: Predicted probabilities of shape (N, C)
        targets: True labels of shape (N,)
        n_bins: Number of bins for calibration calculation
        
    Returns:
        Maximum Calibration Error
    """
    accuracies, confidences, _ = reliability_diagram(probabilities, targets, n_bins)
    
    # Calculate maximum absolute difference
    mce = np.max(np.abs(accuracies - confidences))
    
    return mce

def adaptive_calibration_error(
    probabilities: np.ndarray, 
    targets: np.ndarray, 
    n_bins: int = 15
) -> float:
    """
    Calculate Adaptive Calibration Error (ACE)
    
    Args:
        probabilities: Predicted probabilities of shape (N, C)
        targets: True labels of shape (N,)
        n_bins: Number of bins for calibration calculation
        
    Returns:
        Adaptive Calibration Error
    """
    # Get predicted classes and confidence
    predicted_classes = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    
    # Sort by confidence
    sorted_indices = np.argsort(confidence)
    sorted_confidence = confidence[sorted_indices]
    sorted_predictions = predicted_classes[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    # Create adaptive bins with equal number of samples
    samples_per_bin = len(targets) // n_bins
    ace = 0.0
    
    for i in range(n_bins):
        start_idx = i * samples_per_bin
        end_idx = (i + 1) * samples_per_bin if i < n_bins - 1 else len(targets)
        
        # Calculate accuracy and confidence in this bin
        bin_accuracy = np.mean(sorted_predictions[start_idx:end_idx] == sorted_targets[start_idx:end_idx])
        bin_confidence = np.mean(sorted_confidence[start_idx:end_idx])
        
        # Add to ACE
        ace += np.abs(bin_accuracy - bin_confidence)
    
    # Average over bins
    ace /= n_bins
    
    return ace 