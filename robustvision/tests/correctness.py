"""
Correctness tests for RobustVision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logging import get_logger
from ..metrics.calibration import expected_calibration_error

logger = get_logger(__name__)

class CorrectnessTests:
    """Correctness test suite for vision models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize correctness tests
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics = config.get("metrics", ["accuracy", "ece", "confusion_matrix"])
        self.batch_size = config.get("batch_size", 32)
    
    def run(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Run correctness tests
        
        Args:
            model: PyTorch model to test
            dataloader: DataLoader for evaluation
            device: Device to run tests on
            
        Returns:
            Dictionary containing test results
        """
        logger.info("Running correctness tests...")
        
        results = {}
        
        # Collect predictions and targets
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                probabilities = F.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_targets.append(target.cpu())
                all_probabilities.append(probabilities.cpu())
                
                if batch_idx % 100 == 0:
                    logger.info(f"Processed {batch_idx * self.batch_size} samples")
        
        # Concatenate results
        all_predictions = torch.cat(all_predictions).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_probabilities = torch.cat(all_probabilities).numpy()
        
        logger.info(f"Collected predictions for {len(all_targets)} samples")
        
        # Calculate metrics
        if "accuracy" in self.metrics:
            results["accuracy"] = self._calculate_accuracy(all_predictions, all_targets)
        
        if "ece" in self.metrics:
            results["ece"] = self._calculate_ece(all_probabilities, all_targets)
        
        if "confusion_matrix" in self.metrics:
            results["confusion_matrix"] = self._calculate_confusion_matrix(
                all_predictions, all_targets
            )
        
        # Additional metrics
        results["classification_report"] = self._generate_classification_report(
            all_predictions, all_targets
        )
        
        results["calibration_curve"] = self._calculate_calibration_curve(
            all_probabilities, all_targets
        )
        
        # Per-class accuracy
        results["per_class_accuracy"] = self._calculate_per_class_accuracy(
            all_predictions, all_targets
        )
        
        logger.info("Correctness tests completed")
        return results
    
    def _calculate_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate classification accuracy"""
        accuracy = np.mean(predictions == targets)
        logger.info(f"Accuracy: {accuracy:.4f}")
        return float(accuracy)
    
    def _calculate_ece(self, probabilities: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Expected Calibration Error"""
        ece = expected_calibration_error(probabilities, targets, n_bins=15)
        logger.info(f"Expected Calibration Error: {ece:.4f}")
        return float(ece)
    
    def _calculate_confusion_matrix(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        # Calculate metrics from confusion matrix
        tn = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + np.diag(cm)
        fp = np.sum(cm, axis=0) - np.diag(cm)
        fn = np.sum(cm, axis=1) - np.diag(cm)
        tp = np.diag(cm)
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Handle division by zero
        precision = np.nan_to_num(precision, nan=0.0)
        recall = np.nan_to_num(recall, nan=0.0)
        f1_score = np.nan_to_num(f1_score, nan=0.0)
        
        return {
            "matrix": cm.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1_score": f1_score.tolist()
        }
    
    def _generate_classification_report(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Generate detailed classification report"""
        report = classification_report(
            targets, 
            predictions, 
            output_dict=True, 
            zero_division=0
        )
        return report
    
    def _calculate_calibration_curve(
        self, 
        probabilities: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate calibration curve"""
        # Use the maximum probability for each sample
        max_probs = np.max(probabilities, axis=1)
        predicted_classes = np.argmax(probabilities, axis=1)
        
        # Create binary targets for calibration curve
        binary_targets = (predicted_classes == targets).astype(int)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            binary_targets, max_probs, n_bins=10
        )
        
        return {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist()
        }
    
    def _calculate_per_class_accuracy(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate per-class accuracy"""
        num_classes = max(np.max(targets), np.max(predictions)) + 1
        per_class_acc = {}
        
        for class_idx in range(num_classes):
            class_mask = targets == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predictions[class_mask] == targets[class_mask])
                per_class_acc[f"class_{class_idx}"] = float(class_acc)
        
        return per_class_acc
    
    def generate_plots(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Generate visualization plots for correctness results"""
        
        # Confusion Matrix
        if "confusion_matrix" in results:
            cm = np.array(results["confusion_matrix"]["matrix"])
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            if save_path:
                plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # Calibration Curve
        if "calibration_curve" in results:
            cal_curve = results["calibration_curve"]
            plt.figure(figsize=(8, 6))
            plt.plot(cal_curve["mean_predicted_value"], cal_curve["fraction_of_positives"], 
                    marker='o', label='Model')
            plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Curve')
            plt.legend()
            plt.grid(True)
            if save_path:
                plt.savefig(f"{save_path}/calibration_curve.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # Per-class Accuracy
        if "per_class_accuracy" in results:
            per_class_acc = results["per_class_accuracy"]
            classes = list(per_class_acc.keys())
            accuracies = list(per_class_acc.values())
            
            plt.figure(figsize=(12, 6))
            plt.bar(classes, accuracies)
            plt.title('Per-Class Accuracy')
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            if save_path:
                plt.savefig(f"{save_path}/per_class_accuracy.png", dpi=300, bbox_inches='tight')
            plt.show() 