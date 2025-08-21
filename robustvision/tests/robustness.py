"""
Robustness tests for RobustVision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm

from ..utils.logging import get_logger
from ..corruptions.corruption_utils import apply_corruptions
from ..attacks.adversarial_attacks import AdversarialAttacks

logger = get_logger(__name__)

class RobustnessTests:
    """Robustness test suite for vision models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize robustness tests
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.corruptions = config.get("corruptions", ["gaussian_noise", "motion_blur", "brightness"])
        self.attacks = config.get("attacks", ["fgsm", "pgd"])
        self.attack_params = config.get("attack_params", {})
        self.batch_size = config.get("batch_size", 32)
        
        # Initialize adversarial attacks
        self.adversarial_attacks = AdversarialAttacks()
    
    def run(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Run robustness tests
        
        Args:
            model: PyTorch model to test
            dataloader: DataLoader for evaluation
            device: Device to run tests on
            
        Returns:
            Dictionary containing test results
        """
        logger.info("Running robustness tests...")
        
        results = {}
        
        # Test corruption robustness
        if self.corruptions:
            logger.info("Testing corruption robustness...")
            results["corruptions"] = self._test_corruptions(model, dataloader, device)
        
        # Test adversarial robustness
        if self.attacks:
            logger.info("Testing adversarial robustness...")
            results["adversarial"] = self._test_adversarial_attacks(model, dataloader, device)
        
        # Calculate overall robustness metrics
        results.update(self._calculate_robustness_metrics(results))
        
        logger.info("Robustness tests completed")
        return results
    
    def _test_corruptions(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, Any]:
        """Test model robustness to image corruptions"""
        
        corruption_results = {}
        
        for corruption in self.corruptions:
            logger.info(f"Testing corruption: {corruption}")
            
            correct = 0
            total = 0
            all_predictions = []
            all_targets = []
            
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc=f"Testing {corruption}")):
                    data, target = data.to(device), target.to(device)
                    
                    # Apply corruption
                    corrupted_data = apply_corruptions(data, corruption)
                    
                    # Forward pass
                    output = model(corrupted_data)
                    predictions = torch.argmax(output, dim=1)
                    
                    # Calculate accuracy
                    correct += (predictions == target).sum().item()
                    total += target.size(0)
                    
                    # Store results
                    all_predictions.append(predictions.cpu())
                    all_targets.append(target.cpu())
            
            # Calculate metrics
            accuracy = correct / total
            all_predictions = torch.cat(all_predictions).numpy()
            all_targets = torch.cat(all_targets).numpy()
            
            corruption_results[corruption] = {
                "accuracy": float(accuracy),
                "predictions": all_predictions.tolist(),
                "targets": all_targets.tolist()
            }
            
            logger.info(f"{corruption} accuracy: {accuracy:.4f}")
        
        return corruption_results
    
    def _test_adversarial_attacks(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, Any]:
        """Test model robustness to adversarial attacks"""
        
        attack_results = {}
        
        for attack in self.attacks:
            logger.info(f"Testing adversarial attack: {attack}")
            
            attack_params = self.attack_params.get(attack, {})
            
            correct = 0
            total = 0
            all_predictions = []
            all_targets = []
            all_perturbations = []
            
            model.eval()
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc=f"Testing {attack}")):
                data, target = data.to(device), target.to(device)
                
                # Generate adversarial examples
                if attack == "fgsm":
                    adversarial_data = self.adversarial_attacks.fgsm_attack(
                        model, data, target, **attack_params
                    )
                elif attack == "pgd":
                    adversarial_data = self.adversarial_attacks.pgd_attack(
                        model, data, target, **attack_params
                    )
                elif attack == "cw":
                    adversarial_data = self.adversarial_attacks.cw_attack(
                        model, data, target, **attack_params
                    )
                else:
                    raise ValueError(f"Unknown attack: {attack}")
                
                # Calculate perturbation magnitude
                perturbation = torch.norm(adversarial_data - data, p=2, dim=(1, 2, 3))
                all_perturbations.append(perturbation.detach().cpu())
                
                # Forward pass
                with torch.no_grad():
                    output = model(adversarial_data)
                    predictions = torch.argmax(output, dim=1)
                
                # Calculate accuracy
                correct += (predictions == target).sum().item()
                total += target.size(0)
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_targets.append(target.cpu())
            
            # Calculate metrics
            accuracy = correct / total
            success_rate = 1.0 - accuracy  # Attack success rate
            all_predictions = torch.cat(all_predictions).numpy()
            all_targets = torch.cat(all_targets).numpy()
            all_perturbations = torch.cat(all_perturbations).numpy()
            
            attack_results[attack] = {
                "accuracy": float(accuracy),
                "success_rate": float(success_rate),
                "avg_perturbation": float(np.mean(all_perturbations)),
                "max_perturbation": float(np.max(all_perturbations)),
                "predictions": all_predictions.tolist(),
                "targets": all_targets.tolist(),
                "perturbations": all_perturbations.tolist()
            }
            
            logger.info(f"{attack} accuracy: {accuracy:.4f}, success rate: {success_rate:.4f}")
        
        return attack_results
    
    def _calculate_robustness_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall robustness metrics"""
        
        metrics = {}
        
        # Corruption metrics
        if "corruptions" in results:
            corruption_accuracies = [
                result["accuracy"] for result in results["corruptions"].values()
            ]
            metrics["corruption_accuracy"] = float(np.mean(corruption_accuracies))
            metrics["corruption_accuracy_std"] = float(np.std(corruption_accuracies))
            metrics["worst_corruption_accuracy"] = float(np.min(corruption_accuracies))
        
        # Adversarial metrics
        if "adversarial" in results:
            attack_success_rates = [
                result["success_rate"] for result in results["adversarial"].values()
            ]
            metrics["attack_success_rate"] = float(np.mean(attack_success_rates))
            metrics["attack_success_rate_std"] = float(np.std(attack_success_rates))
            metrics["worst_attack_success_rate"] = float(np.max(attack_success_rates))
            
            # Average perturbation across all attacks
            avg_perturbations = [
                result["avg_perturbation"] for result in results["adversarial"].values()
            ]
            metrics["avg_perturbation"] = float(np.mean(avg_perturbations))
        
        # Overall robustness score (lower is better)
        if "corruption_accuracy" in metrics and "attack_success_rate" in metrics:
            # Combine corruption and adversarial robustness
            corruption_score = 1.0 - metrics["corruption_accuracy"]
            adversarial_score = metrics["attack_success_rate"]
            metrics["overall_robustness_score"] = 0.5 * corruption_score + 0.5 * adversarial_score
        
        return metrics
    
    def generate_plots(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Generate visualization plots for robustness results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Corruption accuracy comparison
        if "corruptions" in results:
            corruptions = list(results["corruptions"].keys())
            accuracies = [results["corruptions"][c]["accuracy"] for c in corruptions]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(corruptions, accuracies)
            plt.title('Model Accuracy Under Different Corruptions')
            plt.xlabel('Corruption Type')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/corruption_accuracy.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # Adversarial attack success rates
        if "adversarial" in results:
            attacks = list(results["adversarial"].keys())
            success_rates = [results["adversarial"][a]["success_rate"] for a in attacks]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(attacks, success_rates, color='red')
            plt.title('Adversarial Attack Success Rates')
            plt.xlabel('Attack Type')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{rate:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/attack_success_rates.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # Perturbation magnitude distribution
        if "adversarial" in results:
            plt.figure(figsize=(12, 8))
            for attack in results["adversarial"]:
                perturbations = results["adversarial"][attack]["perturbations"]
                plt.hist(perturbations, bins=50, alpha=0.7, label=attack.upper())
            
            plt.title('Distribution of Perturbation Magnitudes')
            plt.xlabel('L2 Perturbation Magnitude')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/perturbation_distribution.png", dpi=300, bbox_inches='tight')
            plt.show() 