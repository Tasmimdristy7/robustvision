"""
Security tests for RobustVision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from ..utils.logging import get_logger
from ..attacks.adversarial_attacks import AdversarialAttacks

logger = get_logger(__name__)

class SecurityTests:
    """Security test suite for vision models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize security tests
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.membership_inference = config.get("membership_inference", True)
        self.data_poisoning = config.get("data_poisoning", True)
        self.adversarial_vulnerability = config.get("adversarial_vulnerability", True)
        self.batch_size = config.get("batch_size", 32)
        
        # Initialize adversarial attacks for vulnerability testing
        self.adversarial_attacks = AdversarialAttacks()
    
    def run(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Run security tests
        
        Args:
            model: PyTorch model to test
            dataloader: DataLoader for evaluation
            device: Device to run tests on
            
        Returns:
            Dictionary containing test results
        """
        logger.info("Running security tests...")
        
        results = {}
        
        # Test membership inference vulnerability
        if self.membership_inference:
            logger.info("Testing membership inference vulnerability...")
            results["membership_inference"] = self._test_membership_inference(
                model, dataloader, device
            )
        
        # Test data poisoning vulnerability
        if self.data_poisoning:
            logger.info("Testing data poisoning vulnerability...")
            results["data_poisoning"] = self._test_data_poisoning(
                model, dataloader, device
            )
        
        # Test adversarial vulnerability
        if self.adversarial_vulnerability:
            logger.info("Testing adversarial vulnerability...")
            results["adversarial_vulnerability"] = self._test_adversarial_vulnerability(
                model, dataloader, device
            )
        
        # Calculate overall security metrics
        results.update(self._calculate_security_metrics(results))
        
        logger.info("Security tests completed")
        return results
    
    def _test_membership_inference(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, Any]:
        """Test membership inference vulnerability"""
        
        # Collect model outputs for membership inference
        all_outputs = []
        all_targets = []
        all_membership_labels = []  # 1 for training set, 0 for non-training set
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Membership inference")):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                probabilities = F.softmax(output, dim=1)
                
                # Store outputs and targets
                all_outputs.append(probabilities.cpu())
                all_targets.append(target.cpu())
                
                # For this test, we assume all samples are from training set
                # In practice, you'd need separate training and non-training samples
                all_membership_labels.extend([1] * target.size(0))
        
        # Concatenate results
        all_outputs = torch.cat(all_outputs).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_membership_labels = np.array(all_membership_labels)
        
        # Create membership inference attack
        membership_scores = self._perform_membership_inference_attack(
            all_outputs, all_targets, all_membership_labels
        )
        
        # Calculate metrics
        if len(np.unique(all_membership_labels)) > 1:
            auc = roc_auc_score(all_membership_labels, membership_scores)
            fpr, tpr, _ = roc_curve(all_membership_labels, membership_scores)
        else:
            auc = 0.5  # Random performance if only one class
            fpr, tpr = [0, 1], [0, 1]
        
        return {
            "auc": float(auc),
            "membership_scores": membership_scores.tolist(),
            "membership_labels": all_membership_labels.tolist(),
            "roc_curve": {"fpr": fpr, "tpr": tpr}
        }
    
    def _perform_membership_inference_attack(
        self, 
        outputs: np.ndarray, 
        targets: np.ndarray, 
        membership_labels: np.ndarray
    ) -> np.ndarray:
        """Perform membership inference attack using model outputs"""
        
        # Use prediction confidence as membership score
        # Higher confidence suggests the sample was in training set
        max_probs = np.max(outputs, axis=1)
        
        # Also use prediction correctness
        predicted_classes = np.argmax(outputs, axis=1)
        correct_predictions = (predicted_classes == targets).astype(float)
        
        # Combine confidence and correctness
        membership_scores = 0.7 * max_probs + 0.3 * correct_predictions
        
        return membership_scores
    
    def _test_data_poisoning(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, Any]:
        """Test data poisoning vulnerability"""
        
        # Simulate label-flip poisoning
        label_flip_results = self._test_label_flip_poisoning(model, dataloader, device)
        
        # Simulate backdoor poisoning
        backdoor_results = self._test_backdoor_poisoning(model, dataloader, device)
        
        return {
            "label_flip": label_flip_results,
            "backdoor": backdoor_results
        }
    
    def _test_label_flip_poisoning(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, Any]:
        """Test label-flip data poisoning vulnerability"""
        
        # Simulate label-flip attack by flipping some labels
        poisoned_accuracy = 0
        total_samples = 0
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Label-flip poisoning")):
                data, target = data.to(device), target.to(device)
                
                # Flip some labels (simulate poisoned data)
                flip_mask = torch.rand(target.size(0)) < 0.1  # 10% poisoning rate
                poisoned_target = target.clone()
                poisoned_target[flip_mask] = (poisoned_target[flip_mask] + 1) % model.fc.out_features
                
                # Forward pass
                output = model(data)
                predictions = torch.argmax(output, dim=1)
                
                # Calculate accuracy on poisoned data
                poisoned_accuracy += (predictions == poisoned_target).sum().item()
                total_samples += target.size(0)
        
        accuracy = poisoned_accuracy / total_samples if total_samples > 0 else 0
        
        return {
            "accuracy": float(accuracy),
            "poisoning_rate": 0.1
        }
    
    def _test_backdoor_poisoning(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, Any]:
        """Test backdoor data poisoning vulnerability"""
        
        # Simulate backdoor trigger
        trigger_size = 8
        trigger = torch.ones(1, 3, trigger_size, trigger_size).to(device) * 0.5
        
        backdoor_accuracy = 0
        total_samples = 0
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Backdoor poisoning")):
                data, target = data.to(device), target.to(device)
                
                # Add backdoor trigger to some samples
                trigger_mask = torch.rand(data.size(0)) < 0.1  # 10% trigger rate
                data_with_trigger = data.clone()
                
                for i in range(data.size(0)):
                    if trigger_mask[i]:
                        # Add trigger to bottom-right corner
                        data_with_trigger[i, :, -trigger_size:, -trigger_size:] = trigger
                
                # Forward pass
                output = model(data_with_trigger)
                predictions = torch.argmax(output, dim=1)
                
                # Calculate accuracy
                backdoor_accuracy += (predictions == target).sum().item()
                total_samples += target.size(0)
        
        accuracy = backdoor_accuracy / total_samples if total_samples > 0 else 0
        
        return {
            "accuracy": float(accuracy),
            "trigger_rate": 0.1,
            "trigger_size": trigger_size
        }
    
    def _test_adversarial_vulnerability(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, Any]:
        """Test adversarial vulnerability by finding minimum epsilon"""
        
        epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        vulnerability_results = {}
        
        for epsilon in epsilon_values:
            logger.info(f"Testing adversarial vulnerability with epsilon={epsilon}")
            
            correct = 0
            total = 0
            
            model.eval()
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc=f"Epsilon={epsilon}")):
                data, target = data.to(device), target.to(device)
                
                # Generate adversarial examples with current epsilon
                adversarial_data = self.adversarial_attacks.fgsm_attack(
                    model, data, target, epsilon=epsilon
                )
                
                # Forward pass
                with torch.no_grad():
                    output = model(adversarial_data)
                    predictions = torch.argmax(output, dim=1)
                
                # Calculate accuracy
                correct += (predictions == target).sum().item()
                total += target.size(0)
            
            accuracy = correct / total if total > 0 else 0
            vulnerability_results[f"epsilon_{epsilon}"] = {
                "accuracy": float(accuracy),
                "success_rate": float(1.0 - accuracy)
            }
        
        # Find minimum epsilon for successful attack
        min_epsilon = None
        for epsilon in epsilon_values:
            if vulnerability_results[f"epsilon_{epsilon}"]["success_rate"] > 0.5:
                min_epsilon = epsilon
                break
        
        return {
            "vulnerability_curve": vulnerability_results,
            "min_epsilon_for_attack": min_epsilon,
            "adversarial_vulnerability": float(1.0 - (min_epsilon or 0.3))  # Normalize to [0,1]
        }
    
    def _calculate_security_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall security metrics"""
        
        metrics = {}
        
        # Membership inference metrics
        if "membership_inference" in results:
            metrics["membership_inference_auc"] = results["membership_inference"]["auc"]
            # Convert AUC to vulnerability score (higher AUC = more vulnerable)
            metrics["membership_inference_vulnerability"] = float(
                (results["membership_inference"]["auc"] - 0.5) * 2
            )
        
        # Data poisoning metrics
        if "data_poisoning" in results:
            label_flip_acc = results["data_poisoning"]["label_flip"]["accuracy"]
            backdoor_acc = results["data_poisoning"]["backdoor"]["accuracy"]
            metrics["data_poisoning_vulnerability"] = float(1.0 - (label_flip_acc + backdoor_acc) / 2)
        
        # Adversarial vulnerability
        if "adversarial_vulnerability" in results:
            metrics["adversarial_vulnerability"] = results["adversarial_vulnerability"]["adversarial_vulnerability"]
        
        # Overall security score (lower is better)
        security_scores = []
        if "membership_inference_vulnerability" in metrics:
            security_scores.append(metrics["membership_inference_vulnerability"])
        if "data_poisoning_vulnerability" in metrics:
            security_scores.append(metrics["data_poisoning_vulnerability"])
        if "adversarial_vulnerability" in metrics:
            security_scores.append(metrics["adversarial_vulnerability"])
        
        if security_scores:
            metrics["overall_security_score"] = float(np.mean(security_scores))
        
        return metrics
    
    def generate_plots(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Generate visualization plots for security results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Membership inference ROC curve
        if "membership_inference" in results:
            roc_data = results["membership_inference"]["roc_curve"]
            plt.figure(figsize=(8, 6))
            plt.plot(roc_data["fpr"], roc_data["tpr"], label=f'AUC = {results["membership_inference"]["auc"]:.3f}')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Membership Inference ROC Curve')
            plt.legend()
            plt.grid(True)
            if save_path:
                plt.savefig(f"{save_path}/membership_inference_roc.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # Adversarial vulnerability curve
        if "adversarial_vulnerability" in results:
            vuln_curve = results["adversarial_vulnerability"]["vulnerability_curve"]
            epsilons = [float(k.split('_')[1]) for k in vuln_curve.keys()]
            success_rates = [vuln_curve[k]["success_rate"] for k in vuln_curve.keys()]
            
            plt.figure(figsize=(8, 6))
            plt.plot(epsilons, success_rates, marker='o')
            plt.xlabel('Epsilon')
            plt.ylabel('Attack Success Rate')
            plt.title('Adversarial Vulnerability Curve')
            plt.grid(True)
            if save_path:
                plt.savefig(f"{save_path}/adversarial_vulnerability.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # Security metrics comparison
        security_metrics = {}
        if "membership_inference_vulnerability" in results:
            security_metrics["Membership\nInference"] = results["membership_inference_vulnerability"]
        if "data_poisoning_vulnerability" in results:
            security_metrics["Data\nPoisoning"] = results["data_poisoning_vulnerability"]
        if "adversarial_vulnerability" in results:
            security_metrics["Adversarial\nVulnerability"] = results["adversarial_vulnerability"]
        
        if security_metrics:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(security_metrics.keys(), security_metrics.values(), color='red')
            plt.title('Security Vulnerability Scores')
            plt.ylabel('Vulnerability Score')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, security_metrics.values()):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/security_vulnerabilities.png", dpi=300, bbox_inches='tight')
            plt.show() 