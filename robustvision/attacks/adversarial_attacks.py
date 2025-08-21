"""
Adversarial attack implementations for RobustVision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

class AdversarialAttacks:
    """Adversarial attack implementations"""
    
    def __init__(self):
        """Initialize adversarial attacks"""
        pass
    
    def fgsm_attack(
        self, 
        model: nn.Module, 
        data: torch.Tensor, 
        target: torch.Tensor, 
        epsilon: float = 0.3,
        **kwargs
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack
        
        Args:
            model: PyTorch model
            data: Input data
            target: Target labels
            epsilon: Perturbation magnitude
            
        Returns:
            Adversarial examples
        """
        data.requires_grad_(True)
        
        # Forward pass
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Create perturbation
        perturbation = epsilon * torch.sign(data.grad.data)
        
        # Add perturbation to data
        adversarial_data = data + perturbation
        
        # Clip to valid range [0, 1]
        adversarial_data = torch.clamp(adversarial_data, 0, 1)
        
        return adversarial_data.detach()
    
    def pgd_attack(
        self, 
        model: nn.Module, 
        data: torch.Tensor, 
        target: torch.Tensor, 
        epsilon: float = 0.3,
        alpha: float = 0.01,
        steps: int = 40,
        **kwargs
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack
        
        Args:
            model: PyTorch model
            data: Input data
            target: Target labels
            epsilon: Maximum perturbation magnitude
            alpha: Step size
            steps: Number of steps
            
        Returns:
            Adversarial examples
        """
        # Initialize perturbation
        perturbation = torch.zeros_like(data)
        perturbation.uniform_(-epsilon, epsilon)
        
        # Project to valid range
        adversarial_data = torch.clamp(data + perturbation, 0, 1)
        
        for step in range(steps):
            adversarial_data.requires_grad_(True)
            adversarial_data.retain_grad()
            
            # Forward pass
            output = model(adversarial_data)
            loss = F.cross_entropy(output, target)
            
            # Backward pass
            loss.backward(retain_graph=True)
            
            # Update perturbation
            if adversarial_data.grad is not None:
                perturbation = alpha * torch.sign(adversarial_data.grad.data)
                adversarial_data = adversarial_data + perturbation
            
            # Project to epsilon ball
            delta = adversarial_data - data
            delta = torch.clamp(delta, -epsilon, epsilon)
            adversarial_data = torch.clamp(data + delta, 0, 1)
            
            # Zero gradients
            if adversarial_data.grad is not None:
                adversarial_data.grad.zero_()
        
        return adversarial_data.detach()
    
    def cw_attack(
        self, 
        model: nn.Module, 
        data: torch.Tensor, 
        target: torch.Tensor, 
        c: float = 1.0,
        steps: int = 1000,
        lr: float = 0.01,
        **kwargs
    ) -> torch.Tensor:
        """
        Carlini & Wagner (CW) attack
        
        Args:
            model: PyTorch model
            data: Input data
            target: Target labels
            c: Confidence parameter
            steps: Number of optimization steps
            lr: Learning rate
            
        Returns:
            Adversarial examples
        """
        # Initialize perturbation
        perturbation = torch.zeros_like(data, requires_grad=True)
        optimizer = torch.optim.Adam([perturbation], lr=lr)
        
        for step in range(steps):
            # Create adversarial example
            adversarial_data = data + perturbation
            adversarial_data = torch.clamp(adversarial_data, 0, 1)
            
            # Forward pass
            output = model(adversarial_data)
            
            # Calculate loss
            target_logit = output[torch.arange(output.size(0)), target]
            other_logits = output.clone()
            other_logits[torch.arange(output.size(0)), target] = -float('inf')
            max_other_logit = torch.max(other_logits, dim=1)[0]
            
            # CW loss
            loss = torch.clamp(max_other_logit - target_logit + c, min=0)
            loss = loss.sum()
            
            # L2 regularization
            l2_loss = torch.norm(perturbation, p=2, dim=(1, 2, 3)).sum()
            total_loss = loss + 0.01 * l2_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Project perturbation to valid range
            perturbation.data = torch.clamp(perturbation.data, -0.5, 0.5)
        
        # Final adversarial example
        adversarial_data = data + perturbation
        adversarial_data = torch.clamp(adversarial_data, 0, 1)
        
        return adversarial_data.detach()
    
    def deepfool_attack(
        self, 
        model: nn.Module, 
        data: torch.Tensor, 
        target: torch.Tensor, 
        max_iter: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """
        DeepFool attack
        
        Args:
            model: PyTorch model
            data: Input data
            target: Target labels
            max_iter: Maximum iterations
            
        Returns:
            Adversarial examples
        """
        batch_size = data.size(0)
        adversarial_data = data.clone()
        
        for i in range(batch_size):
            x = adversarial_data[i:i+1].clone()
            x.requires_grad_(True)
            
            for iter_idx in range(max_iter):
                # Forward pass
                output = model(x)
                
                # Check if already misclassified
                pred = torch.argmax(output, dim=1)
                if pred != target[i]:
                    break
                
                # Calculate gradients for all classes
                grads = []
                for class_idx in range(output.size(1)):
                    x.grad.zero_()
                    output[0, class_idx].backward(retain_graph=True)
                    grads.append(x.grad.data.clone())
                
                grads = torch.stack(grads)
                
                # Find closest hyperplane
                current_class = target[i]
                other_classes = [j for j in range(output.size(1)) if j != current_class]
                
                min_dist = float('inf')
                min_grad = None
                
                for other_class in other_classes:
                    grad_diff = grads[other_class] - grads[current_class]
                    output_diff = output[0, other_class] - output[0, current_class]
                    
                    dist = abs(output_diff) / (torch.norm(grad_diff) + 1e-8)
                    
                    if dist < min_dist:
                        min_dist = dist
                        min_grad = grad_diff
                
                # Update perturbation
                if min_grad is not None:
                    perturbation = min_dist * min_grad / (torch.norm(min_grad) + 1e-8)
                    x = x + perturbation
                    x = torch.clamp(x, 0, 1)
            
            adversarial_data[i] = x.detach()
        
        return adversarial_data
    
    def boundary_attack(
        self, 
        model: nn.Module, 
        data: torch.Tensor, 
        target: torch.Tensor, 
        max_iter: int = 1000,
        **kwargs
    ) -> torch.Tensor:
        """
        Boundary attack (decision-based)
        
        Args:
            model: PyTorch model
            data: Input data
            target: Target labels
            max_iter: Maximum iterations
            
        Returns:
            Adversarial examples
        """
        batch_size = data.size(0)
        adversarial_data = data.clone()
        
        for i in range(batch_size):
            x = data[i:i+1].clone()
            
            # Initialize with random noise
            noise = torch.randn_like(x) * 0.1
            current_adv = x + noise
            current_adv = torch.clamp(current_adv, 0, 1)
            
            for iter_idx in range(max_iter):
                # Check if current point is adversarial
                with torch.no_grad():
                    output = model(current_adv)
                    pred = torch.argmax(output, dim=1)
                
                if pred != target[i]:
                    break
                
                # Generate random direction
                direction = torch.randn_like(x)
                direction = direction / torch.norm(direction)
                
                # Binary search to find boundary
                low, high = 0.0, 1.0
                for _ in range(10):
                    mid = (low + high) / 2
                    test_point = x + mid * direction
                    test_point = torch.clamp(test_point, 0, 1)
                    
                    with torch.no_grad():
                        test_output = model(test_point)
                        test_pred = torch.argmax(test_output, dim=1)
                    
                    if test_pred == target[i]:
                        low = mid
                    else:
                        high = mid
                
                # Update current adversarial example
                current_adv = x + high * direction
                current_adv = torch.clamp(current_adv, 0, 1)
            
            adversarial_data[i] = current_adv
        
        return adversarial_data
    
    def get_attack_function(self, attack_name: str):
        """Get attack function by name"""
        attack_functions = {
            "fgsm": self.fgsm_attack,
            "pgd": self.pgd_attack,
            "cw": self.cw_attack,
            "deepfool": self.deepfool_attack,
            "boundary": self.boundary_attack
        }
        
        if attack_name not in attack_functions:
            raise ValueError(f"Unknown attack: {attack_name}")
        
        return attack_functions[attack_name] 