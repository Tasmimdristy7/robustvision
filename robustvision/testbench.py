"""
Main testbench class for RobustVision
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .tests.correctness import CorrectnessTests
from .tests.robustness import RobustnessTests
from .tests.security import SecurityTests
from .reporting.report_generator import ReportGenerator
from .utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class TestConfig:
    """Configuration for test execution"""
    correctness: Dict[str, Any]
    robustness: Dict[str, Any]
    security: Dict[str, Any]
    reporting: Dict[str, Any]

class RobustVisionTestbench:
    """
    Main testbench class for comprehensive vision model evaluation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the testbench
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize test suites
        self.correctness_tests = CorrectnessTests(self.config.correctness)
        self.robustness_tests = RobustnessTests(self.config.robustness)
        self.security_tests = SecurityTests(self.config.security)
        
        # Initialize report generator
        self.report_generator = ReportGenerator(self.config.reporting)
        
        logger.info(f"RobustVision testbench initialized on device: {self.device}")
    
    def _load_config(self, config_path: Optional[str]) -> TestConfig:
        """Load configuration from file or use defaults"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = self._get_default_config()
        
        return TestConfig(**config_dict)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "correctness": {
                "enabled": True,
                "metrics": ["accuracy", "ece", "confusion_matrix"],
                "batch_size": 32
            },
            "robustness": {
                "enabled": True,
                "corruptions": ["gaussian_noise", "motion_blur", "brightness", "contrast"],
                "attacks": ["fgsm", "pgd", "cw"],
                "attack_params": {
                    "fgsm": {"epsilon": 0.3},
                    "pgd": {"epsilon": 0.3, "alpha": 0.01, "steps": 40},
                    "cw": {"c": 1.0, "steps": 1000}
                }
            },
            "security": {
                "enabled": True,
                "membership_inference": True,
                "data_poisoning": True,
                "adversarial_vulnerability": True
            },
            "reporting": {
                "format": ["html", "markdown"],
                "include_plots": True,
                "risk_score_weights": {
                    "correctness": 0.3,
                    "robustness": 0.4,
                    "security": 0.3
                }
            }
        }
    
    def run_tests(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        test_suites: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive tests on the model
        
        Args:
            model: PyTorch model to test
            dataset: Dataset for testing
            test_suites: List of test suites to run (correctness, robustness, security)
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing all test results
        """
        if test_suites is None:
            test_suites = ["correctness", "robustness", "security"]
        
        model = model.to(self.device)
        model.eval()
        
        # Create data loader
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.correctness.get("batch_size", 32),
            shuffle=False,
            num_workers=4
        )
        
        results = {
            "model_info": self._get_model_info(model),
            "dataset_info": self._get_dataset_info(dataset),
            "test_config": asdict(self.config),
            "results": {}
        }
        
        logger.info("Starting comprehensive model evaluation...")
        
        # Run correctness tests
        if "correctness" in test_suites and self.config.correctness["enabled"]:
            logger.info("Running correctness tests...")
            results["results"]["correctness"] = self.correctness_tests.run(
                model, dataloader, self.device
            )
        
        # Run robustness tests
        if "robustness" in test_suites and self.config.robustness["enabled"]:
            logger.info("Running robustness tests...")
            results["results"]["robustness"] = self.robustness_tests.run(
                model, dataloader, self.device
            )
        
        # Run security tests
        if "security" in test_suites and self.config.security["enabled"]:
            logger.info("Running security tests...")
            results["results"]["security"] = self.security_tests.run(
                model, dataloader, self.device
            )
        
        # Calculate overall risk score
        results["risk_score"] = self._calculate_risk_score(results["results"])
        
        # Save results
        if output_dir:
            self._save_results(results, output_dir)
        
        logger.info("Model evaluation completed!")
        return results
    
    def _get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "name": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device)
        }
    
    def _get_dataset_info(self, dataset: torch.utils.data.Dataset) -> Dict[str, Any]:
        """Extract dataset information"""
        return {
            "name": dataset.__class__.__name__,
            "size": len(dataset),
            "num_classes": getattr(dataset, 'num_classes', 'Unknown')
        }
    
    def _calculate_risk_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall risk score from test results"""
        weights = self.config.reporting["risk_score_weights"]
        score = 0.0
        
        # Correctness component (lower is better)
        if "correctness" in results:
            acc = results["correctness"].get("accuracy", 0.0)
            ece = results["correctness"].get("ece", 1.0)
            correctness_score = (1 - acc) * 0.7 + ece * 0.3
            score += correctness_score * weights["correctness"]
        
        # Robustness component (lower is better)
        if "robustness" in results:
            corruption_acc = results["robustness"].get("corruption_accuracy", 0.0)
            attack_success = results["robustness"].get("attack_success_rate", 1.0)
            robustness_score = (1 - corruption_acc) * 0.5 + attack_success * 0.5
            score += robustness_score * weights["robustness"]
        
        # Security component (lower is better)
        if "security" in results:
            membership_auc = results["security"].get("membership_inference_auc", 0.5)
            adv_vulnerability = results["security"].get("adversarial_vulnerability", 1.0)
            security_score = (membership_auc - 0.5) * 2 * 0.5 + adv_vulnerability * 0.5
            score += security_score * weights["security"]
        
        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Save results to output directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(output_path / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    def generate_report(
        self, 
        results: Dict[str, Any], 
        output_dir: str,
        formats: Optional[List[str]] = None
    ):
        """
        Generate comprehensive report from test results
        
        Args:
            results: Test results dictionary
            output_dir: Directory to save reports
            formats: List of report formats (html, markdown)
        """
        if formats is None:
            formats = self.config.reporting["format"]
        
        self.report_generator.generate_report(results, output_dir, formats)
        logger.info(f"Reports generated in {output_dir}")
    
    def run_quick_test(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a quick subset of tests for rapid evaluation
        
        Args:
            model: PyTorch model to test
            dataset: Dataset for testing
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing quick test results
        """
        # Create a quick test configuration
        quick_config = self._get_default_config()
        quick_config["correctness"]["metrics"] = ["accuracy"]
        quick_config["robustness"]["corruptions"] = ["gaussian_noise"]
        quick_config["robustness"]["attacks"] = ["fgsm"]
        quick_config["security"]["membership_inference"] = False
        quick_config["security"]["data_poisoning"] = False
        
        # Temporarily update config
        original_config = self.config
        self.config = TestConfig(**quick_config)
        
        try:
            results = self.run_tests(
                model, dataset, 
                test_suites=["correctness", "robustness"], 
                output_dir=output_dir
            )
        finally:
            # Restore original config
            self.config = original_config
        
        return results 