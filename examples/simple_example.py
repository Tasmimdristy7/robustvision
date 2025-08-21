"""
Simple RobustVision example that works without downloading pre-trained models
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from robustvision import RobustVisionTestbench

class SimpleModel(nn.Module):
    """Simple test model"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    """Simple example of using RobustVision"""
    
    print("RobustVision Simple Example")
    print("=" * 50)
    
    # Initialize testbench
    print("Initializing RobustVision testbench...")
    testbench = RobustVisionTestbench()
    
    # Create a simple model
    print("Creating simple model...")
    model = SimpleModel(num_classes=10)
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    batch_size = 32
    num_samples = 100
    data = torch.randn(num_samples, 3, 32, 32)
    targets = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(data, targets)
    
    # Run comprehensive tests
    print("Running comprehensive tests...")
    results = testbench.run_tests(
        model=model,
        dataset=dataset,
        test_suites=["correctness", "robustness", "security"],
        output_dir="./simple_example_results"
    )
    
    # Generate reports
    print("Generating reports...")
    testbench.generate_report(results, "./simple_example_results")
    
    # Print summary
    print("\nTest Summary:")
    print(f"Overall Risk Score: {results['risk_score']:.3f}")
    
    if "correctness" in results["results"]:
        acc = results["results"]["correctness"]["accuracy"]
        ece = results["results"]["correctness"]["ece"]
        print(f"Accuracy: {acc:.3f}")
        print(f"Expected Calibration Error: {ece:.3f}")
    
    if "robustness" in results["results"]:
        corr_acc = results["results"]["robustness"].get("corruption_accuracy", 0.0)
        attack_success = results["results"]["robustness"].get("attack_success_rate", 0.0)
        print(f"Corruption Accuracy: {corr_acc:.3f}")
        print(f"Attack Success Rate: {attack_success:.3f}")
    
    if "security" in results["results"]:
        membership_auc = results["results"]["security"].get("membership_inference_auc", 0.0)
        adv_vuln = results["results"]["security"].get("adversarial_vulnerability", 0.0)
        print(f"Membership Inference AUC: {membership_auc:.3f}")
        print(f"Adversarial Vulnerability: {adv_vuln:.3f}")
    
    print(f"\nResults saved to: ./simple_example_results")
    print("Check the generated HTML and Markdown reports for detailed analysis!")

def quick_test_example():
    """Quick test example for rapid evaluation"""
    
    print("\nRobustVision Quick Test Example")
    print("=" * 50)
    
    # Initialize testbench
    testbench = RobustVisionTestbench()
    
    # Create model and dataset
    model = SimpleModel(num_classes=5)
    data = torch.randn(50, 3, 32, 32)
    targets = torch.randint(0, 5, (50,))
    dataset = TensorDataset(data, targets)
    
    # Run quick test
    print("Running quick test...")
    results = testbench.run_quick_test(
        model=model,
        dataset=dataset,
        output_dir="./quick_test_results"
    )
    
    # Print results
    print(f"Quick Test Risk Score: {results['risk_score']:.3f}")
    print(f"Results saved to: ./quick_test_results")

if __name__ == "__main__":
    # Run simple example
    main()
    
    # Run quick test example
    quick_test_example() 