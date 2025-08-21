"""
Basic usage example for RobustVision
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from robustvision import RobustVisionTestbench
from robustvision.models import load_model
from robustvision.datasets import load_dataset

def main():
    """Basic example of using RobustVision"""
    
    print("RobustVision Basic Usage Example")
    print("=" * 50)
    
    # Initialize testbench
    print("Initializing RobustVision testbench...")
    testbench = RobustVisionTestbench()
    
    # Load a pre-trained model
    print("Loading ResNet-18 model...")
    model = load_model("resnet18", pretrained=True)
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    dataset = load_dataset("cifar10", split="test", num_samples=1000)  # Use 1000 samples for demo
    
    # Run comprehensive tests
    print("Running comprehensive tests...")
    results = testbench.run_tests(
        model=model,
        dataset=dataset,
        test_suites=["correctness", "robustness", "security"],
        output_dir="./example_results"
    )
    
    # Generate reports
    print("Generating reports...")
    testbench.generate_report(results, "./example_results")
    
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
    
    print(f"\nResults saved to: ./example_results")
    print("Check the generated HTML and Markdown reports for detailed analysis!")

def quick_test_example():
    """Quick test example for rapid evaluation"""
    
    print("\nRobustVision Quick Test Example")
    print("=" * 50)
    
    # Initialize testbench
    testbench = RobustVisionTestbench()
    
    # Load model and dataset
    model = load_model("resnet18", pretrained=True)
    dataset = load_dataset("cifar10", split="test", num_samples=500)
    
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
    # Run basic example
    main()
    
    # Run quick test example
    quick_test_example() 