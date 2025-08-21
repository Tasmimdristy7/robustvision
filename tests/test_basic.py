"""
Basic tests for RobustVision
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from robustvision import RobustVisionTestbench
from robustvision.models import load_model, get_supported_models
from robustvision.datasets import load_dataset, get_supported_datasets

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

def test_imports():
    """Test that all modules can be imported"""
    from robustvision import RobustVisionTestbench
    from robustvision.models import load_model
    from robustvision.datasets import load_dataset
    from robustvision.tests.correctness import CorrectnessTests
    from robustvision.tests.robustness import RobustnessTests
    from robustvision.tests.security import SecurityTests
    
    assert True  # If we get here, imports worked

def test_supported_models():
    """Test that we can get supported models"""
    models = get_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "resnet18" in models

def test_supported_datasets():
    """Test that we can get supported datasets"""
    datasets = get_supported_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    assert "cifar10" in datasets

def test_testbench_initialization():
    """Test testbench initialization"""
    testbench = RobustVisionTestbench()
    assert testbench is not None
    assert hasattr(testbench, 'run_tests')

def test_simple_model_loading():
    """Test loading a simple model"""
    model = SimpleModel()
    assert isinstance(model, nn.Module)
    assert model.fc.out_features == 10

def test_basic_correctness_test():
    """Test basic correctness test with synthetic data"""
    # Create synthetic data
    batch_size = 16
    num_classes = 10
    data = torch.randn(batch_size, 3, 32, 32)
    targets = torch.randint(0, num_classes, (batch_size,))
    dataset = TensorDataset(data, targets)
    
    # Create model
    model = SimpleModel(num_classes)
    
    # Initialize testbench
    testbench = RobustVisionTestbench()
    
    # Run correctness test only
    results = testbench.run_tests(
        model=model,
        dataset=dataset,
        test_suites=["correctness"],
        output_dir=None
    )
    
    # Check results structure
    assert "results" in results
    assert "correctness" in results["results"]
    assert "accuracy" in results["results"]["correctness"]
    assert "ece" in results["results"]["correctness"]
    
    # Check that accuracy is reasonable
    accuracy = results["results"]["correctness"]["accuracy"]
    assert 0.0 <= accuracy <= 1.0

def test_quick_test():
    """Test quick test functionality"""
    # Create synthetic data
    batch_size = 8
    num_classes = 5
    data = torch.randn(batch_size, 3, 32, 32)
    targets = torch.randint(0, num_classes, (batch_size,))
    dataset = TensorDataset(data, targets)
    
    # Create model
    model = SimpleModel(num_classes)
    
    # Initialize testbench
    testbench = RobustVisionTestbench()
    
    # Run quick test
    results = testbench.run_quick_test(
        model=model,
        dataset=dataset,
        output_dir=None
    )
    
    # Check results structure
    assert "risk_score" in results
    assert 0.0 <= results["risk_score"] <= 1.0

if __name__ == "__main__":
    # Run tests
    test_imports()
    test_supported_models()
    test_supported_datasets()
    test_testbench_initialization()
    test_simple_model_loading()
    test_basic_correctness_test()
    test_quick_test()
    
    print("All basic tests passed!") 