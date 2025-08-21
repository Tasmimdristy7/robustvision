# RobustVision Installation Guide

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/robustvision/robustvision.git
cd robustvision
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install RobustVision

```bash
pip install -e .
```

### 4. Verify Installation

```bash
python -c "import robustvision; print('RobustVision installed successfully!')"
```

## Detailed Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Step-by-Step Installation

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv robustvision_env
   source robustvision_env/bin/activate  # On Windows: robustvision_env\Scripts\activate
   ```

2. **Install PyTorch:**
   ```bash
   # For CPU only
   pip install torch torchvision torchaudio
   
   # For CUDA (adjust cuda version as needed)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install RobustVision:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Test the installation:**
   ```bash
   python tests/test_basic.py
   ```

## Usage Examples

### Command Line Interface

```bash
# Basic test run
robustvision test --model resnet18 --dataset cifar10 --output-dir ./results

# Quick test
robustvision test --model resnet50 --dataset cifar10 --quick

# Custom test suite
robustvision test --model resnet18 --dataset cifar10 --tests correctness,robustness

# With custom configuration
robustvision test --model resnet18 --dataset cifar10 --config configs/default.yaml
```

### Python API

```python
from robustvision import RobustVisionTestbench
from robustvision.models import load_model
from robustvision.datasets import load_dataset

# Initialize testbench
testbench = RobustVisionTestbench()

# Load model and dataset
model = load_model("resnet18", pretrained=True)
dataset = load_dataset("cifar10", split="test")

# Run tests
results = testbench.run_tests(
    model=model,
    dataset=dataset,
    test_suites=["correctness", "robustness", "security"]
)

# Generate reports
testbench.generate_report(results, "./results")
```

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'robustvision'**
   - Make sure you've installed the package: `pip install -e .`
   - Check that you're in the correct virtual environment

2. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use fewer samples: `--num-samples 1000`
   - Run on CPU: set `device='cpu'` in code

3. **Dataset Download Issues**
   - Check internet connection
   - Verify dataset paths in configuration
   - For ImageNet, ensure you have the dataset downloaded

4. **Missing Dependencies**
   - Install missing packages: `pip install package_name`
   - Update requirements: `pip install -r requirements.txt --upgrade`

### Getting Help

- Check the [README.md](README.md) for detailed documentation
- Run `robustvision --help` for CLI options
- Open an issue on GitHub for bugs or feature requests

## Development Setup

For developers who want to contribute:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black robustvision/

# Type checking
mypy robustvision/
```

## Supported Platforms

- **Operating Systems:** Linux, macOS, Windows
- **Python Versions:** 3.8, 3.9, 3.10, 3.11
- **PyTorch Versions:** 2.0+
- **CUDA Versions:** 11.8+ (optional) 