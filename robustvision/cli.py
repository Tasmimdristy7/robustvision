"""
Command-line interface for RobustVision
"""

import click
import torch
import torchvision
from pathlib import Path
from typing import List, Optional

from .testbench import RobustVisionTestbench
from .models import load_model
from .datasets import load_dataset
from .utils.logging import get_logger

logger = get_logger(__name__)

@click.group()
@click.version_option()
def cli():
    """RobustVision: Adversarial & Reliability Testbench for Vision Models"""
    pass

@cli.command()
@click.option('--model', required=True, help='Model name or path to model file')
@click.option('--dataset', required=True, help='Dataset name or path to dataset')
@click.option('--output-dir', default='./results', help='Output directory for results')
@click.option('--tests', default='correctness,robustness,security', 
              help='Comma-separated list of test suites to run')
@click.option('--config', help='Path to configuration file')
@click.option('--batch-size', type=int, default=32, help='Batch size for evaluation')
@click.option('--num-samples', type=int, help='Number of samples to evaluate (default: all)')
@click.option('--quick', is_flag=True, help='Run quick test with reduced test suite')
def test(model, dataset, output_dir, tests, config, batch_size, num_samples, quick):
    """Run comprehensive tests on a vision model"""
    
    try:
        # Initialize testbench
        testbench = RobustVisionTestbench(config_path=config)
        
        # Load model
        logger.info(f"Loading model: {model}")
        model_obj = load_model(model)
        
        # Load dataset
        logger.info(f"Loading dataset: {dataset}")
        dataset_obj = load_dataset(dataset, num_samples=num_samples)
        
        # Parse test suites
        test_suites = [suite.strip() for suite in tests.split(',')]
        
        # Run tests
        if quick:
            logger.info("Running quick test...")
            results = testbench.run_quick_test(model_obj, dataset_obj, output_dir)
        else:
            logger.info("Running comprehensive tests...")
            results = testbench.run_tests(
                model=model_obj,
                dataset=dataset_obj,
                test_suites=test_suites,
                output_dir=output_dir
            )
        
        # Generate reports
        logger.info("Generating reports...")
        testbench.generate_report(results, output_dir)
        
        # Print summary
        print_summary(results)
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--results-dir', required=True, help='Directory containing test results')
@click.option('--output-dir', help='Output directory for reports (default: results-dir)')
@click.option('--formats', default='html,markdown', help='Comma-separated list of report formats')
def report(results_dir, output_dir, formats):
    """Generate reports from existing test results"""
    
    try:
        results_path = Path(results_dir)
        if not results_path.exists():
            raise click.ClickException(f"Results directory not found: {results_dir}")
        
        # Load results
        import json
        with open(results_path / "results.json", 'r') as f:
            results = json.load(f)
        
        # Initialize testbench for report generation
        testbench = RobustVisionTestbench()
        
        # Generate reports
        output_dir = output_dir or results_dir
        report_formats = [fmt.strip() for fmt in formats.split(',')]
        
        testbench.generate_report(results, output_dir, report_formats)
        logger.info(f"Reports generated in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--model', required=True, help='Model name to list supported models')
def list_models(model):
    """List supported models or show model details"""
    
    try:
        from .models import get_supported_models, get_model_info
        
        if model.lower() == 'all':
            models = get_supported_models()
            click.echo("Supported models:")
            for model_name in models:
                click.echo(f"  - {model_name}")
        else:
            info = get_model_info(model)
            click.echo(f"Model: {model}")
            for key, value in info.items():
                click.echo(f"  {key}: {value}")
                
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--dataset', required=True, help='Dataset name to list supported datasets')
def list_datasets(dataset):
    """List supported datasets or show dataset details"""
    
    try:
        from .datasets import get_supported_datasets, get_dataset_info
        
        if dataset.lower() == 'all':
            datasets = get_supported_datasets()
            click.echo("Supported datasets:")
            for dataset_name in datasets:
                click.echo(f"  - {dataset_name}")
        else:
            info = get_dataset_info(dataset)
            click.echo(f"Dataset: {dataset}")
            for key, value in info.items():
                click.echo(f"  {key}: {value}")
                
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise click.ClickException(str(e))

@cli.command()
def config_template():
    """Generate a configuration template file"""
    
    config_template = """# RobustVision Configuration Template

tests:
  correctness:
    enabled: true
    metrics: ["accuracy", "ece", "confusion_matrix"]
    batch_size: 32
  
  robustness:
    enabled: true
    corruptions:
      - "gaussian_noise"
      - "motion_blur"
      - "brightness"
      - "contrast"
      - "fog"
      - "snow"
    attacks:
      - "fgsm"
      - "pgd"
      - "cw"
    attack_params:
      fgsm:
        epsilon: 0.3
      pgd:
        epsilon: 0.3
        alpha: 0.01
        steps: 40
      cw:
        c: 1.0
        steps: 1000
  
  security:
    enabled: true
    membership_inference: true
    data_poisoning: true
    adversarial_vulnerability: true

reporting:
  format: ["html", "markdown"]
  include_plots: true
  risk_score_weights:
    correctness: 0.3
    robustness: 0.4
    security: 0.3
"""
    
    click.echo(config_template)

def print_summary(results: dict):
    """Print a summary of test results"""
    
    click.echo("\n" + "="*60)
    click.echo("ROBUSTVISION TEST SUMMARY")
    click.echo("="*60)
    
    # Model info
    model_info = results.get("model_info", {})
    click.echo(f"Model: {model_info.get('name', 'Unknown')}")
    click.echo(f"Parameters: {model_info.get('total_parameters', 0):,}")
    
    # Dataset info
    dataset_info = results.get("dataset_info", {})
    click.echo(f"Dataset: {dataset_info.get('name', 'Unknown')}")
    click.echo(f"Samples: {dataset_info.get('size', 0):,}")
    
    # Risk score
    risk_score = results.get("risk_score", 0.0)
    click.echo(f"Overall Risk Score: {risk_score:.3f}")
    
    # Test results summary
    test_results = results.get("results", {})
    
    if "correctness" in test_results:
        correctness = test_results["correctness"]
        click.echo(f"\nCorrectness:")
        click.echo(f"  Accuracy: {correctness.get('accuracy', 0.0):.3f}")
        click.echo(f"  ECE: {correctness.get('ece', 0.0):.3f}")
    
    if "robustness" in test_results:
        robustness = test_results["robustness"]
        click.echo(f"\nRobustness:")
        click.echo(f"  Corruption Accuracy: {robustness.get('corruption_accuracy', 0.0):.3f}")
        click.echo(f"  Attack Success Rate: {robustness.get('attack_success_rate', 0.0):.3f}")
    
    if "security" in test_results:
        security = test_results["security"]
        click.echo(f"\nSecurity:")
        click.echo(f"  Membership Inference AUC: {security.get('membership_inference_auc', 0.0):.3f}")
        click.echo(f"  Adversarial Vulnerability: {security.get('adversarial_vulnerability', 0.0):.3f}")
    
    click.echo("\n" + "="*60)

if __name__ == '__main__':
    cli() 