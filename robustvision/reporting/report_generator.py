"""
Report generator for RobustVision
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

class ReportGenerator:
    """Generate comprehensive reports from test results"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize report generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.formats = config.get("format", ["html", "markdown"])
        self.include_plots = config.get("include_plots", True)
        self.risk_score_weights = config.get("risk_score_weights", {})
    
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
            output_dir: Output directory for reports
            formats: List of report formats to generate
        """
        if formats is None:
            formats = self.formats
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate plots if requested
        if self.include_plots:
            self._generate_plots(results, output_path)
        
        # Generate reports in requested formats
        for fmt in formats:
            if fmt == "html":
                self._generate_html_report(results, output_path)
            elif fmt == "markdown":
                self._generate_markdown_report(results, output_path)
            else:
                logger.warning(f"Unknown report format: {fmt}")
        
        logger.info(f"Reports generated in {output_path}")
    
    def _generate_plots(self, results: Dict[str, Any], output_path: Path):
        """Generate visualization plots"""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create plots directory
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Correctness plots
        if "correctness" in results.get("results", {}):
            self._generate_correctness_plots(results["results"]["correctness"], plots_dir)
        
        # Robustness plots
        if "robustness" in results.get("results", {}):
            self._generate_robustness_plots(results["results"]["robustness"], plots_dir)
        
        # Security plots
        if "security" in results.get("results", {}):
            self._generate_security_plots(results["results"]["security"], plots_dir)
        
        # Overall summary plots
        self._generate_summary_plots(results, plots_dir)
    
    def _generate_correctness_plots(self, correctness_results: Dict[str, Any], plots_dir: Path):
        """Generate correctness test plots"""
        
        # Confusion Matrix
        if "confusion_matrix" in correctness_results:
            cm = np.array(correctness_results["confusion_matrix"]["matrix"])
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Calibration Curve
        if "calibration_curve" in correctness_results:
            cal_curve = correctness_results["calibration_curve"]
            plt.figure(figsize=(8, 6))
            plt.plot(cal_curve["mean_predicted_value"], cal_curve["fraction_of_positives"], 
                    marker='o', label='Model')
            plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plots_dir / "calibration_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Per-class Accuracy
        if "per_class_accuracy" in correctness_results:
            per_class_acc = correctness_results["per_class_accuracy"]
            classes = list(per_class_acc.keys())
            accuracies = list(per_class_acc.values())
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(classes, accuracies)
            plt.title('Per-Class Accuracy')
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(plots_dir / "per_class_accuracy.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_robustness_plots(self, robustness_results: Dict[str, Any], plots_dir: Path):
        """Generate robustness test plots"""
        
        # Corruption accuracy comparison
        if "corruptions" in robustness_results:
            corruptions = list(robustness_results["corruptions"].keys())
            accuracies = [robustness_results["corruptions"][c]["accuracy"] for c in corruptions]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(corruptions, accuracies)
            plt.title('Model Accuracy Under Different Corruptions')
            plt.xlabel('Corruption Type')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / "corruption_accuracy.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Adversarial attack success rates
        if "adversarial" in robustness_results:
            attacks = list(robustness_results["adversarial"].keys())
            success_rates = [robustness_results["adversarial"][a]["success_rate"] for a in attacks]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(attacks, success_rates, color='red')
            plt.title('Adversarial Attack Success Rates')
            plt.xlabel('Attack Type')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / "attack_success_rates.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_security_plots(self, security_results: Dict[str, Any], plots_dir: Path):
        """Generate security test plots"""
        
        # Membership inference ROC curve
        if "membership_inference" in security_results:
            roc_data = security_results["membership_inference"]["roc_curve"]
            plt.figure(figsize=(8, 6))
            plt.plot(roc_data["fpr"], roc_data["tpr"], 
                    label=f'AUC = {security_results["membership_inference"]["auc"]:.3f}')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Membership Inference ROC Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plots_dir / "membership_inference_roc.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Adversarial vulnerability curve
        if "adversarial_vulnerability" in security_results and isinstance(security_results["adversarial_vulnerability"], dict):
            vuln_curve = security_results["adversarial_vulnerability"].get("vulnerability_curve", {})
            if vuln_curve:
                epsilons = [float(k.split('_')[1]) for k in vuln_curve.keys()]
                success_rates = [vuln_curve[k]["success_rate"] for k in vuln_curve.keys()]
                
                plt.figure(figsize=(8, 6))
                plt.plot(epsilons, success_rates, marker='o')
                plt.xlabel('Epsilon')
                plt.ylabel('Attack Success Rate')
                plt.title('Adversarial Vulnerability Curve')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plots_dir / "adversarial_vulnerability.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def _generate_summary_plots(self, results: Dict[str, Any], plots_dir: Path):
        """Generate summary plots"""
        
        # Risk score breakdown
        risk_scores = {}
        if "correctness" in results.get("results", {}):
            risk_scores["Correctness"] = 1.0 - results["results"]["correctness"].get("accuracy", 0.0)
        if "robustness" in results.get("results", {}):
            risk_scores["Robustness"] = results["results"]["robustness"].get("overall_robustness_score", 0.0)
        if "security" in results.get("results", {}):
            risk_scores["Security"] = results["results"]["security"].get("overall_security_score", 0.0)
        
        if risk_scores:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(risk_scores.keys(), risk_scores.values(), color=['blue', 'orange', 'red'])
            plt.title('Risk Score Breakdown')
            plt.ylabel('Risk Score')
            plt.ylim(0, 1)
            
            # Add overall risk score
            overall_risk = results.get("risk_score", 0.0)
            plt.axhline(y=overall_risk, color='black', linestyle='--', 
                       label=f'Overall Risk: {overall_risk:.3f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "risk_score_breakdown.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_html_report(self, results: Dict[str, Any], output_path: Path):
        """Generate HTML report"""
        
        html_content = self._create_html_content(results)
        
        with open(output_path / "report.html", 'w') as f:
            f.write(html_content)
    
    def _generate_markdown_report(self, results: Dict[str, Any], output_path: Path):
        """Generate Markdown report"""
        
        md_content = self._create_markdown_content(results)
        
        with open(output_path / "report.md", 'w') as f:
            f.write(md_content)
    
    def _create_html_content(self, results: Dict[str, Any]) -> str:
        """Create HTML report content"""
        
        # Get model and dataset info
        model_info = results.get("model_info", {})
        dataset_info = results.get("dataset_info", {})
        risk_score = results.get("risk_score", 0.0)
        
        # Create HTML template
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RobustVision Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .risk-score {{
            font-size: 2em;
            font-weight: bold;
            color: {'red' if risk_score > 0.7 else 'orange' if risk_score > 0.4 else 'green'};
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #007bff;
        }}
        .plot {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RobustVision Test Report</h1>
            <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <div class="risk-score">Overall Risk Score: {risk_score:.3f}</div>
        </div>
        
        <div class="section">
            <h2>Model Information</h2>
            <div class="metric">
                <div>Model Name</div>
                <div class="metric-value">{model_info.get('name', 'Unknown')}</div>
            </div>
            <div class="metric">
                <div>Parameters</div>
                <div class="metric-value">{model_info.get('total_parameters', 0):,}</div>
            </div>
            <div class="metric">
                <div>Device</div>
                <div class="metric-value">{model_info.get('device', 'Unknown')}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Dataset Information</h2>
            <div class="metric">
                <div>Dataset</div>
                <div class="metric-value">{dataset_info.get('name', 'Unknown')}</div>
            </div>
            <div class="metric">
                <div>Samples</div>
                <div class="metric-value">{dataset_info.get('size', 0):,}</div>
            </div>
            <div class="metric">
                <div>Classes</div>
                <div class="metric-value">{dataset_info.get('num_classes', 'Unknown')}</div>
            </div>
        </div>
        
        {self._generate_test_sections_html(results)}
        
        {self._generate_plots_html()}
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _generate_test_sections_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML for test sections"""
        
        html_sections = ""
        test_results = results.get("results", {})
        
        # Correctness section
        if "correctness" in test_results:
            correctness = test_results["correctness"]
            html_sections += f"""
        <div class="section">
            <h2>Correctness Tests</h2>
            <div class="metric">
                <div>Accuracy</div>
                <div class="metric-value">{correctness.get('accuracy', 0.0):.3f}</div>
            </div>
            <div class="metric">
                <div>ECE</div>
                <div class="metric-value">{correctness.get('ece', 0.0):.3f}</div>
            </div>
        </div>
            """
        
        # Robustness section
        if "robustness" in test_results:
            robustness = test_results["robustness"]
            html_sections += f"""
        <div class="section">
            <h2>Robustness Tests</h2>
            <div class="metric">
                <div>Corruption Accuracy</div>
                <div class="metric-value">{robustness.get('corruption_accuracy', 0.0):.3f}</div>
            </div>
            <div class="metric">
                <div>Attack Success Rate</div>
                <div class="metric-value">{robustness.get('attack_success_rate', 0.0):.3f}</div>
            </div>
        </div>
            """
        
        # Security section
        if "security" in test_results:
            security = test_results["security"]
            html_sections += f"""
        <div class="section">
            <h2>Security Tests</h2>
            <div class="metric">
                <div>Membership Inference AUC</div>
                <div class="metric-value">{security.get('membership_inference_auc', 0.0):.3f}</div>
            </div>
            <div class="metric">
                <div>Adversarial Vulnerability</div>
                <div class="metric-value">{security.get('adversarial_vulnerability', 0.0):.3f}</div>
            </div>
        </div>
            """
        
        return html_sections
    
    def _generate_plots_html(self) -> str:
        """Generate HTML for plots"""
        
        plots_html = """
        <div class="section">
            <h2>Visualizations</h2>
        """
        
        plot_files = [
            "confusion_matrix.png",
            "calibration_curve.png", 
            "corruption_accuracy.png",
            "attack_success_rates.png",
            "risk_score_breakdown.png"
        ]
        
        for plot_file in plot_files:
            plots_html += f"""
            <div class="plot">
                <h3>{plot_file.replace('_', ' ').replace('.png', '').title()}</h3>
                <img src="plots/{plot_file}" alt="{plot_file}">
            </div>
            """
        
        plots_html += """
        </div>
        """
        
        return plots_html
    
    def _create_markdown_content(self, results: Dict[str, Any]) -> str:
        """Create Markdown report content"""
        
        # Get model and dataset info
        model_info = results.get("model_info", {})
        dataset_info = results.get("dataset_info", {})
        risk_score = results.get("risk_score", 0.0)
        
        md_content = f"""# RobustVision Test Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Risk Score: {risk_score:.3f}

## Model Information

| Property | Value |
|----------|-------|
| Model Name | {model_info.get('name', 'Unknown')} |
| Total Parameters | {model_info.get('total_parameters', 0):,} |
| Device | {model_info.get('device', 'Unknown')} |

## Dataset Information

| Property | Value |
|----------|-------|
| Dataset Name | {dataset_info.get('name', 'Unknown')} |
| Number of Samples | {dataset_info.get('size', 0):,} |
| Number of Classes | {dataset_info.get('num_classes', 'Unknown')} |

{self._generate_test_sections_markdown(results)}

## Visualizations

The following plots are available in the `plots/` directory:

- Confusion Matrix
- Calibration Curve  
- Corruption Accuracy Comparison
- Adversarial Attack Success Rates
- Risk Score Breakdown

"""
        
        return md_content
    
    def _generate_test_sections_markdown(self, results: Dict[str, Any]) -> str:
        """Generate Markdown for test sections"""
        
        md_sections = ""
        test_results = results.get("results", {})
        
        # Correctness section
        if "correctness" in test_results:
            correctness = test_results["correctness"]
            md_sections += f"""
## Correctness Tests

| Metric | Value |
|--------|-------|
| Accuracy | {correctness.get('accuracy', 0.0):.3f} |
| Expected Calibration Error (ECE) | {correctness.get('ece', 0.0):.3f} |

"""
        
        # Robustness section
        if "robustness" in test_results:
            robustness = test_results["robustness"]
            md_sections += f"""
## Robustness Tests

| Metric | Value |
|--------|-------|
| Corruption Accuracy | {robustness.get('corruption_accuracy', 0.0):.3f} |
| Attack Success Rate | {robustness.get('attack_success_rate', 0.0):.3f} |
| Average Perturbation | {robustness.get('avg_perturbation', 0.0):.3f} |

"""
        
        # Security section
        if "security" in test_results:
            security = test_results["security"]
            md_sections += f"""
## Security Tests

| Metric | Value |
|--------|-------|
| Membership Inference AUC | {security.get('membership_inference_auc', 0.0):.3f} |
| Adversarial Vulnerability | {security.get('adversarial_vulnerability', 0.0):.3f} |
| Data Poisoning Vulnerability | {security.get('data_poisoning_vulnerability', 0.0):.3f} |

"""
        
        return md_sections 