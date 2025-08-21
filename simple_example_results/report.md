# RobustVision Test Report

**Generated on:** 2025-08-22 02:06:57

## Overall Risk Score: 0.693

## Model Information

| Property | Value |
|----------|-------|
| Model Name | SimpleModel |
| Total Parameters | 2,442 |
| Device | cpu |

## Dataset Information

| Property | Value |
|----------|-------|
| Dataset Name | TensorDataset |
| Number of Samples | 100 |
| Number of Classes | Unknown |


## Correctness Tests

| Metric | Value |
|--------|-------|
| Accuracy | 0.110 |
| Expected Calibration Error (ECE) | 0.020 |


## Robustness Tests

| Metric | Value |
|--------|-------|
| Corruption Accuracy | 0.112 |
| Attack Success Rate | 0.890 |
| Average Perturbation | 42.561 |


## Security Tests

| Metric | Value |
|--------|-------|
| Membership Inference AUC | 0.500 |
| Adversarial Vulnerability | 0.990 |
| Data Poisoning Vulnerability | 0.890 |



## Visualizations

The following plots are available in the `plots/` directory:

- Confusion Matrix
- Calibration Curve  
- Corruption Accuracy Comparison
- Adversarial Attack Success Rates
- Risk Score Breakdown

