# Getting Started

This guide will walk you through installing Conformal-Drift and running your first audit.

## Installation

### Basic Installation

```bash
pip install conformal-drift
```

### With Full Dependencies

For running experiments with transformers and embeddings:

```bash
pip install conformal-drift[full]
```

### Development Installation

```bash
git clone https://github.com/debu-sinha/conformaldrift.git
cd conformaldrift
pip install -e ".[dev]"
```

## Prerequisites

Before using Conformal-Drift, you need:

1. **A trained model** that produces predictions
2. **Nonconformity scores** from a calibration set
3. **Test data** to evaluate under shift

## Your First Audit

### Step 1: Prepare Calibration Scores

Conformal prediction requires nonconformity scores from a held-out calibration set:

```python
import numpy as np

# Example: Using 1 - predicted probability as nonconformity score
calibration_scores = 1 - model.predict_proba(X_calibration)[:, true_labels]
```

### Step 2: Initialize the Auditor

```python
from conformal_drift import ConformalDriftAuditor

auditor = ConformalDriftAuditor(
    calibration_scores=calibration_scores,
    alpha=0.1  # Target miscoverage rate (90% coverage)
)
```

### Step 3: Define Your Test Data

```python
# Your test samples with ground truth
test_data = {
    'inputs': test_inputs,
    'labels': test_labels,
    'scores': 1 - model.predict_proba(test_inputs)[:, test_labels]
}
```

### Step 4: Run the Audit

```python
# Audit with temporal shift simulation
results = auditor.audit(
    test_data=test_data,
    shift_type="temporal",
    shift_intensity=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
)
```

### Step 5: Analyze Results

```python
# Print coverage at each shift level
for intensity, coverage in zip(results.shift_intensities, results.coverage):
    print(f"Shift {intensity:.0%}: Coverage = {coverage:.3f}")

# Check if coverage guarantee holds
nominal_coverage = 1 - 0.1  # 90%
for i, cov in enumerate(results.coverage):
    gap = nominal_coverage - cov
    if gap > 0.05:
        print(f"Warning: Coverage gap of {gap:.1%} at shift {results.shift_intensities[i]:.0%}")
```

### Step 6: Visualize Results

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(results.shift_intensities, results.coverage, 'b-o', linewidth=2)
plt.axhline(y=0.9, color='r', linestyle='--', label='Nominal 90%')
plt.xlabel('Shift Intensity')
plt.ylabel('Empirical Coverage')
plt.title('Coverage Degradation Under Distribution Shift')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('coverage_curve.png', dpi=150)
plt.show()
```

## Next Steps

- Learn about [Core Concepts](guide/concepts.md) behind conformal prediction
- Explore different [Shift Operators](guide/shift-operators.md)
- See more [Examples](examples.md) with real datasets

## Common Issues

### ImportError: No module named 'conformal_drift'

Make sure you installed the package:
```bash
pip install conformal-drift
```

### Coverage is always 1.0 or 0.0

Check that your nonconformity scores have appropriate variance. If all scores are identical, conformal prediction cannot distinguish samples.

### Out of memory with large datasets

Use batch processing:
```python
results = auditor.audit(
    test_data=test_data,
    batch_size=1000
)
```
