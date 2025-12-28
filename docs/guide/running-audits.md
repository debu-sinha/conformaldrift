# Running Audits

This guide covers how to configure and run comprehensive audits with Conformal-Drift.

## Basic Audit

The simplest audit tests coverage under a single shift type:

```python
from conformal_drift import ConformalDriftAuditor

auditor = ConformalDriftAuditor(
    calibration_scores=cal_scores,
    alpha=0.1
)

results = auditor.audit(
    test_data=test_data,
    shift_type="temporal"
)
```

## Configuring Shift Intensity

### Discrete Intensities

Test specific shift levels:

```python
results = auditor.audit(
    test_data=test_data,
    shift_type="temporal",
    shift_intensity=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
)
```

### Continuous Range

Generate evenly-spaced intensities:

```python
import numpy as np

results = auditor.audit(
    test_data=test_data,
    shift_type="temporal",
    shift_intensity=np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
)
```

## Multi-Shift Audits

Test multiple shift types in one audit:

```python
results = auditor.comprehensive_audit(
    test_data=test_data,
    shift_types=["temporal", "semantic", "lexical"],
    shift_intensity=[0.0, 0.25, 0.5, 0.75, 1.0]
)

# Results organized by shift type
for shift_type, shift_results in results.items():
    print(f"\n{shift_type.upper()} SHIFT:")
    print(f"  Max coverage gap: {shift_results.max_coverage_gap:.3f}")
    print(f"  Critical threshold: {shift_results.critical_intensity}")
```

## Audit Reports

Generate comprehensive reports:

```python
report = auditor.generate_report(
    test_data=test_data,
    shift_types=["temporal", "semantic", "lexical"],
    output_format="markdown"  # or "html", "json"
)

# Save report
with open("audit_report.md", "w") as f:
    f.write(report)
```

### Report Contents

- Executive summary
- Coverage curves for each shift type
- Critical failure points
- Recommendations

## Batch Processing

For large datasets, use batch processing:

```python
results = auditor.audit(
    test_data=test_data,
    shift_type="temporal",
    batch_size=1000,
    n_workers=4  # Parallel processing
)
```

## Reproducibility

Set random seeds for reproducible results:

```python
auditor = ConformalDriftAuditor(
    calibration_scores=cal_scores,
    alpha=0.1,
    random_state=42
)
```

## Saving and Loading Results

### Save Results

```python
results.save("audit_results.json")
```

### Load Results

```python
from conformal_drift import AuditResults

results = AuditResults.load("audit_results.json")
```

## Visualization

### Coverage Curves

```python
from conformal_drift.viz import plot_coverage_curve

fig = plot_coverage_curve(
    results,
    nominal_coverage=0.9,
    title="Coverage Under Temporal Shift"
)
fig.savefig("coverage_curve.png", dpi=150)
```

### Multi-Shift Comparison

```python
from conformal_drift.viz import plot_coverage_comparison

fig = plot_coverage_comparison(
    results_dict={
        "Temporal": temporal_results,
        "Semantic": semantic_results,
        "Lexical": lexical_results
    },
    nominal_coverage=0.9
)
fig.savefig("shift_comparison.png", dpi=150)
```

### Set Size Analysis

```python
from conformal_drift.viz import plot_set_size_distribution

fig = plot_set_size_distribution(results)
fig.savefig("set_sizes.png", dpi=150)
```

## Performance Optimization

### Memory Management

For very large datasets:

```python
# Stream processing mode
results = auditor.audit(
    test_data=test_data_iterator,  # Generator/iterator
    shift_type="temporal",
    streaming=True,
    checkpoint_every=10000
)
```

### GPU Acceleration

For embedding-based shifts:

```python
auditor = ConformalDriftAuditor(
    calibration_scores=cal_scores,
    device="cuda"  # Use GPU for embeddings
)
```

## Error Handling

### Common Errors

```python
try:
    results = auditor.audit(test_data, shift_type="temporal")
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
```

### Validation

Validate data before auditing:

```python
# Check data format
auditor.validate_data(test_data)

# Check calibration scores
auditor.validate_calibration()
```

## Complete Example

```python
import numpy as np
from conformal_drift import ConformalDriftAuditor
from conformal_drift.viz import plot_coverage_curve

# 1. Prepare calibration scores
cal_scores = np.load("calibration_scores.npy")

# 2. Initialize auditor
auditor = ConformalDriftAuditor(
    calibration_scores=cal_scores,
    alpha=0.1,
    random_state=42
)

# 3. Load test data
test_data = {
    'inputs': test_inputs,
    'labels': test_labels,
    'scores': test_scores,
    'timestamps': test_timestamps
}

# 4. Run comprehensive audit
shift_types = ["temporal", "semantic", "lexical"]
all_results = {}

for shift_type in shift_types:
    print(f"Running {shift_type} audit...")
    results = auditor.audit(
        test_data=test_data,
        shift_type=shift_type,
        shift_intensity=np.linspace(0, 1, 11)
    )
    all_results[shift_type] = results

    # Quick summary
    print(f"  Coverage range: {min(results.coverage):.3f} - {max(results.coverage):.3f}")
    print(f"  Max gap: {results.max_coverage_gap:.3f}")

# 5. Generate visualizations
for shift_type, results in all_results.items():
    fig = plot_coverage_curve(results, nominal_coverage=0.9)
    fig.savefig(f"coverage_{shift_type}.png", dpi=150)

# 6. Save report
report = auditor.generate_report(
    results=all_results,
    output_format="markdown"
)
with open("audit_report.md", "w") as f:
    f.write(report)

print("Audit complete!")
```
