# ConformalDrift

[![PyPI version](https://badge.fury.io/py/conformal-drift.svg)](https://badge.fury.io/py/conformal-drift)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Audit protocol for stress-testing conformal prediction guardrails under distribution shift.**

---

## The Problem

Conformal prediction methods promise coverage guarantees for RAG systems—but these guarantees assume exchangeability between calibration and deployment data. When this assumption fails (as it systematically does in production), what happens to your guarantees?

**Key Finding:** A system can maintain **95% coverage** while having **100% False Positive Rate**—flagging every response as a hallucination. The coverage guarantee is "satisfied" but operationally useless.

---

## Where This Fits in Your AI Pipeline

```
Training --> Calibration --> [AUDIT] --> Deployment --> Monitoring
                               ^
                          YOU ARE HERE
```

**Pipeline Stage:** After calibration, before deployment. Use ConformalDrift to validate your conformal guarantees hold under realistic distribution shifts.

### The ConformalDrift Protocol

| Phase | Action | Output |
|-------|--------|--------|
| **1. Baseline** | Calibrate on source data | Threshold τ, 95% nominal coverage |
| **2. Shift** | Apply distribution shift | RLHF / Temporal / Domain shifted data |
| **3. Measure** | Compute degradation | ΔCov, FPR@Nominal, Recalibration Interval |
| **4. Verdict** | Evaluate | PASS / MARGINAL / FAIL |

**Verdict Criteria:**
- **PASS**: |ΔCov| ≤ 5% and FPR ≤ 2× baseline
- **MARGINAL**: |ΔCov| ≤ 10%
- **FAIL**: Otherwise — do not deploy

### When to Use This Library

| Scenario | Why You Need ConformalDrift |
|----------|---------------------------|
| Deploying hallucination detection | Validate coverage holds on production data |
| Using RLHF-tuned models | RLHF changes embedding distributions |
| Cross-domain deployment | Domain shift breaks calibration |
| Temporal data (news, docs) | Knowledge drift over time |
| MLOps continuous validation | Automated pre-deployment checks |

---

## Installation

```bash
pip install conformal-drift
```

For development:
```bash
pip install conformal-drift[dev]
```

---

## Quick Start (5 Minutes)

### Option 1: Python API

```python
from conformal_drift import ConformalDriftAuditor

# Initialize auditor with target coverage
auditor = ConformalDriftAuditor(nominal_coverage=0.95)

# Phase 1: Calibrate on source distribution
auditor.calibrate(cal_scores, cal_labels)

# Phase 2-3: Test under distribution shift
auditor.add_shift("Production Data", "domain", test_scores, test_labels)

# Phase 4: Generate audit report
report = auditor.generate_report()
print(report.summary())
```

### Option 2: One-Line Quick Audit

```python
from conformal_drift import run_quick_audit

report = run_quick_audit(
    cal_scores, cal_labels,
    test_scores, test_labels,
    shift_name="RLHF Shift",
    nominal_coverage=0.95
)
print(report.summary())
```

### Option 3: Command Line

```bash
# Run interactive demo
conformal-drift demo

# Audit your own data
conformal-drift audit --cal-data calibration.npz --test-data test.npz
```

---

## Practitioner Tutorial: RAG Hallucination Detector Audit

This tutorial walks through auditing a conformal hallucination detector before production deployment.

### Step 1: Prepare Your Data

You need nonconformity scores from your hallucination detector:

```python
import numpy as np
from your_detector import HallucinationDetector

detector = HallucinationDetector()

# Calibration data (use held-out set from same distribution as training)
cal_examples = load_calibration_data()  # List of (context, response, is_hallucinated)
cal_scores = np.array([detector.score(ctx, resp) for ctx, resp, _ in cal_examples])
cal_labels = np.array([label for _, _, label in cal_examples])

# Test data (from production or shifted distribution)
test_examples = load_production_sample()
test_scores = np.array([detector.score(ctx, resp) for ctx, resp, _ in test_examples])
test_labels = np.array([label for _, _, label in test_examples])  # Manual labels or ground truth
```

### Step 2: Run the Audit

```python
from conformal_drift import ConformalDriftAuditor

auditor = ConformalDriftAuditor(nominal_coverage=0.95)

# Calibrate threshold
cal_result = auditor.calibrate(cal_scores, cal_labels)
print(f"Calibration threshold: {cal_result.threshold:.4f}")
print(f"Calibration samples: {cal_result.n_samples}")

# Test under shift
shift_result = auditor.add_shift(
    name="Production Q4 2024",
    shift_type="temporal",
    scores=test_scores,
    labels=test_labels,
    metadata={"source": "production_logs", "date_range": "2024-10-01 to 2024-12-31"}
)

# Get verdict
print(f"\n=== AUDIT RESULTS ===")
print(f"Effective Coverage: {shift_result.coverage_metrics.effective_coverage:.1%}")
print(f"Coverage Drop: {shift_result.coverage_metrics.coverage_drop:.1%}")
print(f"FPR: {shift_result.fpr_metrics.fpr:.1%}")
print(f"Verdict: {shift_result.verdict.status}")
print(f"Recommendation: {shift_result.verdict.recommendation}")
```

### Step 3: Interpret the Verdict

| Verdict | What It Means | What To Do |
|---------|--------------|------------|
| **PASS** | Coverage and FPR within tolerance | Deploy with confidence |
| **MARGINAL** | Coverage slightly degraded | Deploy with monitoring; plan recalibration |
| **FAIL** | Guarantees broken | Do NOT deploy; recalibrate on shifted data |

### Step 4: Recalibration Analysis (if needed)

```python
# If verdict is FAIL, check recalibration requirements
recal = auditor.analyze_recalibration(
    scores_target=test_scores,
    labels_target=test_labels,
    scores_source=cal_scores,
    labels_source=cal_labels
)

print(f"\n=== RECALIBRATION ANALYSIS ===")
print(f"New threshold needed: {recal['new_threshold']:.4f}")
print(f"Old threshold: {recal['old_threshold']:.4f}")
print(f"Threshold shift: {recal['threshold_shift']:.4f}")
print(f"Samples needed for recalibration: {recal['recommended_samples']}")
```

---

## The ConformalDrift Protocol

### Phase 1: Baseline Calibration
Establish conformal threshold on source distribution.

### Phase 2: Shift Injection
Apply distribution shift (RLHF, temporal, domain, cross-dataset).

### Phase 3: Degradation Measurement
Evaluate three key metrics:

| Metric | Formula | Why It Matters |
|--------|---------|----------------|
| **Coverage Drop** | `(1-α) - Cov_effective` | Measures if guarantees hold |
| **FPR@NominalCoverage** | `P(flagged \| faithful)` | Catches "flag everything" collapse |
| **Recalibration Interval** | Drift dose before FAIL | When to recalibrate |

### Phase 4: Audit Verdict

| Verdict | Criteria | Action |
|---------|----------|--------|
| **PASS** | \|ΔCov\| ≤ 5%, FPR ≤ 2× baseline | Deploy with confidence |
| **MARGINAL** | \|ΔCov\| ≤ 10% | Deploy with monitoring |
| **FAIL** | Otherwise | Do not deploy |

---

## Advanced Usage

### Simulating Distribution Shifts

Test your detector against known shift patterns:

```python
from conformal_drift import create_rlhf_shift, create_temporal_shift, mix_distributions

# Simulate RLHF-induced shift (scores of faithful responses increase)
shifted = create_rlhf_shift(
    source_scores, source_labels,
    faithful_score_mean=0.45,  # Faithful scores now high
    faithful_score_std=0.08,
)

# Simulate temporal drift (gradual shift over time)
shifted = create_temporal_shift(
    source_scores, source_labels,
    drift_rate=0.1,  # 10% shift per time unit
    time_steps=5,
)

# Mix source and target distributions
mixed_scores, mixed_labels = mix_distributions(
    source_scores, source_labels,
    target_scores, target_labels,
    drift_dose=0.5,  # 50% from each
)
```

### Multi-Shift Audit

Test against multiple shift scenarios:

```python
auditor = ConformalDriftAuditor(nominal_coverage=0.95)
auditor.calibrate(cal_scores, cal_labels)

# Add multiple shift scenarios
auditor.add_shift("RLHF Shift", "rlhf", rlhf_scores, rlhf_labels)
auditor.add_shift("Temporal Shift", "temporal", temporal_scores, temporal_labels)
auditor.add_shift("Cross-Domain", "domain", domain_scores, domain_labels)

# Generate comprehensive report
report = auditor.generate_report()
print(report.summary())

# Access individual results
for shift_name, result in report.shift_results.items():
    print(f"{shift_name}: {result.verdict.status}")
```

### Integration with MLOps Pipelines

```python
from conformal_drift import ConformalDriftAuditor, AuditVerdict

def pre_deployment_check(model_artifacts, test_data) -> bool:
    """Gate deployment on conformal audit."""
    auditor = ConformalDriftAuditor(nominal_coverage=0.95)

    # Load calibration data from artifacts
    cal_scores = np.load(model_artifacts / "cal_scores.npy")
    cal_labels = np.load(model_artifacts / "cal_labels.npy")
    auditor.calibrate(cal_scores, cal_labels)

    # Test on fresh production sample
    test_scores, test_labels = test_data
    result = auditor.add_shift("pre_deploy_check", "domain", test_scores, test_labels)

    # Gate on verdict
    if result.verdict.status == "FAIL":
        raise ValueError(f"Conformal audit failed: {result.verdict.recommendation}")

    return result.verdict.status == "PASS"
```

---

## API Reference

### Core Classes

```python
from conformal_drift import (
    # Main auditor
    ConformalDriftAuditor,
    run_quick_audit,

    # Result containers
    AuditReport,
    CalibrationResult,
    ShiftResult,

    # Metrics
    CoverageMetrics,
    FPRMetrics,
    AuditVerdict,

    # Shift utilities
    ShiftConfig,
    create_rlhf_shift,
    create_temporal_shift,
    mix_distributions,
)
```

### ConformalDriftAuditor

```python
auditor = ConformalDriftAuditor(
    nominal_coverage: float = 0.95,  # Target coverage level
    confidence_level: float = 0.95,  # CI confidence
)

# Methods
auditor.calibrate(scores, labels) -> CalibrationResult
auditor.add_shift(name, shift_type, scores, labels, metadata=None) -> ShiftResult
auditor.analyze_recalibration(scores_target, labels_target, ...) -> Dict
auditor.generate_report() -> AuditReport
```

---

## Data Format

For CLI usage, provide `.npz` files with:
- `scores`: numpy array of nonconformity scores
- `labels`: numpy array of binary labels (1 = hallucinated, 0 = faithful)

```python
np.savez("calibration.npz", scores=cal_scores, labels=cal_labels)
np.savez("test.npz", scores=test_scores, labels=test_labels)
```

---

## Citation

```bibtex
@article{sinha2025conformaldrift,
  title={ConformalDrift: An Audit Protocol for Testing Conformal Guardrails Under Distribution Shift},
  author={Sinha, Debu},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Related Research

This library is part of a research program on **AI reliability under distribution shift**:

| Paper | Focus | Link |
|-------|-------|------|
| **The Semantic Illusion** | Embedding-based detection fails on RLHF | [arXiv:2512.15068](https://arxiv.org/abs/2512.15068) |
| **ATCB** | Agents don't know when they'll fail | [GitHub](https://github.com/debu-sinha/atcb-benchmark) |
| **ConformalDrift** | Conformal guarantees collapse under shift | This repo |
| **DRIFTBENCH** | RAG reliability degrades over time | [HuggingFace](https://huggingface.co/datasets/dsinha/driftbench) |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please see our [Contributing Guide](CONTRIBUTING.md).

**Author:** [Debu Sinha](https://github.com/debu-sinha)
