# Conformal-Drift

[![PyPI version](https://badge.fury.io/py/conformal-drift.svg)](https://pypi.org/project/conformal-drift/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Audit protocol for stress-testing conformal prediction guardrails under distribution shift.**

Conformal prediction provides calibrated uncertainty estimates with coverage guarantees—but these guarantees assume the test distribution matches calibration. **Conformal-Drift** systematically tests what happens when that assumption breaks.

## Why Conformal-Drift?

Conformal prediction guardrails are increasingly used in production ML systems, especially RAG pipelines, to detect hallucinations and abstain from unreliable predictions. But real-world deployments face:

- **Temporal drift**: Language, topics, and user behavior change over time
- **Domain shift**: Models deployed in new contexts they weren't calibrated for
- **Adversarial inputs**: Intentionally crafted inputs to bypass guardrails

Conformal-Drift provides a structured audit protocol to quantify how guardrail performance degrades under these conditions.

## Quick Start

```bash
pip install conformal-drift
```

```python
from conformal_drift import ConformalDriftAuditor

# Initialize auditor with your conformal predictor
auditor = ConformalDriftAuditor(
    predictor=your_conformal_predictor,
    calibration_scores=calibration_nonconformity_scores
)

# Run audit with temporal shift
results = auditor.audit(
    test_data=test_samples,
    shift_type="temporal",
    shift_intensity=[0.0, 0.25, 0.5, 0.75, 1.0]
)

# Analyze coverage degradation
print(f"Coverage at 0% shift: {results.coverage[0]:.3f}")
print(f"Coverage at 100% shift: {results.coverage[-1]:.3f}")
print(f"Coverage gap: {results.coverage_gap:.3f}")
```

## Key Features

- **Multiple shift operators**: Temporal, semantic, lexical, and adversarial shifts
- **Coverage tracking**: Monitor how empirical coverage deviates from nominal level
- **Set size analysis**: Track prediction set sizes under shift
- **Visualization tools**: Generate coverage curves and reliability diagrams
- **Extensible API**: Easy to add custom shift operators

## Documentation

- [Getting Started](getting-started.md) - Installation and first audit
- [Core Concepts](guide/concepts.md) - Understanding conformal prediction and distribution shift
- [Shift Operators](guide/shift-operators.md) - Available shift types and customization
- [API Reference](api.md) - Full API documentation

## Citation

If you use Conformal-Drift in your research, please cite:

```bibtex
@article{sinha2024conformaldrift,
  title={Conformal-Drift: Stress-Testing Conformal Prediction Under Distribution Shift},
  author={Sinha, Debu},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - see [LICENSE](https://github.com/debu-sinha/conformaldrift/blob/main/LICENSE) for details.
