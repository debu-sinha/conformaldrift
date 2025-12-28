# ConformalDrift

**An Audit Protocol for Testing Conformal Guardrails Under Distribution Shift**

[![arXiv](https://img.shields.io/badge/arXiv-2512.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2512.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Conformal prediction methods promise distribution-free coverage guarantees for RAG systems. However, these guarantees assume exchangeability between calibration and deployment data—an assumption that systematically fails in practice.

**ConformalDrift** is an audit protocol for stress-testing conformal guardrails under realistic distribution shifts.

## Key Findings

| Shift Type | Coverage | FPR | Finding |
|------------|----------|-----|---------|
| In-distribution (NQ→NQ) | 95.8% | 0% | Baseline |
| Temporal (FastAPI v1→v2) | 95.2% | 76% | "Coverage through conservatism" |
| Cross-dataset (NQ→RAGTruth) | 11.0% | - | 84% coverage collapse |

## Installation

```bash
pip install -e .
```

Or install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Run temporal shift audit

```bash
python experiments/run_temporal_shift.py --device cpu
```

### Run cross-dataset shift audit

```bash
python experiments/run_cross_dataset_shift.py
```

### Run full audit protocol

```bash
conformal-drift audit --scenario temporal
conformal-drift audit --scenario cross-dataset
```

## Repository Structure

```
conformaldrift/
├── src/
│   ├── auditor.py              # Main audit protocol
│   ├── metrics.py              # Coverage Drop, FPR, Recalibration metrics
│   ├── shift_injection.py      # Temporal and cross-dataset shift injection
│   └── detectors/
│       └── crg_detector.py     # CRG conformal detector
├── experiments/
│   ├── run_temporal_shift.py   # Temporal shift experiments
│   └── run_cross_dataset_shift.py
├── results/                    # Experiment outputs (JSON)
├── tests/
├── requirements.txt
├── setup.py
└── LICENSE
```

## Metrics

ConformalDrift introduces three audit metrics:

1. **Coverage Drop (ΔCov)**: Gap between nominal and effective coverage
2. **FPR@NominalCoverage**: False positive rate at target coverage
3. **Recalibration Interval**: Drift magnitude requiring recalibration

## Audit Verdicts

| Verdict | Criteria |
|---------|----------|
| **PASS** | \|ΔCov\| ≤ 5% and FPR ≤ 2× baseline |
| **MARGINAL** | 5% < \|ΔCov\| ≤ 10% |
| **FAIL** | \|ΔCov\| > 10% or FPR > 50% |

## Citation

```bibtex
@article{sinha2025conformaldrift,
  title={ConformalDrift: An Audit Protocol for Testing Conformal
         Guardrails Under Distribution Shift},
  author={Sinha, Debu},
  journal={arXiv preprint arXiv:2512.XXXXX},
  year={2025}
}
```

## Related Work

- [The Semantic Illusion](https://arxiv.org/abs/2512.15068) - Detection limits under RLHF shift

## License

MIT License
