# ConformalDrift

**An Audit Protocol for Testing Conformal Guardrails Under Distribution Shift**

[![arXiv](https://img.shields.io/badge/arXiv-2512.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2512.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Conformal prediction methods promise distribution-free coverage guarantees for RAG systems. However, these guarantees assume exchangeability between calibration and deployment data—an assumption that systematically fails in practice.

**ConformalDrift** is an audit protocol for stress-testing conformal guardrails under realistic distribution shifts.

## Key Findings

| Guardrail | Shift Type | Coverage | FPR | Verdict |
|-----------|------------|----------|-----|---------|
| CRG | In-distribution | 93.0% | 93.0% | BLOCK (FPR explosion) |
| CRG | Cross-dataset | 98.0% | 0.0% | DEPLOY |
| CRG | RLHF | 5.3% | 12.3% | BLOCK (Coverage collapse) |
| Rules-Check | In-distribution | 100.0% | 100.0% | BLOCK |
| Rules-Check | Cross-dataset | 100.0% | 0.0% | MONITOR |
| Rules-Check | RLHF | 100.0% | 100.0% | BLOCK |

**Two Failure Modes Identified:**
- **Mode C (Coverage Collapse):** CRG under RLHF shift drops from 95% to 5.3% coverage
- **Mode F (FPR Explosion):** CRG in-distribution achieves 93% coverage with 93% FPR

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

### Run Rules-Check guardrail experiment

```bash
python experiments/run_rules_check.py \
    --nq-cal data/nq_calibration.json \
    --nq-test data/nq_test.json \
    --ragtruth data/ragtruth_test.json
```

### Run recalibration budget experiment

```bash
python experiments/run_recalibration.py \
    --nq-cal data/nq_calibration.json \
    --ragtruth data/ragtruth_test.json \
    --k-values "0,10,25,50,100"
```

### Run CRG audit experiment

```bash
python experiments/run_crg_audit.py \
    --nq-cal data/nq_calibration.json \
    --nq-test data/nq_test.json \
    --ragtruth data/ragtruth_test.json \
    --halueval data/halueval_test.json
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
│   ├── __init__.py
│   └── detectors/
│       ├── __init__.py
│       └── crg_detector.py     # CRG conformal detector
├── experiments/
│   ├── run_temporal_shift.py   # Temporal shift experiments
│   ├── run_rules_check.py      # Rules-Check guardrail
│   ├── run_recalibration.py    # Recalibration budget curve
│   └── run_crg_audit.py        # CRG audit matrix
├── scripts/
│   └── download_real_data.py   # Download datasets from HuggingFace
├── data/                       # Evaluation datasets
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

## Audit Verdicts (2-Axis System)

**Axis 1: Guarantee Validity** (based on |ΔCov|)
| Status | Criteria |
|--------|----------|
| VALID | \|ΔCov\| ≤ 2% |
| DEGRADED | 2% < \|ΔCov\| ≤ 10% |
| INVALID | \|ΔCov\| > 10% |

**Axis 2: Operational Usability** (based on FPR)
| Status | Criteria |
|--------|----------|
| USABLE | FPR ≤ 20% |
| DEGRADED | 20% < FPR ≤ 50% |
| UNUSABLE | FPR > 50% |

**Final Verdict**
| Verdict | Criteria |
|---------|----------|
| **DEPLOY** | VALID + USABLE |
| **MONITOR** | DEGRADED on either axis |
| **BLOCK** | INVALID or UNUSABLE |

**Failure Modes**
| Mode | Description |
|------|-------------|
| **C** | Coverage Collapse - effective coverage drops below target |
| **F** | FPR Explosion - FPR becomes unacceptably high |
| **S** | Stable - both metrics within acceptable range |

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
