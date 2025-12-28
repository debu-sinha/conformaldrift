"""
ConformalDrift: Audit protocol for conformal prediction guardrails under distribution shift.

This package provides tools for stress-testing conformal prediction methods
before deployment, ensuring that coverage guarantees remain valid under
realistic distribution shifts.

Where This Fits in Your AI Pipeline:
    Training → Calibration → **AUDIT WITH CONFORMAL-DRIFT** → Deployment → Monitoring

Key Features:
    - Standardized audit protocol with PASS/MARGINAL/FAIL verdicts
    - Three degradation metrics: Coverage Drop, FPR@NominalCoverage, Recalibration Interval
    - Shift simulation utilities for RLHF, temporal, and domain shifts
    - CLI for quick audits: `conformal-drift demo`

Quick Start:
    >>> from conformal_drift import ConformalDriftAuditor
    >>> auditor = ConformalDriftAuditor(nominal_coverage=0.95)
    >>> auditor.calibrate(cal_scores, cal_labels)
    >>> auditor.add_shift("RLHF Shift", "rlhf", test_scores, test_labels)
    >>> report = auditor.generate_report()
    >>> print(report.summary())

When to Use This Library:
    - Before deploying any conformal prediction system
    - When your data distribution may differ from calibration data
    - To validate coverage guarantees hold under RLHF, temporal, or domain shifts
    - As part of your ML ops pipeline for continuous validation

For more details, see: https://github.com/debu-sinha/conformaldrift
"""

__version__ = "0.1.0"
__author__ = "Debu Sinha"
__email__ = "debusinha2009@gmail.com"

from .auditor import (
    AuditReport,
    CalibrationResult,
    ConformalDriftAuditor,
    ShiftResult,
    run_quick_audit,
)
from .metrics import (
    AuditVerdict,
    CoverageMetrics,
    FPRMetrics,
    compute_audit_verdict,
    compute_coverage_drop,
    compute_fpr_at_nominal,
    compute_recalibration_interval,
)
from .shifts import (
    ShiftConfig,
    ShiftedData,
    create_rlhf_shift,
    create_temporal_shift,
    inject_overlap_shift,
    inject_score_shift,
    mix_distributions,
    SHIFT_PRESETS,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Main auditor
    "ConformalDriftAuditor",
    "run_quick_audit",
    # Result containers
    "AuditReport",
    "CalibrationResult",
    "ShiftResult",
    # Metrics
    "CoverageMetrics",
    "FPRMetrics",
    "AuditVerdict",
    "compute_coverage_drop",
    "compute_fpr_at_nominal",
    "compute_audit_verdict",
    "compute_recalibration_interval",
    # Shift utilities
    "ShiftConfig",
    "ShiftedData",
    "inject_score_shift",
    "inject_overlap_shift",
    "mix_distributions",
    "create_rlhf_shift",
    "create_temporal_shift",
    "SHIFT_PRESETS",
]
