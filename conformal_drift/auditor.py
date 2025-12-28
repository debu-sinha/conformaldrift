"""
ConformalDrift Auditor - Main audit protocol implementation.

The ConformalDrift protocol operates in four phases:
1. Baseline Calibration: Establish threshold on source distribution
2. Shift Injection: Apply distribution shift to test data
3. Degradation Measurement: Evaluate metrics under shift
4. Audit Report: Generate verdict and recommendations
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .metrics import (
    AuditVerdict,
    CoverageMetrics,
    FPRMetrics,
    compute_audit_verdict,
    compute_coverage_drop,
    compute_fpr_at_nominal,
    compute_recalibration_interval,
)


@dataclass
class CalibrationResult:
    """Results from Phase 1: Baseline Calibration."""

    threshold: float
    nominal_coverage: float
    calibration_coverage: CoverageMetrics
    calibration_fpr: FPRMetrics
    n_samples: int
    score_stats: Dict[str, float]


@dataclass
class ShiftResult:
    """Results from testing under a specific shift."""

    shift_name: str
    shift_type: str
    coverage_metrics: CoverageMetrics
    fpr_metrics: FPRMetrics
    verdict: AuditVerdict
    n_samples: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Complete audit report from ConformalDrift protocol."""

    timestamp: str
    protocol_version: str
    calibration: CalibrationResult
    shift_results: List[ShiftResult]
    recalibration_analysis: Optional[Dict[str, Any]]
    overall_verdict: str
    critical_findings: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "protocol_version": self.protocol_version,
            "calibration": {
                "threshold": self.calibration.threshold,
                "nominal_coverage": self.calibration.nominal_coverage,
                "n_samples": self.calibration.n_samples,
                "coverage": self.calibration.calibration_coverage.effective_coverage,
                "fpr": self.calibration.calibration_fpr.fpr,
            },
            "shift_results": [
                {
                    "shift_name": r.shift_name,
                    "shift_type": r.shift_type,
                    "coverage_eff": r.coverage_metrics.effective_coverage,
                    "coverage_drop": r.coverage_metrics.coverage_drop,
                    "fpr": r.fpr_metrics.fpr,
                    "verdict": r.verdict.status,
                }
                for r in self.shift_results
            ],
            "recalibration_analysis": self.recalibration_analysis,
            "overall_verdict": self.overall_verdict,
            "critical_findings": self.critical_findings,
            "recommendations": self.recommendations,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "CONFORMALDRIFT AUDIT REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Protocol Version: {self.protocol_version}",
            "",
            "CALIBRATION",
            "-" * 40,
            f"  Threshold: {self.calibration.threshold:.4f}",
            f"  Nominal Coverage: {self.calibration.nominal_coverage:.1%}",
            f"  Calibration FPR: {self.calibration.calibration_fpr.fpr:.1%}",
            "",
            "SHIFT RESULTS",
            "-" * 40,
        ]

        for r in self.shift_results:
            lines.extend([
                f"  [{r.verdict.status}] {r.shift_name}",
                f"    Coverage: {r.coverage_metrics.effective_coverage:.1%} "
                f"(Delta={r.coverage_metrics.coverage_drop:+.1%})",
                f"    FPR: {r.fpr_metrics.fpr:.1%}",
            ])

        lines.extend([
            "",
            "OVERALL VERDICT",
            "-" * 40,
            f"  {self.overall_verdict}",
            "",
            "CRITICAL FINDINGS",
            "-" * 40,
        ])
        for finding in self.critical_findings:
            lines.append(f"  - {finding}")

        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 40,
        ])
        for rec in self.recommendations:
            lines.append(f"  - {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)


class ConformalDriftAuditor:
    """
    Main auditor class implementing the ConformalDrift protocol.

    Example usage:
        >>> auditor = ConformalDriftAuditor(nominal_coverage=0.95)
        >>> auditor.calibrate(cal_scores, cal_labels)
        >>> auditor.add_shift("RLHF", "rlhf", test_scores, test_labels)
        >>> report = auditor.generate_report()
        >>> print(report.summary())
    """

    PROTOCOL_VERSION = "1.0"

    def __init__(
        self,
        nominal_coverage: float = 0.95,
        score_fn: Optional[Callable] = None,
    ):
        """
        Initialize the auditor.

        Args:
            nominal_coverage: Target coverage level (default: 0.95 for 95%)
            score_fn: Optional function to compute nonconformity scores
                      Signature: score_fn(inputs, outputs) -> np.ndarray
        """
        self.nominal_coverage = nominal_coverage
        self.score_fn = score_fn
        self.calibration: Optional[CalibrationResult] = None
        self.shift_results: List[ShiftResult] = []
        self.recalibration_analysis: Optional[Dict[str, Any]] = None

    def calibrate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> CalibrationResult:
        """
        Phase 1: Baseline Calibration.

        Calibrate the conformal threshold on source distribution.

        Args:
            scores: Nonconformity scores for calibration set
            labels: Binary labels (1 = hallucinated, 0 = faithful)

        Returns:
            CalibrationResult with threshold and baseline metrics
        """
        scores = np.asarray(scores)
        labels = np.asarray(labels)

        # Compute threshold on hallucinated samples
        hall_scores = scores[labels == 1]
        faith_scores = scores[labels == 0]

        if len(hall_scores) == 0:
            raise ValueError("No hallucinated samples in calibration set")
        if len(faith_scores) == 0:
            raise ValueError("No faithful samples in calibration set")

        # Threshold: quantile that achieves nominal coverage
        threshold = np.quantile(hall_scores, 1 - self.nominal_coverage)

        # Generate predictions at this threshold
        predictions = (scores >= threshold).astype(int)

        # Compute baseline metrics
        coverage_metrics = compute_coverage_drop(
            self.nominal_coverage, predictions, labels
        )
        fpr_metrics = compute_fpr_at_nominal(predictions, labels)

        # Score statistics
        score_stats = {
            "hallucinated_mean": float(hall_scores.mean()),
            "hallucinated_std": float(hall_scores.std()),
            "faithful_mean": float(faith_scores.mean()),
            "faithful_std": float(faith_scores.std()),
            "threshold": float(threshold),
        }

        self.calibration = CalibrationResult(
            threshold=threshold,
            nominal_coverage=self.nominal_coverage,
            calibration_coverage=coverage_metrics,
            calibration_fpr=fpr_metrics,
            n_samples=len(scores),
            score_stats=score_stats,
        )

        return self.calibration

    def add_shift(
        self,
        shift_name: str,
        shift_type: str,
        scores: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShiftResult:
        """
        Phase 2 & 3: Shift Injection and Degradation Measurement.

        Test conformal guarantees under a specific distribution shift.

        Args:
            shift_name: Human-readable name for this shift (e.g., "NQ -> HaluEval")
            shift_type: Type of shift ("rlhf", "temporal", "domain", "cross_dataset")
            scores: Nonconformity scores for shifted test set
            labels: Binary labels (1 = hallucinated, 0 = faithful)
            metadata: Optional additional metadata about the shift

        Returns:
            ShiftResult with metrics and verdict for this shift
        """
        if self.calibration is None:
            raise RuntimeError("Must call calibrate() before add_shift()")

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        # Generate predictions using calibrated threshold
        predictions = (scores >= self.calibration.threshold).astype(int)

        # Compute metrics
        coverage_metrics = compute_coverage_drop(
            self.nominal_coverage, predictions, labels
        )
        fpr_metrics = compute_fpr_at_nominal(
            predictions, labels,
            baseline_fpr=self.calibration.calibration_fpr.fpr
        )

        # Generate verdict
        verdict = compute_audit_verdict(coverage_metrics, fpr_metrics)

        result = ShiftResult(
            shift_name=shift_name,
            shift_type=shift_type,
            coverage_metrics=coverage_metrics,
            fpr_metrics=fpr_metrics,
            verdict=verdict,
            n_samples=len(scores),
            metadata=metadata or {},
        )

        self.shift_results.append(result)
        return result

    def analyze_recalibration(
        self,
        scores_target: np.ndarray,
        labels_target: np.ndarray,
        scores_source: Optional[np.ndarray] = None,
        labels_source: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Analyze recalibration interval for drift dose sensitivity.

        Args:
            scores_target: Scores from target (shifted) distribution
            labels_target: Labels for target distribution
            scores_source: Scores from source distribution (uses calibration if None)
            labels_source: Labels for source distribution

        Returns:
            Dictionary with recalibration analysis results
        """
        if self.calibration is None:
            raise RuntimeError("Must call calibrate() before analyze_recalibration()")

        # Use calibration data if source not provided
        if scores_source is None:
            raise ValueError("scores_source required for recalibration analysis")

        self.recalibration_analysis = compute_recalibration_interval(
            scores_source=np.asarray(scores_source),
            scores_target=np.asarray(scores_target),
            labels_source=np.asarray(labels_source),
            labels_target=np.asarray(labels_target),
            nominal_coverage=self.nominal_coverage,
        )

        return self.recalibration_analysis

    def generate_report(self) -> AuditReport:
        """
        Phase 4: Generate Audit Report.

        Compile all results into a comprehensive audit report.

        Returns:
            AuditReport with all findings and recommendations
        """
        if self.calibration is None:
            raise RuntimeError("Must call calibrate() before generate_report()")

        # Determine overall verdict
        verdicts = [r.verdict.status for r in self.shift_results]
        if "FAIL" in verdicts:
            overall_verdict = "FAIL - Do not deploy without recalibration"
        elif "MARGINAL" in verdicts:
            overall_verdict = "MARGINAL - Deploy with caution and monitoring"
        elif verdicts:
            overall_verdict = "PASS - Conformal guarantees reliable"
        else:
            overall_verdict = "INCOMPLETE - No shift tests performed"

        # Extract critical findings
        critical_findings = []
        for r in self.shift_results:
            if r.fpr_metrics.fpr >= 0.5:
                critical_findings.append(
                    f"FPR COLLAPSE on {r.shift_name}: {r.fpr_metrics.fpr:.0%} of "
                    "faithful responses incorrectly flagged"
                )
            if r.coverage_metrics.coverage_drop > 0.5:
                critical_findings.append(
                    f"COVERAGE COLLAPSE on {r.shift_name}: "
                    f"{r.coverage_metrics.coverage_drop:.0%} coverage drop"
                )

        if not critical_findings:
            critical_findings.append("No critical issues detected")

        # Generate recommendations
        recommendations = [
            "Never trust coverage alone - always measure FPR alongside coverage",
            "Run ConformalDrift audit before any production deployment",
        ]

        if self.recalibration_analysis:
            interval = self.recalibration_analysis.get("recalibration_interval", 0.1)
            recommendations.append(
                f"Recalibrate when estimated drift exceeds {interval:.0%}"
            )

        for r in self.shift_results:
            if r.verdict.status == "FAIL":
                recommendations.append(
                    f"Address {r.shift_name} shift before deployment"
                )

        return AuditReport(
            timestamp=datetime.now().isoformat(),
            protocol_version=self.PROTOCOL_VERSION,
            calibration=self.calibration,
            shift_results=self.shift_results,
            recalibration_analysis=self.recalibration_analysis,
            overall_verdict=overall_verdict,
            critical_findings=critical_findings,
            recommendations=recommendations,
        )


def run_quick_audit(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    shift_name: str = "Distribution Shift",
    nominal_coverage: float = 0.95,
) -> AuditReport:
    """
    Convenience function for quick single-shift audit.

    Args:
        cal_scores: Calibration set nonconformity scores
        cal_labels: Calibration set labels (1 = hallucinated)
        test_scores: Test set nonconformity scores
        test_labels: Test set labels
        shift_name: Name for the shift being tested
        nominal_coverage: Target coverage level

    Returns:
        AuditReport with results

    Example:
        >>> report = run_quick_audit(
        ...     cal_scores, cal_labels,
        ...     test_scores, test_labels,
        ...     shift_name="NQ -> HaluEval"
        ... )
        >>> print(report.overall_verdict)
    """
    auditor = ConformalDriftAuditor(nominal_coverage=nominal_coverage)
    auditor.calibrate(cal_scores, cal_labels)
    auditor.add_shift(shift_name, "unknown", test_scores, test_labels)
    return auditor.generate_report()
