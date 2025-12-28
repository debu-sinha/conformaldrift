"""
Metrics for auditing conformal prediction guardrails under distribution shift.

This module defines three key metrics:
- Coverage Drop (ΔCov): Gap between nominal and effective coverage
- FPR@NominalCoverage: False positive rate when targeting stated coverage
- Recalibration Interval: Shift magnitude before guarantees become unreliable
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class CoverageMetrics:
    """Container for coverage-related metrics."""

    nominal_coverage: float
    effective_coverage: float
    coverage_drop: float
    coverage_ci: Tuple[float, float]

    def __repr__(self) -> str:
        return (
            f"CoverageMetrics(nominal={self.nominal_coverage:.1%}, "
            f"effective={self.effective_coverage:.1%}, "
            f"Δcov={self.coverage_drop:+.1%})"
        )


@dataclass
class FPRMetrics:
    """Container for false positive rate metrics."""

    fpr: float
    fpr_ci: Tuple[float, float]
    baseline_fpr: Optional[float] = None

    def __repr__(self) -> str:
        return f"FPRMetrics(fpr={self.fpr:.1%}, ci=[{self.fpr_ci[0]:.1%}, {self.fpr_ci[1]:.1%}])"


@dataclass
class AuditVerdict:
    """Audit verdict with status and recommendations."""

    status: str  # "PASS", "MARGINAL", or "FAIL"
    coverage_metrics: CoverageMetrics
    fpr_metrics: FPRMetrics
    recommendation: str

    def __repr__(self) -> str:
        return f"AuditVerdict(status={self.status}, Δcov={self.coverage_metrics.coverage_drop:+.1%}, fpr={self.fpr_metrics.fpr:.1%})"


def compute_coverage_drop(
    nominal_coverage: float,
    predictions: np.ndarray,
    labels: np.ndarray,
    confidence: float = 0.95,
) -> CoverageMetrics:
    """
    Compute coverage drop between nominal and effective coverage.

    Coverage Drop (ΔCov) measures the gap between what the conformal method
    claims (nominal coverage) and what it actually achieves (effective coverage).

    Args:
        nominal_coverage: Target coverage level (e.g., 0.95 for 95%)
        predictions: Binary array where 1 = flagged as hallucination
        labels: Binary array where 1 = actually hallucinated
        confidence: Confidence level for bootstrap CI (default: 0.95)

    Returns:
        CoverageMetrics with nominal, effective, drop, and CI
    """
    # Effective coverage = recall on hallucination class
    # P(flagged | hallucinated)
    hallucinated_mask = labels == 1
    if hallucinated_mask.sum() == 0:
        raise ValueError("No hallucinated samples in labels")

    flagged_hallucinations = predictions[hallucinated_mask]
    effective_coverage = flagged_hallucinations.mean()

    # Coverage drop: positive = detecting fewer than promised
    coverage_drop = nominal_coverage - effective_coverage

    # Bootstrap CI for effective coverage
    ci = _bootstrap_ci(flagged_hallucinations, confidence=confidence)

    return CoverageMetrics(
        nominal_coverage=nominal_coverage,
        effective_coverage=effective_coverage,
        coverage_drop=coverage_drop,
        coverage_ci=ci,
    )


def compute_fpr_at_nominal(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidence: float = 0.95,
    baseline_fpr: Optional[float] = None,
) -> FPRMetrics:
    """
    Compute false positive rate when detector is calibrated for nominal coverage.

    FPR@NominalCoverage measures how often faithful responses are incorrectly
    flagged as hallucinations. Under distribution shift, FPR can become
    catastrophically high even when coverage appears maintained.

    Args:
        predictions: Binary array where 1 = flagged as hallucination
        labels: Binary array where 1 = actually hallucinated
        confidence: Confidence level for bootstrap CI (default: 0.95)
        baseline_fpr: Optional baseline FPR for comparison

    Returns:
        FPRMetrics with FPR value, CI, and optional baseline
    """
    # FPR = P(flagged | faithful)
    faithful_mask = labels == 0
    if faithful_mask.sum() == 0:
        raise ValueError("No faithful samples in labels")

    flagged_faithful = predictions[faithful_mask]
    fpr = flagged_faithful.mean()

    # Bootstrap CI
    ci = _bootstrap_ci(flagged_faithful, confidence=confidence)

    return FPRMetrics(fpr=fpr, fpr_ci=ci, baseline_fpr=baseline_fpr)


def compute_audit_verdict(
    coverage_metrics: CoverageMetrics,
    fpr_metrics: FPRMetrics,
    coverage_threshold: float = 0.05,
    marginal_threshold: float = 0.10,
    fpr_multiplier: float = 2.0,
) -> AuditVerdict:
    """
    Determine audit verdict based on coverage drop and FPR.

    Verdict criteria:
    - PASS: |ΔCov| ≤ 0.05 AND FPR ≤ 2 × baseline_FPR
    - MARGINAL: |ΔCov| ≤ 0.10
    - FAIL: Otherwise

    Args:
        coverage_metrics: Coverage metrics from compute_coverage_drop
        fpr_metrics: FPR metrics from compute_fpr_at_nominal
        coverage_threshold: Threshold for PASS verdict (default: 0.05)
        marginal_threshold: Threshold for MARGINAL verdict (default: 0.10)
        fpr_multiplier: Maximum FPR increase multiplier for PASS (default: 2.0)

    Returns:
        AuditVerdict with status and recommendation
    """
    delta_cov = abs(coverage_metrics.coverage_drop)
    fpr = fpr_metrics.fpr
    baseline_fpr = fpr_metrics.baseline_fpr or 0.0

    # Check FPR condition
    fpr_ok = fpr <= max(fpr_multiplier * baseline_fpr, 0.10)  # At least 10% threshold

    # FPR collapse is always a FAIL, regardless of coverage
    if fpr > 0.5:
        status = "FAIL"
        recommendation = (
            "CRITICAL: FPR collapse detected. Detector is flagging "
            f"{fpr:.0%} of faithful responses. Do not deploy."
        )
    elif delta_cov <= coverage_threshold and fpr_ok:
        status = "PASS"
        recommendation = "Conformal guarantees are reliable under this shift."
    elif delta_cov <= marginal_threshold:
        status = "MARGINAL"
        recommendation = "Consider recalibration. Monitor closely in production."
    else:
        status = "FAIL"
        recommendation = (
            f"Coverage dropped by {delta_cov:.0%}. "
            "Recalibration required before deployment."
        )

    return AuditVerdict(
        status=status,
        coverage_metrics=coverage_metrics,
        fpr_metrics=fpr_metrics,
        recommendation=recommendation,
    )


def compute_recalibration_interval(
    scores_source: np.ndarray,
    scores_target: np.ndarray,
    labels_source: np.ndarray,
    labels_target: np.ndarray,
    nominal_coverage: float = 0.95,
    drift_doses: Optional[np.ndarray] = None,
) -> dict:
    """
    Determine at what drift magnitude recalibration becomes necessary.

    This metric helps practitioners understand when their conformal guarantees
    will break and recalibration is needed.

    Args:
        scores_source: Nonconformity scores from source distribution
        scores_target: Nonconformity scores from target distribution
        labels_source: Labels for source (1 = hallucinated)
        labels_target: Labels for target (1 = hallucinated)
        nominal_coverage: Target coverage level (default: 0.95)
        drift_doses: Array of drift percentages to test (default: [0, 0.1, 0.25, 0.5, 1.0])

    Returns:
        Dictionary with drift dose analysis and recommended interval
    """
    if drift_doses is None:
        drift_doses = np.array([0.0, 0.1, 0.25, 0.5, 1.0])

    # Calibrate threshold on source distribution
    hall_scores_source = scores_source[labels_source == 1]
    threshold = np.quantile(hall_scores_source, 1 - nominal_coverage)

    results = []
    for dose in drift_doses:
        # Mix source and target
        n_target = int(len(scores_target) * dose)
        n_source = len(scores_target) - n_target

        if n_source > 0:
            source_idx = np.random.choice(len(scores_source), n_source, replace=True)
            mixed_scores = np.concatenate([scores_source[source_idx], scores_target[:n_target]])
            mixed_labels = np.concatenate([labels_source[source_idx], labels_target[:n_target]])
        else:
            mixed_scores = scores_target
            mixed_labels = labels_target

        # Compute predictions at calibrated threshold
        predictions = (mixed_scores >= threshold).astype(int)

        # Compute metrics
        hall_mask = mixed_labels == 1
        faith_mask = mixed_labels == 0

        if hall_mask.sum() > 0:
            cov_eff = predictions[hall_mask].mean()
        else:
            cov_eff = 0.0

        if faith_mask.sum() > 0:
            fpr = predictions[faith_mask].mean()
        else:
            fpr = 0.0

        delta_cov = nominal_coverage - cov_eff

        # Determine status
        if abs(delta_cov) <= 0.05 and fpr <= 0.10:
            status = "PASS"
        elif abs(delta_cov) <= 0.10:
            status = "MARGINAL"
        else:
            status = "FAIL"

        results.append({
            "drift_dose": dose,
            "cov_eff": cov_eff,
            "delta_cov": delta_cov,
            "fpr": fpr,
            "status": status,
        })

    # Find recommended recalibration interval
    for i, r in enumerate(results):
        if r["status"] == "FAIL":
            if i > 0:
                interval = results[i - 1]["drift_dose"]
            else:
                interval = 0.0
            break
    else:
        interval = 1.0  # No failure found

    return {
        "results": results,
        "recalibration_interval": interval,
        "recommendation": f"Recalibrate when drift exceeds {interval:.0%}",
    }


def _bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(42)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_means.append(sample.mean())

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return (lower, upper)
