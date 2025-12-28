"""Tests for conformal_drift metrics module."""

import numpy as np
import pytest

from conformal_drift import (
    compute_coverage_drop,
    compute_fpr_at_nominal,
    compute_audit_verdict,
    CoverageMetrics,
    FPRMetrics,
)


class TestCoverageDrop:
    """Tests for compute_coverage_drop function."""

    def test_perfect_coverage(self):
        """Test when all hallucinations are detected."""
        predictions = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        result = compute_coverage_drop(0.95, predictions, labels)

        assert result.effective_coverage == 1.0
        assert result.coverage_drop == pytest.approx(-0.05, abs=0.01)
        assert result.nominal_coverage == 0.95

    def test_zero_coverage(self):
        """Test when no hallucinations are detected."""
        predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        result = compute_coverage_drop(0.95, predictions, labels)

        assert result.effective_coverage == 0.0
        assert result.coverage_drop == pytest.approx(0.95, abs=0.01)

    def test_partial_coverage(self):
        """Test partial coverage detection."""
        predictions = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        result = compute_coverage_drop(0.95, predictions, labels)

        assert result.effective_coverage == pytest.approx(0.4, abs=0.01)
        assert result.coverage_drop == pytest.approx(0.55, abs=0.01)

    def test_no_hallucinations_raises(self):
        """Test that error is raised when no hallucinations in labels."""
        predictions = np.array([0, 0, 0, 0, 0])
        labels = np.array([0, 0, 0, 0, 0])

        with pytest.raises(ValueError, match="No hallucinated samples"):
            compute_coverage_drop(0.95, predictions, labels)


class TestFPRAtNominal:
    """Tests for compute_fpr_at_nominal function."""

    def test_zero_fpr(self):
        """Test when no faithful responses are incorrectly flagged."""
        predictions = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        result = compute_fpr_at_nominal(predictions, labels)

        assert result.fpr == 0.0

    def test_full_fpr(self):
        """Test when all faithful responses are incorrectly flagged."""
        predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        result = compute_fpr_at_nominal(predictions, labels)

        assert result.fpr == 1.0

    def test_partial_fpr(self):
        """Test partial FPR."""
        predictions = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        result = compute_fpr_at_nominal(predictions, labels)

        assert result.fpr == pytest.approx(0.4, abs=0.01)

    def test_no_faithful_raises(self):
        """Test that error is raised when no faithful samples in labels."""
        predictions = np.array([1, 1, 1, 1, 1])
        labels = np.array([1, 1, 1, 1, 1])

        with pytest.raises(ValueError, match="No faithful samples"):
            compute_fpr_at_nominal(predictions, labels)


class TestAuditVerdict:
    """Tests for compute_audit_verdict function."""

    def test_pass_verdict(self):
        """Test PASS verdict conditions."""
        coverage = CoverageMetrics(
            nominal_coverage=0.95,
            effective_coverage=0.93,
            coverage_drop=0.02,
            coverage_ci=(0.90, 0.96),
        )
        fpr = FPRMetrics(fpr=0.05, fpr_ci=(0.02, 0.08), baseline_fpr=0.03)

        verdict = compute_audit_verdict(coverage, fpr)

        assert verdict.status == "PASS"

    def test_marginal_verdict(self):
        """Test MARGINAL verdict conditions."""
        coverage = CoverageMetrics(
            nominal_coverage=0.95,
            effective_coverage=0.87,
            coverage_drop=0.08,
            coverage_ci=(0.83, 0.91),
        )
        fpr = FPRMetrics(fpr=0.05, fpr_ci=(0.02, 0.08), baseline_fpr=0.03)

        verdict = compute_audit_verdict(coverage, fpr)

        assert verdict.status == "MARGINAL"

    def test_fail_verdict_coverage(self):
        """Test FAIL verdict due to high coverage drop."""
        coverage = CoverageMetrics(
            nominal_coverage=0.95,
            effective_coverage=0.50,
            coverage_drop=0.45,
            coverage_ci=(0.45, 0.55),
        )
        fpr = FPRMetrics(fpr=0.05, fpr_ci=(0.02, 0.08), baseline_fpr=0.03)

        verdict = compute_audit_verdict(coverage, fpr)

        assert verdict.status == "FAIL"

    def test_fail_verdict_fpr_collapse(self):
        """Test FAIL verdict due to FPR collapse."""
        coverage = CoverageMetrics(
            nominal_coverage=0.95,
            effective_coverage=0.94,
            coverage_drop=0.01,
            coverage_ci=(0.91, 0.97),
        )
        fpr = FPRMetrics(fpr=0.95, fpr_ci=(0.92, 0.98), baseline_fpr=0.03)

        verdict = compute_audit_verdict(coverage, fpr)

        assert verdict.status == "FAIL"
        assert "FPR" in verdict.recommendation or "CRITICAL" in verdict.recommendation
