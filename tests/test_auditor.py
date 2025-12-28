"""Tests for conformal_drift auditor module."""

import numpy as np
import pytest

from conformal_drift import ConformalDriftAuditor, run_quick_audit


class TestConformalDriftAuditor:
    """Tests for ConformalDriftAuditor class."""

    @pytest.fixture
    def calibration_data(self):
        """Generate synthetic calibration data with clear separation."""
        np.random.seed(42)
        n = 500

        hall_scores = np.random.normal(0.5, 0.05, n // 2)
        faith_scores = np.random.normal(0.1, 0.03, n // 2)
        scores = np.concatenate([hall_scores, faith_scores])
        labels = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(int)

        return scores, labels

    @pytest.fixture
    def shifted_data_rlhf(self):
        """Generate shifted data simulating RLHF collapse."""
        np.random.seed(43)
        n = 200

        # RLHF shift: faithful scores become similar to hallucinations
        hall_scores = np.random.normal(0.48, 0.08, n // 2)
        faith_scores = np.random.normal(0.45, 0.07, n // 2)
        scores = np.concatenate([hall_scores, faith_scores])
        labels = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(int)

        return scores, labels

    @pytest.fixture
    def shifted_data_temporal(self):
        """Generate shifted data simulating temporal drift."""
        np.random.seed(44)
        n = 200

        # Temporal shift: hallucination scores decay
        hall_scores = np.random.normal(0.35, 0.08, n // 2)
        faith_scores = np.random.normal(0.1, 0.03, n // 2)
        scores = np.concatenate([hall_scores, faith_scores])
        labels = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(int)

        return scores, labels

    def test_calibration(self, calibration_data):
        """Test calibration phase."""
        cal_scores, cal_labels = calibration_data

        auditor = ConformalDriftAuditor(nominal_coverage=0.95)
        result = auditor.calibrate(cal_scores, cal_labels)

        # Check threshold is reasonable
        assert 0.3 < result.threshold < 0.6
        assert result.n_samples == len(cal_scores)
        assert result.calibration_fpr.fpr < 0.05  # Low FPR on calibration

    def test_calibration_required_before_shift(self):
        """Test that calibrate() must be called before add_shift()."""
        auditor = ConformalDriftAuditor()

        with pytest.raises(RuntimeError, match="Must call calibrate"):
            auditor.add_shift("Test", "test", np.array([0.5]), np.array([1]))

    def test_rlhf_shift_detection(self, calibration_data, shifted_data_rlhf):
        """Test detection of RLHF-induced FPR collapse."""
        cal_scores, cal_labels = calibration_data
        test_scores, test_labels = shifted_data_rlhf

        auditor = ConformalDriftAuditor(nominal_coverage=0.95)
        auditor.calibrate(cal_scores, cal_labels)
        result = auditor.add_shift("RLHF Shift", "rlhf", test_scores, test_labels)

        # RLHF shift should cause high FPR
        assert result.fpr_metrics.fpr > 0.5
        assert result.verdict.status == "FAIL"

    def test_temporal_shift_detection(self, calibration_data, shifted_data_temporal):
        """Test detection of temporal coverage drop."""
        cal_scores, cal_labels = calibration_data
        test_scores, test_labels = shifted_data_temporal

        auditor = ConformalDriftAuditor(nominal_coverage=0.95)
        auditor.calibrate(cal_scores, cal_labels)
        result = auditor.add_shift("Temporal Shift", "temporal", test_scores, test_labels)

        # Temporal shift should cause coverage drop
        assert result.coverage_metrics.coverage_drop > 0.1

    def test_generate_report(self, calibration_data, shifted_data_rlhf):
        """Test report generation."""
        cal_scores, cal_labels = calibration_data
        test_scores, test_labels = shifted_data_rlhf

        auditor = ConformalDriftAuditor(nominal_coverage=0.95)
        auditor.calibrate(cal_scores, cal_labels)
        auditor.add_shift("RLHF Shift", "rlhf", test_scores, test_labels)
        report = auditor.generate_report()

        assert report.protocol_version == "1.0"
        assert len(report.shift_results) == 1
        assert "FAIL" in report.overall_verdict
        assert len(report.critical_findings) > 0
        assert len(report.recommendations) > 0

    def test_report_to_dict(self, calibration_data, shifted_data_rlhf):
        """Test report serialization."""
        cal_scores, cal_labels = calibration_data
        test_scores, test_labels = shifted_data_rlhf

        auditor = ConformalDriftAuditor(nominal_coverage=0.95)
        auditor.calibrate(cal_scores, cal_labels)
        auditor.add_shift("RLHF Shift", "rlhf", test_scores, test_labels)
        report = auditor.generate_report()

        report_dict = report.to_dict()

        assert "timestamp" in report_dict
        assert "calibration" in report_dict
        assert "shift_results" in report_dict
        assert "overall_verdict" in report_dict

    def test_report_summary(self, calibration_data, shifted_data_rlhf):
        """Test report summary generation."""
        cal_scores, cal_labels = calibration_data
        test_scores, test_labels = shifted_data_rlhf

        auditor = ConformalDriftAuditor(nominal_coverage=0.95)
        auditor.calibrate(cal_scores, cal_labels)
        auditor.add_shift("RLHF Shift", "rlhf", test_scores, test_labels)
        report = auditor.generate_report()

        summary = report.summary()

        assert "CONFORMALDRIFT AUDIT REPORT" in summary
        assert "CALIBRATION" in summary
        assert "SHIFT RESULTS" in summary
        assert "OVERALL VERDICT" in summary


class TestQuickAudit:
    """Tests for run_quick_audit convenience function."""

    def test_quick_audit(self):
        """Test quick audit function."""
        np.random.seed(42)

        # Calibration
        cal_scores = np.concatenate([
            np.random.normal(0.5, 0.05, 100),
            np.random.normal(0.1, 0.03, 100),
        ])
        cal_labels = np.concatenate([np.ones(100), np.zeros(100)]).astype(int)

        # Test (with shift)
        test_scores = np.concatenate([
            np.random.normal(0.35, 0.08, 50),
            np.random.normal(0.1, 0.03, 50),
        ])
        test_labels = np.concatenate([np.ones(50), np.zeros(50)]).astype(int)

        report = run_quick_audit(
            cal_scores, cal_labels,
            test_scores, test_labels,
            shift_name="Test Shift",
        )

        assert report is not None
        assert len(report.shift_results) == 1
        assert report.shift_results[0].shift_name == "Test Shift"
