"""
Command-line interface for ConformalDrift.

Usage:
    conformal-drift audit --cal-data CAL.npz --test-data TEST.npz
    conformal-drift demo
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .auditor import ConformalDriftAuditor, run_quick_audit


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="conformal-drift",
        description="Audit protocol for conformal prediction guardrails under distribution shift",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Run audit on data files")
    audit_parser.add_argument(
        "--cal-data",
        type=str,
        required=True,
        help="Path to calibration data (.npz with 'scores' and 'labels')",
    )
    audit_parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data (.npz with 'scores' and 'labels')",
    )
    audit_parser.add_argument(
        "--shift-name",
        type=str,
        default="Distribution Shift",
        help="Name for the shift being tested",
    )
    audit_parser.add_argument(
        "--coverage",
        type=float,
        default=0.95,
        help="Nominal coverage level (default: 0.95)",
    )
    audit_parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON report",
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with synthetic data")
    demo_parser.add_argument(
        "--scenario",
        type=str,
        choices=["rlhf", "temporal", "both"],
        default="both",
        help="Which shift scenario to demo",
    )

    args = parser.parse_args()

    if args.command == "audit":
        run_audit_command(args)
    elif args.command == "demo":
        run_demo_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_audit_command(args):
    """Run audit on provided data files."""
    # Load data
    cal_data = np.load(args.cal_data)
    test_data = np.load(args.test_data)

    cal_scores = cal_data["scores"]
    cal_labels = cal_data["labels"]
    test_scores = test_data["scores"]
    test_labels = test_data["labels"]

    # Run audit
    report = run_quick_audit(
        cal_scores=cal_scores,
        cal_labels=cal_labels,
        test_scores=test_scores,
        test_labels=test_labels,
        shift_name=args.shift_name,
        nominal_coverage=args.coverage,
    )

    # Print summary
    print(report.summary())

    # Save JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {output_path}")


def run_demo_command(args):
    """Run demo with synthetic data."""
    print("=" * 60)
    print("CONFORMALDRIFT DEMO")
    print("=" * 60)
    print()

    # Generate synthetic calibration data
    np.random.seed(42)
    n_cal = 500

    # Calibration: clear separation between hallucinations and faithful
    cal_hall_scores = np.random.normal(0.5, 0.05, n_cal // 2)
    cal_faith_scores = np.random.normal(0.1, 0.03, n_cal // 2)
    cal_scores = np.concatenate([cal_hall_scores, cal_faith_scores])
    cal_labels = np.concatenate([np.ones(n_cal // 2), np.zeros(n_cal // 2)]).astype(int)

    # Create auditor
    auditor = ConformalDriftAuditor(nominal_coverage=0.95)
    cal_result = auditor.calibrate(cal_scores, cal_labels)

    print("CALIBRATION COMPLETE")
    print(f"  Threshold: {cal_result.threshold:.4f}")
    print(f"  Coverage: {cal_result.calibration_coverage.effective_coverage:.1%}")
    print(f"  FPR: {cal_result.calibration_fpr.fpr:.1%}")
    print()

    if args.scenario in ["rlhf", "both"]:
        # RLHF shift: faithful scores become similar to hallucinations
        print("SCENARIO 1: RLHF SHIFT")
        print("-" * 40)

        n_test = 200
        test_hall_scores = np.random.normal(0.48, 0.08, n_test // 2)
        test_faith_scores = np.random.normal(0.45, 0.07, n_test // 2)  # Now similar!
        test_scores = np.concatenate([test_hall_scores, test_faith_scores])
        test_labels = np.concatenate([np.ones(n_test // 2), np.zeros(n_test // 2)]).astype(int)

        result = auditor.add_shift(
            "NQ -> HaluEval (RLHF)",
            "rlhf",
            test_scores,
            test_labels,
        )

        print(f"  Coverage: {result.coverage_metrics.effective_coverage:.1%}")
        print(f"  FPR: {result.fpr_metrics.fpr:.1%}")
        print(f"  Verdict: {result.verdict.status}")
        print(f"  {result.verdict.recommendation}")
        print()

    if args.scenario in ["temporal", "both"]:
        # Temporal shift: hallucination scores decay
        print("SCENARIO 2: TEMPORAL SHIFT")
        print("-" * 40)

        n_test = 200
        test_hall_scores = np.random.normal(0.35, 0.08, n_test // 2)  # Decayed
        test_faith_scores = np.random.normal(0.1, 0.03, n_test // 2)
        test_scores = np.concatenate([test_hall_scores, test_faith_scores])
        test_labels = np.concatenate([np.ones(n_test // 2), np.zeros(n_test // 2)]).astype(int)

        result = auditor.add_shift(
            "NQ -> RAGTruth (Temporal)",
            "temporal",
            test_scores,
            test_labels,
        )

        print(f"  Coverage: {result.coverage_metrics.effective_coverage:.1%}")
        print(f"  FPR: {result.fpr_metrics.fpr:.1%}")
        print(f"  Verdict: {result.verdict.status}")
        print(f"  {result.verdict.recommendation}")
        print()

    # Generate final report
    report = auditor.generate_report()
    print(report.summary())


if __name__ == "__main__":
    main()
