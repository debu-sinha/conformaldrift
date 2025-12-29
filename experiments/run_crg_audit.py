#!/usr/bin/env python3
"""
CRG (Conformal RAG Guardrails) Experiment for Audit Matrix
Tests CRG detector under different distribution shift scenarios

Generates results for Table 1 (Audit Matrix) in the ConformalDrift paper.
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detectors.crg_detector import (
    ConformalRAGGuardrails,
    CRGConfig,
    RAGExample
)


def load_dataset(path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def dataset_to_examples(data: List[Dict]) -> List[RAGExample]:
    """Convert dataset dicts to RAGExample objects."""
    examples = []
    for item in data:
        examples.append(RAGExample(
            query=item['query'],
            documents=item['documents'],
            response=item['response'],
            label=item['label']
        ))
    return examples


def evaluate_guardrail(
    model: ConformalRAGGuardrails,
    examples: List[RAGExample],
    threshold: float
) -> Dict:
    """Evaluate CRG model on test set."""
    scores = model.compute_individual_scores(examples)
    ensemble = model.compute_ensemble_score(scores)

    labels = torch.tensor([ex.label for ex in examples])
    predictions = (ensemble >= threshold).float().cpu()
    labels_cpu = labels.cpu()

    hall_mask = labels_cpu == 1
    faith_mask = labels_cpu == 0

    n_hall = hall_mask.sum().item()
    n_faith = faith_mask.sum().item()

    coverage = (predictions[hall_mask] == 1).float().mean().item() if n_hall > 0 else 0
    fpr = (predictions[faith_mask] == 1).float().mean().item() if n_faith > 0 else 0

    return {
        'coverage': coverage * 100,
        'fpr': fpr * 100,
        'n_hallucinated': n_hall,
        'n_faithful': n_faith,
        'n_total': len(examples)
    }


def get_verdict(coverage: float, fpr: float, target_coverage: float = 95.0) -> Dict:
    """Compute 2-axis verdict."""
    delta_cov = coverage - target_coverage

    # Validity axis (based on coverage)
    if delta_cov >= -2:
        validity = "VALID"
    elif delta_cov >= -10:
        validity = "DEGRADED"
    else:
        validity = "INVALID"

    # Usability axis (based on FPR)
    if fpr <= 20:
        usability = "USABLE"
    elif fpr <= 50:
        usability = "DEGRADED"
    else:
        usability = "UNUSABLE"

    # Final verdict
    if validity == "VALID" and usability == "USABLE":
        verdict = "DEPLOY"
    elif validity == "INVALID" or usability == "UNUSABLE":
        verdict = "BLOCK"
    else:
        verdict = "MONITOR"

    # Failure mode
    if delta_cov < -10:
        mode = "C"  # Coverage collapse
    elif fpr > 50:
        mode = "F"  # FPR explosion
    else:
        mode = "S"  # Stable

    return {
        'delta_cov': delta_cov,
        'validity_axis': validity,
        'usability_axis': usability,
        'verdict': verdict,
        'failure_mode': mode
    }


def run_crg_audit(
    nq_cal_path: str,
    nq_test_path: str,
    ragtruth_path: str,
    halueval_path: str,
    target_coverage: float = 0.95,
    device: str = "cpu",
    seed: int = 42
) -> Dict:
    """Run CRG audit experiment."""

    print("=" * 70)
    print("ConformalDrift: CRG Audit Matrix Experiment")
    print("=" * 70)
    print(f"Target Coverage: {target_coverage:.0%}")
    print(f"Device: {device}")
    print()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize CRG model
    print("Initializing CRG model...")
    config = CRGConfig(
        device=device,
        use_fp16=device == "cuda",
        use_weighted_rad=True,
        use_sentence_level_sec=True,
        grounding_threshold=0.3,
        temperature=0.1,
    )
    model = ConformalRAGGuardrails(config)

    # Load calibration data
    print("\nLoading calibration data (NQ)...")
    nq_cal_data = load_dataset(nq_cal_path)
    cal_examples = dataset_to_examples(nq_cal_data)
    cal_hallucinated = [ex for ex in cal_examples if ex.label == 1]
    print(f"  Total: {len(cal_examples)}, Hallucinated: {len(cal_hallucinated)}")

    # Calibrate
    print("\nCalibrating CRG...")
    alpha = 1 - target_coverage
    model.calibrate(cal_hallucinated, alpha=alpha)
    threshold = model._threshold
    print(f"  Threshold: {threshold:.4f}")

    results = {
        'experiment': 'crg_audit_matrix',
        'guardrail': 'CRG',
        'target_coverage': target_coverage * 100,
        'threshold': threshold,
        'settings': []
    }

    # Test on different distributions
    test_configs = [
        ('NQ -> NQ (In-Dist)', nq_test_path),
        ('NQ -> RAGTruth (Cross-Dataset)', ragtruth_path),
        ('NQ -> HaluEval (RLHF)', halueval_path),
    ]

    print("\n" + "-" * 70)
    print("Evaluating on test distributions...")
    print("-" * 70)

    for setting_name, test_path in test_configs:
        print(f"\n{setting_name}")

        test_data = load_dataset(test_path)
        test_examples = dataset_to_examples(test_data)

        metrics = evaluate_guardrail(model, test_examples, threshold)
        verdict_info = get_verdict(metrics['coverage'], metrics['fpr'], target_coverage * 100)

        print(f"  N={metrics['n_total']} (H={metrics['n_hallucinated']}, F={metrics['n_faithful']})")
        print(f"  Coverage: {metrics['coverage']:.1f}%")
        print(f"  FPR: {metrics['fpr']:.1f}%")
        print(f"  Verdict: {verdict_info['verdict']} (V={verdict_info['validity_axis']}, U={verdict_info['usability_axis']})")

        results['settings'].append({
            'setting': setting_name,
            'n_test': metrics['n_total'],
            'n_hallucinated': metrics['n_hallucinated'],
            'n_faithful': metrics['n_faithful'],
            'effective_coverage': round(metrics['coverage'], 1),
            'fpr': round(metrics['fpr'], 1),
            'threshold': round(threshold, 4),
            **verdict_info
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: CRG Audit Matrix")
    print("=" * 70)
    print(f"{'Setting':<35} {'Cov_eff':>8} {'FPR':>8} {'Verdict':>10}")
    print("-" * 70)

    for s in results['settings']:
        print(f"{s['setting']:<35} {s['effective_coverage']:>7.1f}% {s['fpr']:>7.1f}% {s['verdict']:>10}")

    results['timestamp'] = datetime.now().isoformat()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run CRG Audit Matrix experiment")
    parser.add_argument("--nq-cal", type=str, default="data/nq_calibration.json")
    parser.add_argument("--nq-test", type=str, default="data/nq_test.json")
    parser.add_argument("--ragtruth", type=str, default="data/ragtruth_test.json")
    parser.add_argument("--halueval", type=str, default="data/halueval_test.json")
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    results = run_crg_audit(
        nq_cal_path=args.nq_cal,
        nq_test_path=args.nq_test,
        ragtruth_path=args.ragtruth,
        halueval_path=args.halueval,
        target_coverage=args.coverage,
        device=args.device,
        seed=args.seed
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = args.output or (output_dir / "crg_audit_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
