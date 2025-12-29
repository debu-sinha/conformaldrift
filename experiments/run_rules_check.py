#!/usr/bin/env python3
"""
Rules-Check Guardrail Experiment for ConformalDrift Paper

Implements a simple rule-based guardrail as a third method for the Audit Matrix.
This provides contrast with semantic methods (CRG, Emb-Sim).

Rules-Check scores based on:
1. Citation/reference presence
2. Hedge word usage ("I think", "might be", "probably")
3. Response length anomalies
4. Query echoing (response just repeats the question)
"""
import os
import sys
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detectors.crg_detector import RAGExample


# Hedge words that indicate uncertainty
HEDGE_WORDS = [
    "i think", "i believe", "might be", "could be", "probably",
    "possibly", "perhaps", "maybe", "it seems", "appears to",
    "i'm not sure", "not certain", "uncertain", "unclear",
    "approximately", "roughly", "around", "about"
]

# Citation patterns
CITATION_PATTERNS = [
    r"according to",
    r"as stated in",
    r"the documentation says",
    r"based on the",
    r"from the context",
    r"the source indicates",
    r"as mentioned",
    r"per the",
]


@dataclass
class RulesCheckConfig:
    """Configuration for Rules-Check guardrail."""
    hedge_weight: float = 0.25
    citation_weight: float = 0.30
    length_weight: float = 0.25
    echo_weight: float = 0.20
    min_response_words: int = 5
    max_response_words: int = 500


class RulesCheckGuardrail:
    """
    Simple rule-based hallucination detector.

    Higher scores indicate higher likelihood of hallucination.
    """

    def __init__(self, config: Optional[RulesCheckConfig] = None):
        self.config = config or RulesCheckConfig()
        self._threshold = None

    def _count_hedge_words(self, text: str) -> int:
        """Count hedge words/phrases in text."""
        text_lower = text.lower()
        count = 0
        for hedge in HEDGE_WORDS:
            count += text_lower.count(hedge)
        return count

    def _has_citation(self, text: str) -> bool:
        """Check if response contains citation patterns."""
        text_lower = text.lower()
        for pattern in CITATION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def _check_length_anomaly(self, text: str) -> float:
        """Check for response length anomalies. Returns 0-1 score."""
        words = len(text.split())
        if words < self.config.min_response_words:
            return 1.0  # Too short
        elif words > self.config.max_response_words:
            return 0.5  # Too long (less severe)
        return 0.0

    def _check_echo(self, query: str, response: str) -> float:
        """Check if response just echoes the query."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        if len(response_words) == 0:
            return 1.0

        # Jaccard similarity
        intersection = len(query_words & response_words)
        union = len(query_words | response_words)

        if union == 0:
            return 0.0

        similarity = intersection / union

        # High similarity suggests echoing
        if similarity > 0.7:
            return 1.0
        elif similarity > 0.5:
            return 0.5
        return 0.0

    def compute_score(self, example: RAGExample) -> float:
        """
        Compute nonconformity score for a single example.
        Higher scores = more likely hallucination.
        """
        response = example.response
        query = example.query

        score = 0.0

        # 1. Hedge words (more hedging = more uncertain = higher score)
        hedge_count = self._count_hedge_words(response)
        hedge_score = min(hedge_count * 0.15, 1.0)
        score += hedge_score * self.config.hedge_weight

        # 2. Citation presence (no citation = higher score)
        has_citation = self._has_citation(response)
        citation_score = 0.0 if has_citation else 1.0
        score += citation_score * self.config.citation_weight

        # 3. Length anomaly
        length_score = self._check_length_anomaly(response)
        score += length_score * self.config.length_weight

        # 4. Query echoing
        echo_score = self._check_echo(query, response)
        score += echo_score * self.config.echo_weight

        return min(score, 1.0)

    def compute_scores(self, examples: List[RAGExample]) -> np.ndarray:
        """Compute scores for a batch of examples."""
        scores = []
        for ex in tqdm(examples, desc="Computing Rules-Check scores"):
            scores.append(self.compute_score(ex))
        return np.array(scores)

    def calibrate(self, hallucinated_examples: List[RAGExample], alpha: float = 0.05):
        """
        Calibrate threshold on hallucinated examples for (1-alpha) coverage.
        """
        scores = self.compute_scores(hallucinated_examples)
        n = len(scores)

        # Conformal quantile (clamp to valid range)
        quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
        quantile_level = min(quantile_level, 1.0)
        self._threshold = np.quantile(scores, max(0, 1 - quantile_level))

        print(f"  Calibrated threshold: {self._threshold:.4f}")
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Score mean: {scores.mean():.4f}")

        return self._threshold

    def predict(self, examples: List[RAGExample]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict hallucinations using calibrated threshold.
        Returns (predictions, scores).
        """
        if self._threshold is None:
            raise ValueError("Must calibrate before predicting")

        scores = self.compute_scores(examples)
        predictions = (scores >= self._threshold).astype(int)

        return predictions, scores


def load_json_data(data_path: str) -> List[RAGExample]:
    """Load data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    examples = []
    for item in data:
        examples.append(RAGExample(
            query=item.get('query', item.get('question', '')),
            documents=item.get('documents', [item.get('context', item.get('evidence', ''))]),
            response=item.get('response', item.get('answer', '')),
            label=item.get('label', None)
        ))
    return examples


def compute_2axis_verdict(coverage: float, fpr: float, target_coverage: float = 0.95) -> Dict:
    """
    Compute 2-axis verdict for audit report.

    Axis 1: Guarantee Validity (based on |delta_cov|)
    Axis 2: Operational Usability (based on FPR)
    """
    delta_cov = target_coverage - coverage

    # Axis 1: Guarantee Validity
    if abs(delta_cov) <= 0.05:
        validity = "VALID"
    elif abs(delta_cov) <= 0.15:
        validity = "DEGRADED"
    else:
        validity = "INVALID"

    # Axis 2: Operational Usability
    if fpr <= 0.25:
        usability = "USABLE"
    elif fpr <= 0.50:
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
    if delta_cov > 0.15:
        failure_mode = "C"  # Coverage collapse
    elif fpr > 0.50:
        failure_mode = "F"  # FPR explosion
    else:
        failure_mode = "S"  # Stable

    return {
        'delta_cov': round(delta_cov * 100, 1),
        'validity_axis': validity,
        'usability_axis': usability,
        'verdict': verdict,
        'failure_mode': failure_mode
    }


def evaluate_guardrail(
    model: RulesCheckGuardrail,
    test_examples: List[RAGExample],
    setting_name: str,
    target_coverage: float = 0.95
) -> Dict:
    """Evaluate guardrail on test set with 2-axis verdict."""
    predictions, scores = model.predict(test_examples)
    labels = np.array([ex.label for ex in test_examples])

    # Coverage (recall on hallucinations)
    hall_mask = labels == 1
    faith_mask = labels == 0

    if hall_mask.sum() > 0:
        coverage = predictions[hall_mask].mean()
    else:
        coverage = 0.0

    if faith_mask.sum() > 0:
        fpr = predictions[faith_mask].mean()
    else:
        fpr = 0.0

    # Get 2-axis verdict
    verdict_info = compute_2axis_verdict(coverage, fpr, target_coverage)

    result = {
        'setting': setting_name,
        'n_test': len(test_examples),
        'n_hallucinated': int(hall_mask.sum()),
        'n_faithful': int(faith_mask.sum()),
        'effective_coverage': round(coverage * 100, 1),
        'fpr': round(fpr * 100, 1),
        'threshold': round(model._threshold, 4),
        **verdict_info
    }

    print(f"\n{setting_name}:")
    print(f"  Coverage: {coverage:.1%} (delta: {verdict_info['delta_cov']:+.1f}%)")
    print(f"  FPR: {fpr:.1%}")
    print(f"  Verdict: {verdict_info['verdict']} ({verdict_info['failure_mode']})")
    print(f"  Axes: {verdict_info['validity_axis']} / {verdict_info['usability_axis']}")

    return result


def run_rules_check_experiment(
    nq_cal_path: str,
    nq_test_path: str,
    ragtruth_path: str,
    halueval_path: Optional[str] = None,
    target_coverage: float = 0.95,
    seed: int = 42
) -> Dict:
    """Run the complete Rules-Check experiment."""

    print("=" * 70)
    print("ConformalDrift: Rules-Check Guardrail Experiment")
    print("=" * 70)

    np.random.seed(seed)

    # Initialize model
    model = RulesCheckGuardrail()

    results = {
        'experiment': 'rules_check_guardrail',
        'guardrail': 'Rules-Check',
        'target_coverage': target_coverage,
        'settings': [],
        'timestamp': datetime.now().isoformat()
    }

    # Load calibration data
    print("\nLoading calibration data (NQ)...")
    cal_examples = load_json_data(nq_cal_path)
    cal_hallucinated = [ex for ex in cal_examples if ex.label == 1]
    print(f"  Calibration examples: {len(cal_hallucinated)} hallucinated")

    # Calibrate
    print("\nCalibrating...")
    alpha = 1 - target_coverage
    model.calibrate(cal_hallucinated, alpha=alpha)

    # Setting 1: In-distribution (NQ -> NQ)
    print("\n" + "-" * 70)
    print("Setting 1: In-Distribution (NQ -> NQ)")
    print("-" * 70)

    nq_test = load_json_data(nq_test_path)
    result = evaluate_guardrail(model, nq_test, "NQ -> NQ (In-Dist)", target_coverage)
    results['settings'].append(result)

    # Setting 2: Cross-Dataset (NQ -> RAGTruth)
    print("\n" + "-" * 70)
    print("Setting 2: Cross-Dataset Shift (NQ -> RAGTruth)")
    print("-" * 70)

    ragtruth_test = load_json_data(ragtruth_path)
    result = evaluate_guardrail(model, ragtruth_test, "NQ -> RAGTruth (Cross-Dataset)", target_coverage)
    results['settings'].append(result)

    # Setting 3: RLHF Shift (NQ -> HaluEval) if available
    if halueval_path and os.path.exists(halueval_path):
        print("\n" + "-" * 70)
        print("Setting 3: RLHF Shift (NQ -> HaluEval)")
        print("-" * 70)

        halueval_test = load_json_data(halueval_path)
        result = evaluate_guardrail(model, halueval_test, "NQ -> HaluEval (RLHF)", target_coverage)
        results['settings'].append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Rules-Check Audit Matrix Results")
    print("=" * 70)
    print(f"{'Setting':<35} {'Cov':>8} {'FPR':>8} {'Verdict':>10}")
    print("-" * 70)

    for s in results['settings']:
        print(f"{s['setting']:<35} {s['effective_coverage']:>7.1f}% {s['fpr']:>7.1f}% {s['verdict']:>10}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Rules-Check guardrail experiment")
    parser.add_argument(
        "--nq-cal",
        type=str,
        default=r"data/nq_calibration.json",
        help="Path to NQ calibration data"
    )
    parser.add_argument(
        "--nq-test",
        type=str,
        default=r"data/nq_test.json",
        help="Path to NQ test data"
    )
    parser.add_argument(
        "--ragtruth",
        type=str,
        default=r"data/ragtruth_test.json",
        help="Path to RAGTruth test data"
    )
    parser.add_argument(
        "--halueval",
        type=str,
        default=None,
        help="Path to HaluEval test data (optional)"
    )
    parser.add_argument("--coverage", type=float, default=0.95, help="Target coverage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    results = run_rules_check_experiment(
        nq_cal_path=args.nq_cal,
        nq_test_path=args.nq_test,
        ragtruth_path=args.ragtruth,
        halueval_path=args.halueval,
        target_coverage=args.coverage,
        seed=args.seed
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = args.output or (output_dir / "rules_check_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
