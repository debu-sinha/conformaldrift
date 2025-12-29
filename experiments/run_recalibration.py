#!/usr/bin/env python3
"""
Recalibration Budget Experiment for ConformalDrift Paper

Tests how many labeled examples (k) from the target distribution are needed
to recover coverage after distribution shift.

This experiment validates Table 3 (Recalibration Budget) of the paper:
- k=0: No recalibration (baseline cross-dataset performance)
- k=10, 25, 50, 100: Progressive recalibration with k target examples

Uses embedding similarity (Emb-Sim) as the guardrail for consistency.
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detectors.crg_detector import RAGExample


class EmbeddingSimilarityGuardrail:
    """
    Simple embedding-based hallucination detector.
    Uses cosine similarity between response and documents.
    Higher scores = lower similarity = more likely hallucination.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._encoder = None
        self._tokenizer = None
        self._threshold = None

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._encoder is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                print(f"  Loading encoder: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._encoder = AutoModel.from_pretrained(self.model_name)
                self._encoder = self._encoder.to(self.device)
                self._encoder.eval()
            except ImportError:
                print("  Warning: transformers not available, using fallback")
                self._encoder = "fallback"

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        self._load_model()

        if self._encoder == "fallback":
            # Fallback: simple TF-IDF-like encoding
            from sklearn.feature_extraction.text import TfidfVectorizer
            if not hasattr(self, '_vectorizer'):
                self._vectorizer = TfidfVectorizer(max_features=768)
                self._vectorizer.fit(texts)
            return self._vectorizer.transform(texts).toarray()

        import torch
        with torch.no_grad():
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self._encoder(**encoded)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            return embeddings.cpu().numpy()

    def compute_score(self, example: RAGExample) -> float:
        """
        Compute nonconformity score based on embedding similarity.
        Higher scores = less similar to documents = more likely hallucination.
        """
        response_emb = self._encode([example.response])
        doc_embs = self._encode(example.documents)

        # Cosine similarity
        similarities = np.dot(response_emb, doc_embs.T).flatten()
        max_sim = similarities.max()

        # Convert to nonconformity (1 - similarity)
        return float(1.0 - max_sim)

    def compute_scores(self, examples: List[RAGExample]) -> np.ndarray:
        """Compute scores for a batch of examples."""
        scores = []
        for ex in tqdm(examples, desc="Computing Emb-Sim scores"):
            scores.append(self.compute_score(ex))
        return np.array(scores)

    def calibrate(self, hallucinated_examples: List[RAGExample], alpha: float = 0.05) -> float:
        """Calibrate threshold on hallucinated examples."""
        scores = self.compute_scores(hallucinated_examples)
        n = len(scores)

        # Conformal quantile
        quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
        quantile_level = min(quantile_level, 1.0)
        self._threshold = np.quantile(scores, 1 - quantile_level)

        return self._threshold

    def recalibrate(
        self,
        original_scores: np.ndarray,
        new_examples: List[RAGExample],
        alpha: float = 0.05
    ) -> float:
        """
        Recalibrate using both original scores and new examples.
        This is the "mixed calibration" approach.
        """
        new_scores = self.compute_scores(new_examples)
        combined_scores = np.concatenate([original_scores, new_scores])

        n = len(combined_scores)
        quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
        quantile_level = min(quantile_level, 1.0)
        self._threshold = np.quantile(combined_scores, 1 - quantile_level)

        return self._threshold

    def predict(self, examples: List[RAGExample]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict hallucinations using calibrated threshold."""
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


def evaluate_at_k(
    model: EmbeddingSimilarityGuardrail,
    original_cal_scores: np.ndarray,
    target_examples: List[RAGExample],
    test_examples: List[RAGExample],
    k: int,
    alpha: float,
    seed: int = 42
) -> Dict:
    """
    Evaluate coverage after recalibrating with k target examples.
    """
    np.random.seed(seed)

    # Get labels
    labels = np.array([ex.label for ex in test_examples])
    hall_mask = labels == 1
    faith_mask = labels == 0

    if k == 0:
        # No recalibration - use original threshold
        predictions, scores = model.predict(test_examples)
    else:
        # Sample k hallucinated examples from target for recalibration
        target_hallucinated = [ex for ex in target_examples if ex.label == 1]
        if len(target_hallucinated) < k:
            # Not enough hallucinated examples, sample with replacement
            sampled_indices = np.random.choice(len(target_hallucinated), k, replace=True)
            recal_examples = [target_hallucinated[i] for i in sampled_indices]
        else:
            sampled_indices = np.random.choice(len(target_hallucinated), k, replace=False)
            recal_examples = [target_hallucinated[i] for i in sampled_indices]

        # Recalibrate with combined data
        model.recalibrate(original_cal_scores, recal_examples, alpha)

        # Evaluate on test set (excluding sampled examples)
        predictions, scores = model.predict(test_examples)

    # Compute metrics
    if hall_mask.sum() > 0:
        coverage = predictions[hall_mask].mean()
    else:
        coverage = 0.0

    if faith_mask.sum() > 0:
        fpr = predictions[faith_mask].mean()
    else:
        fpr = 0.0

    return {
        'k': k,
        'effective_coverage': round(coverage * 100, 1),
        'fpr': round(fpr * 100, 1),
        'threshold': round(model._threshold, 4),
        'n_recal_examples': k
    }


def run_recalibration_experiment(
    nq_cal_path: str,
    ragtruth_path: str,
    k_values: List[int] = [0, 10, 25, 50, 100],
    target_coverage: float = 0.95,
    device: str = "cpu",
    seed: int = 42
) -> Dict:
    """Run the complete recalibration budget experiment."""

    print("=" * 70)
    print("ConformalDrift: Recalibration Budget Experiment")
    print("=" * 70)

    np.random.seed(seed)

    # Initialize model
    print("\nInitializing Emb-Sim model...")
    model = EmbeddingSimilarityGuardrail(device=device)

    results = {
        'experiment': 'recalibration_budget',
        'guardrail': 'Emb-Sim',
        'source_distribution': 'NQ',
        'target_distribution': 'RAGTruth',
        'target_coverage': target_coverage,
        'k_values': k_values,
        'settings': [],
        'timestamp': datetime.now().isoformat()
    }

    # Load source calibration data (NQ)
    print("\nLoading source calibration data (NQ)...")
    nq_cal = load_json_data(nq_cal_path)
    nq_hallucinated = [ex for ex in nq_cal if ex.label == 1]
    print(f"  NQ hallucinated examples: {len(nq_hallucinated)}")

    # Calibrate on source
    print("\nCalibrating on source (NQ)...")
    alpha = 1 - target_coverage
    original_threshold = model.calibrate(nq_hallucinated, alpha=alpha)
    print(f"  Original threshold: {original_threshold:.4f}")

    # Store original calibration scores for recalibration
    original_cal_scores = model.compute_scores(nq_hallucinated)

    # Load target data (RAGTruth)
    print("\nLoading target data (RAGTruth)...")
    ragtruth = load_json_data(ragtruth_path)
    print(f"  RAGTruth examples: {len(ragtruth)}")

    # Split RAGTruth into recalibration pool and test set
    np.random.shuffle(ragtruth)
    n_recal_pool = min(150, len(ragtruth) // 2)  # Reserve up to 150 for recalibration
    recal_pool = ragtruth[:n_recal_pool]
    test_set = ragtruth[n_recal_pool:]

    print(f"  Recalibration pool: {len(recal_pool)}")
    print(f"  Test set: {len(test_set)}")

    # Evaluate at each k value
    print("\n" + "-" * 70)
    print("Recalibration Budget Analysis")
    print("-" * 70)

    for k in k_values:
        print(f"\nk = {k}:")

        # Reset model threshold for k=0
        if k == 0:
            model._threshold = original_threshold

        result = evaluate_at_k(
            model=model,
            original_cal_scores=original_cal_scores,
            target_examples=recal_pool,
            test_examples=test_set,
            k=k,
            alpha=alpha,
            seed=seed
        )

        print(f"  Coverage: {result['effective_coverage']:.1f}%")
        print(f"  FPR: {result['fpr']:.1f}%")
        print(f"  Threshold: {result['threshold']:.4f}")

        results['settings'].append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Recalibration Budget (Table 3)")
    print("=" * 70)
    print(f"{'k':>5} {'Cov_eff':>10} {'FPR':>8} {'Threshold':>12}")
    print("-" * 40)

    for s in results['settings']:
        print(f"{s['k']:>5} {s['effective_coverage']:>9.1f}% {s['fpr']:>7.1f}% {s['threshold']:>12.4f}")

    # Add analysis
    k0_cov = results['settings'][0]['effective_coverage']
    for s in results['settings']:
        if s['effective_coverage'] >= 90:
            results['k_for_90pct_coverage'] = s['k']
            break

    results['analysis'] = {
        'baseline_coverage_k0': k0_cov,
        'coverage_recovery': {
            k_res['k']: round(k_res['effective_coverage'] - k0_cov, 1)
            for k_res in results['settings']
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run recalibration budget experiment")
    parser.add_argument(
        "--nq-cal",
        type=str,
        default=r"data/nq_calibration.json",
        help="Path to NQ calibration data"
    )
    parser.add_argument(
        "--ragtruth",
        type=str,
        default=r"data/ragtruth_test.json",
        help="Path to RAGTruth data"
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="0,10,25,50,100",
        help="Comma-separated k values to test"
    )
    parser.add_argument("--coverage", type=float, default=0.95, help="Target coverage")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    k_values = [int(k.strip()) for k in args.k_values.split(',')]

    results = run_recalibration_experiment(
        nq_cal_path=args.nq_cal,
        ragtruth_path=args.ragtruth,
        k_values=k_values,
        target_coverage=args.coverage,
        device=args.device,
        seed=args.seed
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = args.output or (output_dir / "recalibration_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
