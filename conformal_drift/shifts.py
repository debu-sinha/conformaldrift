"""
Shift injection utilities for ConformalDrift.

This module provides utilities for simulating and injecting various types
of distribution shifts commonly encountered in RAG systems:

- RLHF Shift: Semantic shift from RLHF-trained model hallucinations
- Temporal Shift: Knowledge drift from API/documentation version changes
- Domain Shift: Transfer to different domain than calibration
- Cross-Dataset Shift: Deployment on different dataset distribution
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class ShiftConfig:
    """Configuration for a distribution shift."""

    name: str
    shift_type: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class ShiftedData:
    """Container for shifted data."""

    scores: np.ndarray
    labels: np.ndarray
    shift_config: ShiftConfig
    metadata: Dict[str, Any]


def inject_score_shift(
    scores: np.ndarray,
    labels: np.ndarray,
    shift_magnitude: float = 0.1,
    shift_direction: str = "decrease",
    target_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject additive shift to scores for a specific class.

    This simulates scenarios where the nonconformity score distribution
    shifts for one class (e.g., hallucinations become harder to detect).

    Args:
        scores: Original nonconformity scores
        labels: Binary labels (1 = hallucinated, 0 = faithful)
        shift_magnitude: Amount to shift scores (as fraction of range)
        shift_direction: "increase" or "decrease"
        target_class: Which class to shift (default: 1 for hallucinations)

    Returns:
        Tuple of (shifted_scores, labels)
    """
    scores = np.asarray(scores).copy()
    labels = np.asarray(labels)

    score_range = scores.max() - scores.min()
    shift_amount = score_range * shift_magnitude

    if shift_direction == "decrease":
        shift_amount = -shift_amount

    mask = labels == target_class
    scores[mask] += shift_amount

    return scores, labels


def inject_overlap_shift(
    scores: np.ndarray,
    labels: np.ndarray,
    target_overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject shift to achieve target overlap between class distributions.

    This simulates scenarios where hallucination and faithful score
    distributions become harder to separate.

    Args:
        scores: Original nonconformity scores
        labels: Binary labels
        target_overlap: Desired overlap coefficient (0 = no overlap, 1 = complete)

    Returns:
        Tuple of (shifted_scores, labels)
    """
    scores = np.asarray(scores).copy()
    labels = np.asarray(labels)

    hall_scores = scores[labels == 1]
    faith_scores = scores[labels == 0]

    hall_mean = hall_scores.mean()
    faith_mean = faith_scores.mean()

    # Move faithful scores toward hallucination scores
    target_faith_mean = faith_mean + target_overlap * (hall_mean - faith_mean)
    shift = target_faith_mean - faith_mean

    scores[labels == 0] += shift

    return scores, labels


def mix_distributions(
    source_scores: np.ndarray,
    source_labels: np.ndarray,
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    drift_dose: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mix source and target distributions with specified drift dose.

    Drift dose controls the fraction of samples from target distribution:
    - dose=0.0: All source (no drift)
    - dose=0.5: 50% source, 50% target
    - dose=1.0: All target (full drift)

    Args:
        source_scores: Scores from source distribution
        source_labels: Labels from source distribution
        target_scores: Scores from target distribution
        target_labels: Labels from target distribution
        drift_dose: Fraction of target samples (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (mixed_scores, mixed_labels)
    """
    rng = np.random.default_rng(seed)

    source_scores = np.asarray(source_scores)
    source_labels = np.asarray(source_labels)
    target_scores = np.asarray(target_scores)
    target_labels = np.asarray(target_labels)

    # Determine sample sizes
    total_size = len(target_scores)
    n_target = int(total_size * drift_dose)
    n_source = total_size - n_target

    # Sample from each distribution
    if n_source > 0:
        source_idx = rng.choice(len(source_scores), n_source, replace=True)
        source_sample_scores = source_scores[source_idx]
        source_sample_labels = source_labels[source_idx]
    else:
        source_sample_scores = np.array([])
        source_sample_labels = np.array([])

    if n_target > 0:
        target_idx = rng.choice(len(target_scores), n_target, replace=False)
        target_sample_scores = target_scores[target_idx]
        target_sample_labels = target_labels[target_idx]
    else:
        target_sample_scores = np.array([])
        target_sample_labels = np.array([])

    # Combine and shuffle
    mixed_scores = np.concatenate([source_sample_scores, target_sample_scores])
    mixed_labels = np.concatenate([source_sample_labels, target_sample_labels])

    shuffle_idx = rng.permutation(len(mixed_scores))
    return mixed_scores[shuffle_idx], mixed_labels[shuffle_idx]


def create_rlhf_shift(
    source_scores: np.ndarray,
    source_labels: np.ndarray,
    faithful_score_mean: float,
    faithful_score_std: float,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> ShiftedData:
    """
    Simulate RLHF-induced shift where faithful responses get high scores.

    Under RLHF shift, the model produces confident but incorrect responses,
    causing faithful responses to have score distributions similar to
    hallucinations.

    Args:
        source_scores: Scores from source distribution
        source_labels: Labels from source distribution
        faithful_score_mean: Mean score for faithful responses under RLHF
        faithful_score_std: Std dev for faithful response scores
        n_samples: Number of samples to generate (default: same as source)
        seed: Random seed

    Returns:
        ShiftedData with simulated RLHF shift
    """
    rng = np.random.default_rng(seed)

    source_scores = np.asarray(source_scores)
    source_labels = np.asarray(source_labels)

    n_samples = n_samples or len(source_scores)

    # Keep hallucination score distribution
    hall_scores = source_scores[source_labels == 1]
    hall_mean = hall_scores.mean()
    hall_std = hall_scores.std()

    # Generate shifted data
    n_hall = n_samples // 2
    n_faith = n_samples - n_hall

    shifted_hall_scores = rng.normal(hall_mean, hall_std, n_hall)
    shifted_faith_scores = rng.normal(faithful_score_mean, faithful_score_std, n_faith)

    shifted_scores = np.concatenate([shifted_hall_scores, shifted_faith_scores])
    shifted_labels = np.concatenate([np.ones(n_hall), np.zeros(n_faith)])

    # Shuffle
    shuffle_idx = rng.permutation(len(shifted_scores))
    shifted_scores = shifted_scores[shuffle_idx]
    shifted_labels = shifted_labels[shuffle_idx].astype(int)

    config = ShiftConfig(
        name="RLHF Shift",
        shift_type="rlhf",
        description="Faithful responses have high scores due to RLHF training",
        parameters={
            "faithful_score_mean": faithful_score_mean,
            "faithful_score_std": faithful_score_std,
        },
    )

    return ShiftedData(
        scores=shifted_scores,
        labels=shifted_labels,
        shift_config=config,
        metadata={
            "source_hall_mean": float(hall_mean),
            "shifted_faith_mean": faithful_score_mean,
            "overlap": _compute_overlap(shifted_hall_scores, shifted_faith_scores),
        },
    )


def create_temporal_shift(
    source_scores: np.ndarray,
    source_labels: np.ndarray,
    score_decay_rate: float = 0.2,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> ShiftedData:
    """
    Simulate temporal/knowledge drift where scores decay over time.

    As documentation or APIs change, the nonconformity scores may shift
    because the grounding context becomes outdated.

    Args:
        source_scores: Scores from source distribution
        source_labels: Labels from source distribution
        score_decay_rate: Rate of score decay (fraction of mean)
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        ShiftedData with simulated temporal shift
    """
    rng = np.random.default_rng(seed)

    source_scores = np.asarray(source_scores)
    source_labels = np.asarray(source_labels)

    n_samples = n_samples or len(source_scores)

    # Sample from source
    idx = rng.choice(len(source_scores), n_samples, replace=True)
    shifted_scores = source_scores[idx].copy()
    shifted_labels = source_labels[idx].copy()

    # Apply decay to hallucination scores (makes them harder to detect)
    hall_mask = shifted_labels == 1
    decay_amount = shifted_scores[hall_mask].mean() * score_decay_rate
    shifted_scores[hall_mask] -= decay_amount

    config = ShiftConfig(
        name="Temporal Shift",
        shift_type="temporal",
        description="Hallucination scores decay due to knowledge drift",
        parameters={"score_decay_rate": score_decay_rate},
    )

    return ShiftedData(
        scores=shifted_scores,
        labels=shifted_labels,
        shift_config=config,
        metadata={
            "decay_amount": float(decay_amount),
            "original_hall_mean": float(source_scores[source_labels == 1].mean()),
            "shifted_hall_mean": float(shifted_scores[hall_mask].mean()),
        },
    )


def _compute_overlap(dist1: np.ndarray, dist2: np.ndarray) -> float:
    """Compute overlap coefficient between two distributions."""
    # Simple histogram-based overlap
    min_val = min(dist1.min(), dist2.min())
    max_val = max(dist1.max(), dist2.max())

    bins = np.linspace(min_val, max_val, 50)

    hist1, _ = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, density=True)

    # Overlap is integral of min(p1, p2)
    overlap = np.minimum(hist1, hist2).sum() * (bins[1] - bins[0])
    return float(overlap)


# Preset shift configurations
SHIFT_PRESETS = {
    "rlhf_mild": ShiftConfig(
        name="Mild RLHF Shift",
        shift_type="rlhf",
        description="Mild RLHF-induced score overlap",
        parameters={"overlap_increase": 0.2},
    ),
    "rlhf_severe": ShiftConfig(
        name="Severe RLHF Shift",
        shift_type="rlhf",
        description="Severe RLHF collapse (near-complete overlap)",
        parameters={"overlap_increase": 0.8},
    ),
    "temporal_weekly": ShiftConfig(
        name="Weekly Temporal Drift",
        shift_type="temporal",
        description="Typical weekly knowledge drift",
        parameters={"score_decay_rate": 0.05},
    ),
    "temporal_monthly": ShiftConfig(
        name="Monthly Temporal Drift",
        shift_type="temporal",
        description="Monthly knowledge drift accumulation",
        parameters={"score_decay_rate": 0.15},
    ),
    "domain_mild": ShiftConfig(
        name="Mild Domain Shift",
        shift_type="domain",
        description="Shift to related domain",
        parameters={"distribution_shift": 0.1},
    ),
    "domain_severe": ShiftConfig(
        name="Severe Domain Shift",
        shift_type="domain",
        description="Shift to unrelated domain",
        parameters={"distribution_shift": 0.4},
    ),
}
