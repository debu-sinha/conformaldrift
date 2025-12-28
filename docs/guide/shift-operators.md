# Shift Operators

Conformal-Drift provides several shift operators to simulate different types of distribution shift.

## Overview

| Operator | Description | Use Case |
|----------|-------------|----------|
| `temporal` | Simulates time-based drift | News, social media, evolving domains |
| `semantic` | Changes meaning/topic | Domain transfer, topic shift |
| `lexical` | Surface-level text changes | Paraphrasing, style variation |
| `adversarial` | Targeted perturbations | Security testing, robustness |

## Temporal Shift

Simulates how data distributions change over time.

### How It Works

1. Sorts or weights samples by temporal proximity
2. At higher intensities, samples from further time periods
3. Mimics deployment scenarios where models see data from different time periods

### Usage

```python
results = auditor.audit(
    test_data=test_data,
    shift_type="temporal",
    shift_intensity=[0.0, 0.25, 0.5, 0.75, 1.0],
    shift_params={
        "time_column": "timestamp",  # Column with temporal info
        "direction": "forward"       # or "backward"
    }
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `time_column` | str | Name of timestamp column |
| `direction` | str | "forward" or "backward" |
| `window_size` | int | Size of temporal window |

## Semantic Shift

Changes the semantic content while preserving surface form.

### How It Works

1. Computes embeddings for all samples
2. At higher intensities, samples from more distant regions of embedding space
3. Simulates topic or domain shift

### Usage

```python
results = auditor.audit(
    test_data=test_data,
    shift_type="semantic",
    shift_intensity=[0.0, 0.25, 0.5, 0.75, 1.0],
    shift_params={
        "embedding_model": "all-MiniLM-L6-v2",
        "cluster_method": "kmeans"
    }
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedding_model` | str | Sentence transformer model |
| `cluster_method` | str | "kmeans", "dbscan", or "hierarchical" |
| `n_clusters` | int | Number of semantic clusters |

## Lexical Shift

Applies surface-level text transformations.

### How It Works

1. Applies text transformations (synonyms, paraphrasing)
2. At higher intensities, applies more aggressive transformations
3. Tests robustness to surface variation

### Usage

```python
results = auditor.audit(
    test_data=test_data,
    shift_type="lexical",
    shift_intensity=[0.0, 0.25, 0.5, 0.75, 1.0],
    shift_params={
        "transforms": ["synonym", "paraphrase", "typo"],
        "preserve_meaning": True
    }
)
```

### Available Transformations

| Transform | Description |
|-----------|-------------|
| `synonym` | Replace words with synonyms |
| `paraphrase` | Rephrase sentences |
| `typo` | Introduce realistic typos |
| `case` | Change capitalization |
| `contraction` | Expand/contract (e.g., "don't" ↔ "do not") |

## Adversarial Shift

Applies targeted perturbations to maximize failure.

### How It Works

1. Identifies samples near decision boundaries
2. Applies perturbations to push samples across boundaries
3. Tests worst-case robustness

### Usage

```python
results = auditor.audit(
    test_data=test_data,
    shift_type="adversarial",
    shift_intensity=[0.0, 0.25, 0.5, 0.75, 1.0],
    shift_params={
        "attack_type": "textfooler",
        "max_perturbation": 0.1
    }
)
```

!!! warning "Computational Cost"
    Adversarial shifts are significantly more expensive to compute than other shift types.

## Custom Shift Operators

You can define custom shift operators by implementing the `ShiftOperator` interface:

```python
from conformal_drift import ShiftOperator

class CustomShiftOperator(ShiftOperator):
    """Custom shift operator for domain-specific needs."""

    def __init__(self, **params):
        self.params = params

    def apply(self, data, intensity: float):
        """
        Apply shift to data at given intensity.

        Args:
            data: Input data to shift
            intensity: Float in [0, 1], where 0 = no shift

        Returns:
            Shifted data
        """
        # Implement your shift logic
        shifted_data = self._custom_transform(data, intensity)
        return shifted_data

    def _custom_transform(self, data, intensity):
        # Your transformation logic here
        pass

# Register and use
auditor.register_shift_operator("custom", CustomShiftOperator)
results = auditor.audit(test_data, shift_type="custom")
```

## Combining Shift Operators

Apply multiple shifts simultaneously:

```python
results = auditor.audit(
    test_data=test_data,
    shift_type=["temporal", "lexical"],
    shift_intensity=[0.0, 0.25, 0.5],
    combination="sequential"  # or "parallel"
)
```

### Combination Methods

| Method | Description |
|--------|-------------|
| `sequential` | Apply shifts one after another |
| `parallel` | Apply all shifts simultaneously |
| `max` | Take worst-case across shifts |

## Best Practices

1. **Start with mild shifts**: Begin at low intensity to establish baseline
2. **Use multiple shift types**: Different shifts reveal different failure modes
3. **Match deployment scenario**: Choose shifts that reflect real-world conditions
4. **Document parameters**: Record exact shift configurations for reproducibility
