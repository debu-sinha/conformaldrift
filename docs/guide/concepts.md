# Core Concepts

This guide explains the key concepts behind Conformal-Drift.

## Conformal Prediction

Conformal prediction is a framework for creating prediction sets with guaranteed coverage. Unlike point predictions, conformal predictors output a *set* of possible values that contains the true value with a specified probability.

### The Coverage Guarantee

For a target miscoverage rate α (e.g., α = 0.1 for 90% coverage), conformal prediction guarantees:

$$P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$$

This guarantee holds under the **exchangeability assumption**: calibration and test data come from the same distribution.

### Nonconformity Scores

Nonconformity scores measure how "strange" a sample is relative to calibration data:

- **Low scores**: Sample conforms well to calibration data
- **High scores**: Sample is unusual or difficult

Common choices:
- Classification: `1 - predicted_probability`
- Regression: `|y - y_pred|`
- RAG/LLM: Embedding distance, perplexity, or confidence scores

## Distribution Shift

Distribution shift occurs when test data differs from calibration data. This violates the exchangeability assumption and can break coverage guarantees.

### Types of Shift

| Type | Description | Example |
|------|-------------|---------|
| **Covariate shift** | P(X) changes, P(Y\|X) stays same | Different input demographics |
| **Label shift** | P(Y) changes, P(X\|Y) stays same | Class imbalance changes |
| **Concept drift** | P(Y\|X) changes | Meaning of terms evolves |

### Impact on Conformal Prediction

Under distribution shift:

1. **Coverage drops**: Prediction sets may not contain true values at the nominal rate
2. **Sets become uninformative**: Sets may become very large or very small
3. **Calibration degrades**: Quantile thresholds become unreliable

## The Audit Protocol

Conformal-Drift provides a systematic protocol to stress-test conformal predictors:

### 1. Establish Baseline

Measure coverage on in-distribution test data to verify the conformal predictor works correctly:

```python
baseline_results = auditor.audit(
    test_data=test_data,
    shift_intensity=[0.0]  # No shift
)
assert baseline_results.coverage[0] >= 0.9 - 0.02  # Within tolerance
```

### 2. Apply Graduated Shift

Incrementally increase shift intensity to observe degradation:

```python
results = auditor.audit(
    test_data=test_data,
    shift_intensity=[0.0, 0.25, 0.5, 0.75, 1.0]
)
```

### 3. Measure Coverage Gap

The coverage gap quantifies how much coverage drops below nominal:

$$\text{Coverage Gap} = (1 - \alpha) - \text{Empirical Coverage}$$

### 4. Identify Failure Modes

Analyze which shift types and intensities cause failures:

```python
for shift_type in ['temporal', 'semantic', 'lexical']:
    results = auditor.audit(test_data, shift_type=shift_type)
    if results.max_coverage_gap > 0.1:
        print(f"Critical failure under {shift_type} shift")
```

## Key Metrics

### Coverage

The fraction of samples where the true value is in the prediction set:

$$\text{Coverage} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}[Y_i \in C(X_i)]$$

### Set Size

Average size of prediction sets. Larger sets indicate higher uncertainty:

$$\text{Avg Set Size} = \frac{1}{n}\sum_{i=1}^{n} |C(X_i)|$$

### Efficiency

Inverse of set size. More efficient predictors produce smaller sets while maintaining coverage.

## Mathematical Background

### Split Conformal Prediction

Given calibration scores $\{s_1, ..., s_n\}$ and target miscoverage α:

1. Compute quantile: $\hat{q} = \text{Quantile}_{(1-\alpha)(1+1/n)}(\{s_1, ..., s_n\})$
2. Prediction set: $C(X_{new}) = \{y : s(X_{new}, y) \leq \hat{q}\}$

### Exchangeability Assumption

Samples $(X_1, Y_1), ..., (X_n, Y_n), (X_{n+1}, Y_{n+1})$ are exchangeable if their joint distribution is invariant to permutations.

Under exchangeability, calibration and test samples can be "swapped" without changing the distribution, which is why conformal prediction works.

## Further Reading

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*
- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction*
- Barber, R. F., et al. (2023). *Conformal Prediction Beyond Exchangeability*
