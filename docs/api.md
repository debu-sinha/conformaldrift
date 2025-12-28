# API Reference

Complete API documentation for Conformal-Drift.

## ConformalDriftAuditor

The main class for running distribution shift audits.

::: conformal_drift.ConformalDriftAuditor
    options:
      show_root_heading: true
      show_source: true

### Constructor

```python
ConformalDriftAuditor(
    calibration_scores: np.ndarray,
    alpha: float = 0.1,
    random_state: Optional[int] = None,
    device: str = "cpu"
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `calibration_scores` | np.ndarray | required | Nonconformity scores from calibration set |
| `alpha` | float | 0.1 | Target miscoverage rate |
| `random_state` | int, optional | None | Random seed for reproducibility |
| `device` | str | "cpu" | Device for computations ("cpu" or "cuda") |

### Methods

#### audit

```python
def audit(
    self,
    test_data: Dict[str, Any],
    shift_type: str = "temporal",
    shift_intensity: Union[List[float], np.ndarray] = None,
    shift_params: Optional[Dict] = None,
    batch_size: Optional[int] = None
) -> AuditResults
```

Run a distribution shift audit.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_data` | dict | required | Test data with 'inputs', 'labels', 'scores' |
| `shift_type` | str | "temporal" | Type of shift to apply |
| `shift_intensity` | list | [0, 0.25, 0.5, 0.75, 1.0] | Shift intensity levels |
| `shift_params` | dict | None | Additional parameters for shift operator |
| `batch_size` | int | None | Batch size for processing |

**Returns:** `AuditResults` object

#### comprehensive_audit

```python
def comprehensive_audit(
    self,
    test_data: Dict[str, Any],
    shift_types: List[str] = None,
    shift_intensity: Union[List[float], np.ndarray] = None
) -> Dict[str, AuditResults]
```

Run audits across multiple shift types.

#### generate_report

```python
def generate_report(
    self,
    results: Union[AuditResults, Dict[str, AuditResults]],
    output_format: str = "markdown"
) -> str
```

Generate a formatted audit report.

---

## AuditResults

Container for audit results.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shift_type` | str | Type of shift applied |
| `shift_intensities` | np.ndarray | Array of shift intensity levels |
| `coverage` | np.ndarray | Empirical coverage at each level |
| `set_sizes` | np.ndarray | Average set size at each level |
| `coverage_gap` | float | Maximum coverage gap observed |
| `critical_intensity` | float | Intensity where coverage drops below threshold |

### Methods

#### save

```python
def save(self, path: str) -> None
```

Save results to JSON file.

#### load

```python
@classmethod
def load(cls, path: str) -> "AuditResults"
```

Load results from JSON file.

---

## Shift Operators

### ShiftOperator (Base Class)

```python
class ShiftOperator(ABC):
    @abstractmethod
    def apply(self, data: Any, intensity: float) -> Any:
        """Apply shift at given intensity."""
        pass
```

### TemporalShiftOperator

```python
from conformal_drift.shifts import TemporalShiftOperator

operator = TemporalShiftOperator(
    time_column="timestamp",
    direction="forward"
)
```

### SemanticShiftOperator

```python
from conformal_drift.shifts import SemanticShiftOperator

operator = SemanticShiftOperator(
    embedding_model="all-MiniLM-L6-v2",
    cluster_method="kmeans",
    n_clusters=10
)
```

### LexicalShiftOperator

```python
from conformal_drift.shifts import LexicalShiftOperator

operator = LexicalShiftOperator(
    transforms=["synonym", "paraphrase"],
    preserve_meaning=True
)
```

---

## Visualization

### plot_coverage_curve

```python
from conformal_drift.viz import plot_coverage_curve

fig = plot_coverage_curve(
    results: AuditResults,
    nominal_coverage: float = 0.9,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure
```

### plot_coverage_comparison

```python
from conformal_drift.viz import plot_coverage_comparison

fig = plot_coverage_comparison(
    results_dict: Dict[str, AuditResults],
    nominal_coverage: float = 0.9
) -> plt.Figure
```

### plot_set_size_distribution

```python
from conformal_drift.viz import plot_set_size_distribution

fig = plot_set_size_distribution(
    results: AuditResults,
    bins: int = 20
) -> plt.Figure
```

---

## Utilities

### compute_quantile

```python
from conformal_drift.utils import compute_quantile

quantile = compute_quantile(
    scores: np.ndarray,
    alpha: float
) -> float
```

Compute conformal quantile threshold.

### validate_coverage

```python
from conformal_drift.utils import validate_coverage

is_valid, gap = validate_coverage(
    empirical: float,
    nominal: float,
    tolerance: float = 0.02
) -> Tuple[bool, float]
```

Check if empirical coverage meets nominal guarantee.
