# Contributing

We welcome contributions to Conformal-Drift! This guide will help you get started.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/conformaldrift.git
cd conformaldrift
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We use the following tools:

- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

Run all checks:
```bash
black conformal_drift/ tests/
ruff check conformal_drift/ tests/
mypy conformal_drift/
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ -v --cov=conformal_drift --cov-report=html
```

## Pull Request Process

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and add tests

3. Run the test suite:
```bash
pytest tests/ -v
```

4. Commit with a descriptive message:
```bash
git commit -m "Add feature: description of what you added"
```

5. Push and create a pull request:
```bash
git push origin feature/your-feature-name
```

## Adding a New Shift Operator

1. Create a new file in `conformal_drift/shifts/`:
```python
# conformal_drift/shifts/my_shift.py
from conformal_drift.shifts.base import ShiftOperator

class MyShiftOperator(ShiftOperator):
    """My custom shift operator."""

    def __init__(self, param1: str = "default"):
        self.param1 = param1

    def apply(self, data, intensity: float):
        """Apply shift at given intensity."""
        # Implement your shift logic
        return shifted_data
```

2. Register in `__init__.py`:
```python
from .my_shift import MyShiftOperator
```

3. Add tests in `tests/test_shifts.py`:
```python
def test_my_shift_operator():
    operator = MyShiftOperator(param1="value")
    result = operator.apply(sample_data, intensity=0.5)
    assert result is not None
```

## Documentation

Build docs locally:
```bash
pip install -e ".[docs]"
mkdocs serve
```

Then visit `http://localhost:8000`.

## Questions?

Open an issue on GitHub or reach out to the maintainers.
