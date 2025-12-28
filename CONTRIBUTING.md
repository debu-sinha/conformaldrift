# Contributing to ConformalDrift

Thank you for your interest in contributing to ConformalDrift!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/debu-sinha/conformaldrift.git
cd conformaldrift
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Run tests:
```bash
pytest
```

## Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** throughout

Before submitting a PR:
```bash
black conformal_drift tests
ruff check conformal_drift tests
pytest
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## Adding New Shift Types

To add a new shift type:

1. Add shift simulation in `conformal_drift/shifts.py`
2. Add corresponding tests in `tests/test_shifts.py`
3. Update README with examples
4. Add preset configuration to `SHIFT_PRESETS`

## Questions?

Open an issue or reach out to [debusinha2009@gmail.com](mailto:debusinha2009@gmail.com).
