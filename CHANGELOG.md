# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-28

### Added
- Initial release of conformal-drift package
- `ConformalDriftAuditor` class for auditing conformal prediction systems
- Coverage metrics: effective coverage, coverage drop, FPR at nominal coverage
- Shift simulation utilities: RLHF shift, temporal shift, domain shift
- PASS/MARGINAL/FAIL verdict system
- CLI interface: `conformal-drift demo` and `conformal-drift audit`
- Comprehensive documentation and practitioner tutorials

### Features
- Three-phase audit protocol (Calibrate, Shift, Measure)
- Multi-shift testing support
- Recalibration analysis
- Integration examples for MLOps pipelines
