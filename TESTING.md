# Testing Guide

## Overview

This project contains **22 automated tests** covering models, filters, and integration tests, with **32% code coverage** overall (core filter modules reach **98-100%**).

## Installation

Install testing dependencies:

```bash
pip install pytest pytest-cov
```

## Running Tests

### Run All Tests
```bash
pytest
```

Expected output:
```
======================= 22 passed, 4 warnings in 23.33s ========================
```

### Run Specific Test Files
```bash
pytest tests/test_models.py
pytest tests/test_integration.py
```

### Run Specific Test Class
```bash
pytest tests/test_models.py::TestSVModel
```

### Run Specific Test
```bash
pytest tests/test_models.py::TestSVModel::test_model_initialization
```

## Coverage Reports

Generate HTML coverage report:

```bash
pytest --cov=src --cov-report=html
```

This generates an HTML report in the `htmlcov/` directory. Open `htmlcov/index.html` to view detailed coverage.

### Current Coverage

| Module | Coverage | Status |
|--------|----------|--------|
| `src/filters/edh_flow.py` | 100% | ✅ Fully covered |
| `src/filters/pfpf_edh.py` | 100% | ✅ Fully covered |
| `src/filters/pfpf_ledh.py` | 100% | ✅ Fully covered |
| `src/filters/ledh_flow.py` | 98% | ✅ Nearly complete |
| `src/models/sv_model.py` | 94% | ✅ Core functionality covered |

## Test Structure

### 1. **test_models.py** - Model Unit Tests (17 tests)
- **TestSVModel**: Validates SV model functionality
  - ✅ Parameter initialization correctness
  - ✅ State transition shapes and values
  - ✅ Observation function output
  - ✅ Log-likelihood computation and numerical properties
  - ✅ Gradient computability
  
- **TestFilterShapes**: Validates filter outputs
  - ✅ EDH filter complete run
  - ✅ LEDH filter complete run
  
- **TestNumericalStability**: Validates numerical stability
  - ✅ PF-PF (EDH) returns correct format

### 2. **test_integration.py** - Integration Tests (7 tests)
- **TestCompleteFiltering**: Validates complete filtering pipeline
  - ✅ EDH filter runs on full time series
  - ✅ LEDH filter runs on full time series
  - ✅ PF-PF (EDH) returns estimates and ESS
  - ✅ PF-PF (LEDH) returns estimates and ESS
  
- **TestComparativePerformance**: Validates relative performance
  - ✅ PF-PF (EDH) outperforms pure EDH flow
  
- **TestEffectiveSampleSize**: Validates ESS computation
  - ✅ ESS within [1, N] range (PF-PF EDH)
  - ✅ ESS within [1, N] range (PF-PF LEDH)

## Test Coverage

✅ **Models**
- Parameter initialization
- State transitions
- Observation model
- Log-likelihood
- Gradient computation
- Numerical stability

✅ **Filters**
- Output format correctness
- No numerical exceptions (NaN/Inf)
- Complete time series execution
- ESS validity

✅ **Numerical Properties**
- ESS computation correctness
- Performance comparison validity

## Continuous Integration

Add a pre-commit hook to your Git workflow:

```bash
# In .git/hooks/pre-commit
#!/bin/bash
pytest --tb=short
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

## Debugging Test Failures

### View Detailed Error Information
```bash
pytest -v --tb=long
```

### Show Only Failed Tests
```bash
pytest --lf  # last-failed
```

### Stop at First Failure
```bash
pytest -x
```

## Extending Tests

Suggested additions for comprehensive testing:

1. **Performance Tests**: Runtime with different particle counts and time steps
2. **Numerical Accuracy Tests**: Comparison with reference implementations
3. **Edge Cases**: Extreme parameter values
4. **Memory Usage**: Memory footprint for large-scale problems

## Test Data

Tests use fixtures defined in `conftest.py`:
- `sv_model`: SV model instance
- `synthetic_data`: Synthetic test data (T=50, N=100)
- `particles_and_weights`: Initial particles and weights

All tests use a fixed random seed (42) to ensure reproducibility.
