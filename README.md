# Differentiable Particle Filtering

Implementation of particle flow methods for nonlinear filtering, focusing on the Exact Daum-Huang (EDH) and Particle Flow Particle Filter (PF-PF) methods from Li & Coates (2017).

## Features

- **EDH Flow Filter**: Exact Daum-Huang particle flow with global linearization
- **LEDH Flow Filter**: Local Exact Daum-Huang with per-particle linearization
- **PF-PF (EDH)**: Particle Flow Particle Filter using EDH proposal
- **PF-PF (LEDH)**: Particle Flow Particle Filter using LEDH proposal
- **Stochastic Volatility Model**: Implementation for financial time series

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- SciPy

## Quick Start

### Run Main Replication Script

```bash
python examples/replicate_li17.py
```

This will run EDH Flow and PF-PF (EDH) on the Stochastic Volatility model and save results to `results/` folder.

### Visualize SV Model

```bash
python examples/visualize_sv_model.py
```

### Compare Performance

```bash
python examples/compare_perfromance.py
```

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage report:

```bash
pytest --cov=src --cov-report=html
```

See [TESTING.md](TESTING.md) for detailed testing instructions.

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── sv_model.py         # Stochastic Volatility model
│   │   └── base_model.py       # Base state space model
│   └── filters/
│       ├── edh_flow.py         # EDH flow filter
│       ├── ledh_flow.py        # LEDH flow filter
│       ├── pfpf_edh.py         # PF-PF with EDH
│       └── pfpf_ledh.py        # PF-PF with LEDH
├── examples/
│   ├── replicate_li17.py       # Main replication script
│   └── visualize_sv_model.py   # Visualization tools
├── tests/
│   ├── test_models.py          # Model unit tests
│   └── test_integration.py     # Integration tests
└── results/                     # Output folder for results
```

## Key Results

Performance on Stochastic Volatility Model (T=100, N=100 particles):

| Method      | RMSE  | ESS   | Runtime |
|-------------|-------|-------|---------|
| EDH Flow    | 3.43  | N/A   | 7.4s    |
| PF-PF (EDH) | 1.35  | 54.4% | 7.6s    |

- PF-PF (EDH) achieves **61% improvement** over EDH Flow
- Effective Sample Size maintains at **54%** (good particle utilization)

See `LEDH_INVESTIGATION.md` for detailed analysis of LEDH performance.

## References

- Li, Y., & Coates, M. (2017). Particle filtering with invertible particle flow. *arXiv preprint arXiv:1712.08776*.
- Daum, F., & Huang, J. (2010). Exact particle flow for nonlinear filters. *SPIE Defense, Security, and Sensing*.

## License

MIT