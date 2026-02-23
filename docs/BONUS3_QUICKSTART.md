# Bonus Question 3: Quick Start Guide

## Overview

This implementation addresses **Bonus Question 3** on Particle-Flow Inference for Neural State-Space Models, comparing **DPF-HMC** with **Particle Gibbs** on State Space LSTM models from Zheng et al. (2017).

## What's Implemented

### Core Components

1. **State Space LSTM Models** (`src/models/state_space_lstm.py`)
   - `GaussianSSL`: Continuous states and observations (Example 1)
   - `TopicalSSL`: Discrete topics and words (Example 2)

2. **Particle Gibbs Sampler** (`src/inference/particle_gibbs.py`)
   - Conditional particle filter with systematic resampling
   - Joint posterior sampling for trajectories and parameters

3. **DPF-HMC Integration** (uses existing DPF infrastructure)
   - LEDH particle flow with optimal homotopy
   - OT resampling for differentiability
   - HMC with gradient-based proposals

### Experiments

- **Example 1**: Gaussian SSL trajectory tracking (`examples/bonus3_example1_gaussian_ssl.py`)
- **Example 2**: Topical SSL language modeling (`examples/bonus3_example2_topical_ssl.py`)

## Quick Start

### Installation

Ensure you have all dependencies:

```bash
pip install tensorflow numpy matplotlib scipy
```

### Running Experiments

**Quick test** (reduced iterations, ~5 minutes):
```bash
python run_bonus3.py --quick
```

**Full experiments** (~30-40 minutes):
```bash
python run_bonus3.py
```

**Run specific example**:
```bash
# Example 1 only
python run_bonus3.py --example 1

# Example 2 only
python run_bonus3.py --example 2
```

### Manual Execution

You can also run examples individually:

```bash
# Example 1: Gaussian SSL
cd examples
python bonus3_example1_gaussian_ssl.py

# Example 2: Topical SSL
python bonus3_example2_topical_ssl.py
```

## Results

Results are saved to:
- `results/bonus3_example1/`: Example 1 results and plots
- `results/bonus3_example2/`: Example 2 results and plots

Each directory contains:
- `results.npz`: Numerical results (trajectories, metrics, timings)
- `*.png`: Visualization plots

## Key Files

```
differentiable-particle-filtering/
├── BONUS3_REPORT.md                   # Comprehensive report
├── run_bonus3.py                       # Main experiment script
├── BONUS3_QUICKSTART.md               # This file
├── src/
│   ├── models/
│   │   └── state_space_lstm.py        # SSL models
│   └── inference/
│       └── particle_gibbs.py          # PG sampler
└── examples/
    ├── bonus3_example1_gaussian_ssl.py
    └── bonus3_example2_topical_ssl.py
```

## Understanding the Results

### Example 1 Metrics

- **RMSE**: Lower is better (tracking accuracy)
- **ESS**: Higher is better (sampling efficiency)
- **Acceptance Rate**: Optimal around 0.65-0.75 for HMC
- **Time/Iter**: Wall-clock time per iteration

**Expected:** DPF-HMC has better RMSE and ESS but slower per iteration.

### Example 2 Metrics

- **Perplexity**: Lower is better (prediction quality)
- **Topic Accuracy**: Higher is better (topic identification)
- **NNZ**: Lower is better (sparsity)

**Expected:** Particle Gibbs outperforms due to discrete state space challenges.

## Customization

### Adjust Hyperparameters

Edit the experiment scripts to change:

```python
# Example 1
results = run_example1_experiment(
    T=50,                      # Sequence length
    trajectory_type='sine',    # 'sine', 'circle', 'line', 'swiss_roll'
    num_pg_iterations=100,     # PG iterations
    num_hmc_iterations=50,     # HMC iterations
    num_particles_pg=30,       # Particles for PG
    num_particles_hmc=50,      # Particles for DPF-HMC
)

# Example 2
results = run_example2_experiment(
    num_topics=5,              # Number of topics
    vocab_size=30,             # Vocabulary size
    T=100,                     # Document length
    num_pg_iterations=150,     # PG iterations
    num_particles_pg=40,       # Particles for PG
)
```

### Model Architecture

Modify LSTM units in model creation:

```python
model = GaussianSSL(
    state_dim=2,
    obs_dim=2,
    lstm_units=64,    # Increase for more capacity
    min_std=0.01
)
```

### DPF Settings

Adjust resampling and flow parameters:

```python
hmc_sampler = DPF_HMC_Sampler(
    model=model,
    num_particles=100,
    resampling_method='ot',   # 'ot', 'soft', 'gumbel'
    ot_epsilon=0.5            # OT regularization
)
```

## Troubleshooting

### Out of Memory

Reduce number of particles or sequence length:
```bash
python run_bonus3.py --quick
```

### Slow Execution

DPF-HMC is computationally expensive. Options:
1. Use `--quick` flag
2. Reduce `num_hmc_iterations`
3. Reduce `num_particles_hmc`
4. Run only Example 1: `--example 1`

### Import Errors

Ensure you're running from the project root:
```bash
cd differentiable-particle-filtering
python run_bonus3.py
```

### Numerical Instabilities

If you see NaN or Inf values:
1. Reduce HMC step size (default: 0.0001)
2. Increase OT epsilon (default: 0.5)
3. Enable gradient clipping (already implemented)

## Expected Runtime

On a modern CPU (no GPU required):

| Configuration | Example 1 | Example 2 | Total |
|--------------|-----------|-----------|-------|
| **Quick**    | 3-5 min   | 2-3 min   | ~8 min |
| **Full**     | 15-20 min | 10-15 min | ~35 min |

GPU acceleration provides minimal benefit due to small batch sizes.

## Interpretation Guide

### When to Use Particle Gibbs

✅ Discrete state spaces
✅ Long sequences (T > 100)
✅ Fast iteration needed
✅ Simple baseline required

### When to Use DPF-HMC

✅ Continuous high-dimensional states
✅ Short sequences (T < 100)
✅ Gradient information available
✅ High sample quality needed

### Hybrid Approach

For best results, consider:
1. Initialize with Particle Gibbs (fast exploration)
2. Refine with DPF-HMC (local optimization)
3. Use PG for diagnostics (guaranteed acceptance)

## Next Steps

After running experiments:

1. **Review Report**: Read `BONUS3_REPORT.md` for detailed analysis
2. **Check Results**: Examine plots in `results/` directories
3. **Compare Metrics**: Look at RMSE, ESS, perplexity trade-offs
4. **Experiment**: Try different trajectory types or hyperparameters

## Citation

If you use this implementation, please cite:

```bibtex
@misc{dpf_bonus3_2024,
  title={Particle-Flow Inference for Neural State-Space Models},
  author={Implementation of Zheng et al. (2017) with DPF-HMC},
  year={2024}
}
```

## Support

For issues or questions:
1. Check `BONUS3_REPORT.md` for detailed explanations
2. Review code comments in source files
3. Examine existing issues in the repository

## License

This implementation is provided for educational purposes as part of a course assignment on advanced particle filtering methods.
