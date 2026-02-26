# Differentiable Particle Filtering

Implementation of particle flow filters and differentiable particle filters for state-space models, covering the full pipeline from classical Kalman filtering to HMC-based Bayesian parameter inference.

Built with **TensorFlow 2** and **TensorFlow Probability**.

---

## Project Structure

```
differentiable-particle-filtering/
├── src/
│   ├── models/                              # State-space models
│   │   ├── base_model.py                    #   Abstract base class (AutoDiff Jacobians)
│   │   ├── lgssm.py                         #   Linear Gaussian SSM (Doucet 2009)
│   │   ├── sv_model.py                      #   Stochastic Volatility model
│   │   ├── nonlinear_ssm.py                 #   Nonlinear SSM (Andrieu et al. 2010)
│   │   ├── bearing_only_tracking.py         #   Bearing-only tracking (Dai 2022)
│   │   └── state_space_lstm.py              #   State-Space LSTM (Zheng 2017)
│   │
│   ├── filters/                             # Filtering algorithms
│   │   ├── kalman_filter.py                 #   Kalman Filter (Joseph stabilized)
│   │   ├── ekf.py                           #   Extended Kalman Filter
│   │   ├── ukf.py                           #   Unscented Kalman Filter
│   │   ├── particle_filter.py               #   Standard SIR Particle Filter
│   │   ├── edh_flow.py                      #   Exact Daum-Huang flow
│   │   ├── ledh_flow.py                     #   Local EDH flow
│   │   ├── pfpf_edh.py                      #   PF-PF with EDH (Li & Coates 2017)
│   │   ├── pfpf_ledh.py                     #   PF-PF with LEDH (Li & Coates 2017)
│   │   ├── pfpf_enhanced.py                 #   Enhanced PF-PF + optimal homotopy
│   │   ├── flow_pf.py                       #   Invertible flow PF
│   │   ├── particle_flow_filters.py         #   Stochastic particle flow (Dai 2022)
│   │   ├── homotopy_optimizer.py            #   TPBVP solver for optimal homotopy
│   │   ├── homotopy_optimizer_robust.py     #   Robust TPBVP solver (non-convex)
│   │   ├── differentiable_particle_filter.py#   DPF with OT resampling
│   │   ├── differentiable_pfpf.py           #   Differentiable LEDH + OT for HMC
│   │   ├── neural_ot_resampling.py          #   Neural OT: mGradNet, FNO, DeepONet
│   │   └── gradient_diagnostics.py          #   Gradient variance/SNR diagnostics
│   │
│   └── inference/                           # Bayesian parameter inference
│       ├── hmc.py                           #   Hamiltonian Monte Carlo
│       ├── pmmh.py                          #   Particle Marginal MH (Andrieu 2010)
│       ├── particle_gibbs.py                #   Particle Gibbs (Zheng 2017)
│       └── utils.py                         #   ESS estimation (Geyer 1992)
│
├── examples/                                # Runnable scripts organized by topic
│   ├── part1_classical_filters/             #   KF, EKF, UKF, PF comparisons
│   ├── part1_particle_flows/                #   EDH, LEDH, PF-PF (Li 2017)
│   ├── part2_stochastic_flow/               #   Stochastic flow (Dai 2022)
│   ├── part2_differentiable_pf/             #   DPF with OT resampling
│   ├── bonus1_hmc_flows/                    #   HMC + invertible flows + OT
│   ├── bonus2_neural_ot/                    #   Neural OT data/train/benchmark/demo
│   └── bonus3_neural_ssm/                   #   SSL models + Particle Gibbs
│
├── tests/                                   # Unit and integration tests
├── results/                                 # Generated figures and tables
├── docs/                                    # Detailed documentation (19 guides)
├── reports/                                 # LaTeX report and slides
└── archive/                                 # Historical experiment/debug scripts
```

---

## Question Coverage

### Part 1: From Classical Filters to Particle Flows

| Question | Implementation | Example |
|----------|---------------|---------|
| 1.I — Kalman Filter for LGSSM | `src/filters/kalman_filter.py`, `src/models/lgssm.py` | `examples/part1_classical_filters/run_kalman_filter.py` |
| 1.II.a — Nonlinear SSM design | `src/models/sv_model.py`, `src/models/bearing_only_tracking.py` | `examples/part1_classical_filters/visualize_sv_model.py` |
| 1.II.b — EKF and UKF | `src/filters/ekf.py`, `src/filters/ukf.py` | `examples/part1_classical_filters/compare_performance.py` |
| 1.II.c — Standard Particle Filter | `src/filters/particle_filter.py` | `examples/part1_classical_filters/run_particle_filter.py` |
| 1.II.d — PF vs EKF/UKF benchmark | — | `examples/part1_classical_filters/compare_performance.py` |
| 2.a — EDH, LEDH, PF-PF (Li 2017) | `src/filters/edh_flow.py`, `ledh_flow.py`, `pfpf_*.py` | `examples/part1_particle_flows/replicate_li17.py` |
| 2.c — Flow comparison on SV model | — | `examples/part1_particle_flows/compare_flow.py` |

### Part 2: Stochastic Particle Flow and Differentiable PF

| Question | Implementation | Example |
|----------|---------------|---------|
| 1.a — Stochastic flow (Dai 2022) | `src/filters/particle_flow_filters.py`, `homotopy_optimizer*.py` | `examples/part2_stochastic_flow/replicate_dai22.py` |
| 1.b — Optimal flow as PF-PF proposal | `src/filters/pfpf_enhanced.py` | `examples/part2_stochastic_flow/replicate_dai22_robust.py` |
| 2.i — DPF with soft + OT resampling | `src/filters/differentiable_particle_filter.py` | `examples/part2_differentiable_pf/compare_dpf_resampling.py` |

### Bonus Questions

| Question | Implementation | Example |
|----------|---------------|---------|
| Bonus 1 — HMC + invertible flows | `src/filters/differentiable_pfpf.py`, `src/inference/hmc.py` | `examples/bonus1_hmc_flows/bonus1_hmc_invertible_flows.py` |
| Bonus 2 — Neural OT acceleration | `src/filters/neural_ot_resampling.py` | `examples/bonus2_neural_ot/` |
| Bonus 3 — SSL + Particle Gibbs | `src/models/state_space_lstm.py`, `src/inference/particle_gibbs.py` | `examples/bonus3_neural_ssm/bonus3_example*.py` |

---

## Key Features

### Models (6)
- **Linear Gaussian SSM** — standard benchmark with analytic KF solution
- **Stochastic Volatility** — financial time series with state-dependent noise
- **Nonlinear SSM** — Andrieu et al. (2010) strongly nonlinear dynamics
- **Bearing-only Tracking** — 2D position inference from angle-only sensors (Dai 2022)
- **Gaussian SSL** — State-Space LSTM with continuous Gaussian latent states (Zheng 2017)
- **Topical SSL** — State-Space LSTM with discrete topic/word and Gumbel-Softmax relaxation

### Filters (18 modules)
- **Classical**: Kalman Filter, EKF (AutoDiff Jacobians), UKF (sigma-point)
- **Particle**: SIR bootstrap PF with adaptive resampling
- **Particle Flow**: EDH, LEDH, PF-PF with EDH/LEDH, enhanced PF-PF with optimal homotopy
- **Stochastic Flow**: Dai (2022) SDE-based flow with drift clipping and regularization
- **Homotopy Optimization**: TPBVP solver + robust variant (ridge, gradient clipping, continuation)
- **Differentiable PF**: soft, Sinkhorn OT, and Gumbel-Softmax resampling (Corenflos 2021)
- **Differentiable PF-PF**: LEDH flow + OT resampling, fully differentiable w.r.t. model parameters
- **Neural OT**: mGradNet, Fourier Neural Operator, DeepONet architectures for accelerated OT
- **Gradient Diagnostics**: variance, SNR, condition number, Lipschitz estimation

### Inference (3 algorithms)
- **HMC** — Hamiltonian Monte Carlo with leapfrog integrator and mass matrix support
- **PMMH** — Particle Marginal Metropolis-Hastings (Andrieu et al. 2010)
- **Particle Gibbs** — conditional PF with reference trajectory preservation

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: `numpy`, `matplotlib`, `tensorflow==2.16.2`, `tensorflow-probability==0.24.0`, `tf_keras`, `scipy`, `pytest`

### Quick Examples

```bash
# Part 1: Classical filters on LGSSM
python examples/part1_classical_filters/run_kalman_filter.py

# Part 1: Particle flow filters (Li & Coates 2017)
python examples/part1_particle_flows/replicate_li17.py

# Part 2: Stochastic particle flow (Dai 2022)
python examples/part2_stochastic_flow/replicate_dai22_robust.py

# Part 2: Differentiable PF with OT resampling
python examples/part2_differentiable_pf/compare_dpf_resampling.py

# Bonus 1: HMC with invertible flows
python examples/bonus1_hmc_flows/bonus1_hmc_invertible_flows.py

# Bonus 2: Neural OT acceleration
python examples/bonus2_neural_ot/demo.py

# Bonus 3: SSL models with Particle Gibbs
python examples/bonus3_neural_ssm/bonus3_example1_gaussian_ssl.py
```

### Running Tests

```bash
# All tests
pytest

# By category
pytest -m unit
pytest -m integration
pytest -m gradient
pytest -m convergence
```

---

## Documentation

Detailed guides are available in [docs/](docs/):

| Guide | Description |
|-------|-------------|
| [PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) | Complete project overview |
| [TECHNICAL_REFERENCE.md](docs/TECHNICAL_REFERENCE.md) | Algorithm details and equations |
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Dai22 vs Li17 decision guide |
| [ROBUST_HOMOTOPY_GUIDE.md](docs/ROBUST_HOMOTOPY_GUIDE.md) | Robust homotopy optimizer |
| [DPF_OT_RESAMPLING.md](docs/DPF_OT_RESAMPLING.md) | Differentiable PF with OT |
| [BONUS1_HMC_FLOWS_README.md](docs/BONUS1_HMC_FLOWS_README.md) | HMC + invertible flows |
| [BONUS2_NEURAL_OT_ACCELERATION.md](docs/BONUS2_NEURAL_OT_ACCELERATION.md) | Neural OT: mGradNet, FNO, DeepONet |
| [BONUS3_README.md](docs/BONUS3_README.md) | DPF-HMC vs Particle Gibbs |

---

## Key References

- [Doucet (2009)] — A tutorial on particle filtering and smoothing
- [Daum (2010, 2011)] — Exact particle flow for nonlinear filters
- [Li & Coates (2017)] — Particle filtering with invertible particle flow
- [Hu (2021)] — Kernel-embedded particle flow filter in RKHS
- [Dai (2021, 2022)] — Stochastic particle flow and stiffness mitigation
- [Corenflos (2021)] — Differentiable PF via entropy-regularized OT
- [Chen (2023)] — Overview of differentiable particle filters
- [Andrieu et al. (2010)] — Particle MCMC methods
- [Zheng (2017)] — State-space LSTM with particle MCMC inference
- [Chaudhari et al. (2025)] — GradNetOT: Learning optimal transport maps with GradNets
- [Jha (2025)] — Neural operators in scientific computing
