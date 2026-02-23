# Differentiable Particle Filter with OT Resampling

This note summarizes the implemented DPF variants and the comparison metrics requested.

## Implemented Algorithms

### 1) Soft-Resampling (Chen 2023)
Mixture of normalized weights with uniform to keep gradients through the weight path:

$$\tilde{W}^i = \lambda W^i + (1-\lambda)\frac{1}{N}$$

Resampling uses $\tilde{W}$, then applies weight correction:

$$\tilde{w}^i = \frac{W^i}{\tilde{W}^i}$$

### 2) Entropy-Regularized OT Resampling (Corenflos 2021)
Solve the entropic OT problem between weighted particles and uniform weights using Sinkhorn, then apply the barycentric projection:

$$\tilde{X} = N P^\top X$$

Key knobs:
- $\epsilon$ (regularization): smaller reduces bias but slows convergence and may hurt stability
- Sinkhorn iterations: more iterations improves marginal matching but increases compute

### 3) Gumbel-Softmax Resampling (Relaxed Categorical)
A differentiable continuous relaxation for multinomial resampling. It produces a soft resampling matrix via Gumbel-Softmax and then uses a weighted average of particles.

## Metrics & Diagnostics

**Accuracy**
- RMSE against ground truth state trajectories
- Optional: negative log-likelihood (if available)

**Differentiability**
- Gradient variance of a training loss w.r.t. model parameters (lower is better)
- Sanity check: non-NaN gradients for parameters and particle states

**Efficiency**
- Runtime per step or per sequence
- ESS (effective sample size)

## How to Run the Comparison

See [examples/compare_dpf_resampling.py](examples/compare_dpf_resampling.py). It reports:
- RMSE
- Average ESS
- Runtime
- Gradient variance (w.r.t. $\alpha$ in the SV model)

## Tuning Guidelines

- **Soft-resampling**: $\lambda \in [0.8, 0.95]$ usually balances bias and variance.
- **OT resampling**:
  - $\epsilon \in [0.1, 1.0]$ for stability; smaller for less bias
  - 30–100 Sinkhorn iterations for moderate particle counts
- **Gumbel-Softmax**:
  - Temperature $\tau \in [0.5, 1.0]$ for smoother gradients; smaller is closer to hard resampling
