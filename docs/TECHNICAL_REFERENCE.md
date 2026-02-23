# Differentiable Particle Filtering: A Technical Reference

This document is a self-contained reference for the theory, algorithms, and implementation decisions in this library. It targets a reader with graduate-level statistics who is new to state-space models and particle-flow methods.

---

## 1. Motivation and Problem Setting

### 1.1 State-Space Models

A **state-space model (SSM)** describes a system with a hidden state **x**_t that evolves over time and produces noisy observations **y**_t:

- **Transition model**: x_t = f(x_{t-1}) + v_t, where v_t ~ N(0, Q)
- **Observation model**: y_t = h(x_t) + w_t, where w_t ~ N(0, R)

The **filtering problem** is: given observations y_{1:T}, estimate the posterior distribution p(x_t | y_{1:t}) at each time step t.

### 1.2 Why Filtering Is Hard

For linear Gaussian SSMs, the Kalman filter gives the exact posterior in closed form. For nonlinear or non-Gaussian models, the posterior is intractable because the required integrals have no analytical solution. This forces us to use approximate methods, each with different trade-offs in accuracy, computational cost, and differentiability.

---

## 2. Classical Approaches

### 2.1 Kalman Filter

For linear Gaussian systems (f(x) = Fx, h(x) = Hx), the posterior is Gaussian and can be updated in two steps:

- **Predict**: x_{t|t-1} = F x_{t-1|t-1}, P_{t|t-1} = F P_{t-1|t-1} F' + Q
- **Update**: K_t = P_{t|t-1} H' (H P_{t|t-1} H' + R)^{-1}, then x_{t|t} = x_{t|t-1} + K_t (y_t - H x_{t|t-1})

**Implementation decision**: We use the Joseph-form covariance update P_{t|t} = (I - K H) P_{t|t-1} (I - K H)' + K R K' instead of the standard form. This guarantees the covariance remains symmetric positive semi-definite even with finite-precision arithmetic.

See: `src/filters/kalman_filter.py`

### 2.2 Extended and Unscented Kalman Filters

When f or h is nonlinear, linearization provides approximate Gaussian posteriors:

- **EKF**: Linearizes via Jacobians F_t = df/dx|_{x_{t-1}}, H_t = dh/dx|_{x_t}. Our implementation uses `tf.GradientTape()` for automatic Jacobian computation, eliminating the need for model-specific derivative code.
- **UKF**: Uses sigma points to propagate means and covariances through the nonlinearity without explicit Jacobians. More accurate than EKF for highly nonlinear h(x), e.g., h(x) = x^2/20 in the Andrieu (2010) benchmark.

See: `src/filters/ekf.py`, `src/filters/ukf.py`

---

## 3. Particle Filters

### 3.1 Sequential Importance Resampling (SIR)

When linearization fails (multi-modal posteriors, heavy tails), particle filters represent the posterior as a weighted set of samples {x_t^{(i)}, w_t^{(i)}}:

1. **Predict**: Propagate each particle through the transition: x_t^{(i)} ~ f(x_{t-1}^{(i)}, v_t)
2. **Weight**: Compute importance weights: w_t^{(i)} proportional to p(y_t | x_t^{(i)})
3. **Resample**: Draw N particles with replacement according to weights

### 3.2 Weight Degeneracy and ESS

After several time steps, most weight concentrates on a few particles. The **Effective Sample Size (ESS)** measures this:

ESS = 1 / sum_i (w_t^{(i)})^2

When ESS is close to N, particles are well-distributed; when ESS approaches 1, the approximation has collapsed. Our filters resample when ESS drops below a threshold (typically 0.5N).

### 3.3 The Resampling Bottleneck

Standard resampling draws discrete indices from a categorical distribution. This is non-differentiable: the gradient of an argmax over discrete indices is zero almost everywhere. This blocks gradient-based parameter learning and motivates the differentiable methods in Section 6.

See: `src/filters/particle_filter.py`

---

## 4. Particle Flow Filters (Li & Coates 2017)

### 4.1 The Core Idea

Instead of resampling, transport particles from the prior to the posterior via a continuous deterministic flow. Introduce a homotopy parameter lambda in [0, 1]:

log p_lambda(x) = (1-lambda) log p_0(x) + lambda log p(y|x) + const

At lambda=0, particles represent the prior; at lambda=1, the posterior. The flow dx/dlambda is designed so that the particle distribution tracks p_lambda.

### 4.2 Exact Daum-Huang (EDH) Flow

The EDH flow uses ensemble-level statistics (mean and covariance) to compute the flow parameters A(lambda) and b(lambda):

dx/dlambda = A(lambda) x + b(lambda)

where A and b are determined from the ensemble covariance P and the observation Jacobian H. All particles share the same A, b — a global approximation.

### 4.3 Local EDH (LEDH) Flow

LEDH computes per-particle flow parameters using local linearization of h(x) at each particle. This is more flexible but more expensive (O(N) Jacobian computations per flow step).

### 4.4 Particle Flow Particle Filter (PF-PF)

Use the flow as an improved proposal distribution, then correct with importance weights accounting for the Jacobian determinant of the flow map:

w_i proportional to p(y|x_1^i) |det J_T^i| p(x_1^i|x_prev) / p(x_0^i|x_prev)

The Jacobian determinant term |det J| accounts for volume change under the flow. For 1D systems, det(I + epsilon*A) = 1 + epsilon*A.

**Implementation decisions**:
- Step size epsilon and number of flow steps K are hyperparameters. We use epsilon = 1/K for stability.
- Particle values are clipped to [-10, 10] to prevent numerical overflow.

See: `src/filters/edh_flow.py`, `src/filters/ledh_flow.py`, `src/filters/pfpf_edh.py`, `src/filters/pfpf_ledh.py`

---

## 5. Stochastic Particle Flow (Dai 2022)

### 5.1 From ODE to SDE

Dai (2022) extends the deterministic flow to a stochastic differential equation:

dx = f(x, lambda) dlambda + q(x, lambda) dw_lambda

Adding diffusion (the q term) provides robustness against model misspecification and linearization errors. The drift f is computed from the gradient of log p_lambda(x).

### 5.2 Stiffness and Optimal Homotopy

The matrix M(lambda) = -nabla^2 log p_lambda(x) can become ill-conditioned (large eigenvalue ratio), causing numerical stiffness. The homotopy schedule beta*(lambda) is optimized to minimize the condition number kappa(M) by solving a two-point boundary value problem:

minimize integral_0^1 [0.5 u^2 + mu kappa(M(lambda))] dlambda

We use the nuclear norm kappa*(M) = tr(M) tr(M^{-1}) (more stable than the spectral norm) with analytical derivatives d_kappa/d_beta.

**Key finding**: Stiffness reduction benefit is non-monotonic; maximum improvement occurs for moderately stiff problems.

See: `src/filters/particle_flow_filters.py`, `src/filters/homotopy_optimizer.py`, `src/filters/homotopy_optimizer_robust.py`

---

## 6. Differentiable Particle Filters

### 6.1 The Gradient Flow Problem

Standard resampling is non-differentiable: selecting particles by discrete indices produces zero gradients. Three strategies restore gradient flow:

### 6.2 Soft Resampling

Mix the empirical weights with a uniform distribution:

w_mix = alpha * w + (1-alpha) / N

Then resample from w_mix and correct with importance weights w/w_mix. The mixing parameter alpha controls the bias-variance trade-off: alpha=1 recovers standard PF (no gradient), alpha=0 gives uniform sampling (no learning signal).

### 6.3 Optimal Transport Resampling (Corenflos et al. 2021)

Compute an entropy-regularized optimal transport plan T between the weighted particles and a uniform target:

minimize sum_{ij} T_{ij} C_{ij} - epsilon H(T)

subject to T 1 = w, T' 1 = 1/N

where C_{ij} = ||x_i - x_j||^2 is the cost matrix and H(T) is the entropy. The Sinkhorn algorithm iteratively computes T in O(K N^2) operations. New particles are the barycentric projection: x_new_j = N * sum_i T_{ij} x_i.

**Trade-off**: epsilon controls regularization. Small epsilon approximates true OT but can cause vanishing gradients; large epsilon gives smoother gradients but blurrier particle positions.

### 6.4 Gumbel-Softmax Resampling

Apply the Gumbel-Softmax reparameterization trick:

s_j = softmax((log w + g_j) / tau)

where g_j are i.i.d. Gumbel(0,1) samples and tau is the temperature. New particles are x_new_j = sum_i s_{ji} x_i. Low tau approaches hard (accurate) sampling; high tau gives smooth gradients.

### 6.5 Gradient Quality Criteria

A differentiable PF produces gradients suitable for HMC when:
1. **Gradient SNR > 1**: E[grad]^2 / Var[grad] > 1 (signal dominates noise)
2. **Gradient agreement > 0.5**: cosine similarity between analytic and finite-difference gradients
3. **Condition number < 10^4**: Hessian of the loss is not severely ill-conditioned

Use `src/filters/gradient_diagnostics.py` to compute these metrics. Expected behavior:
- OT resampling gives the lowest gradient variance (deterministic transport).
- Soft resampling gives the best gradient agreement with finite differences.
- Gumbel-Softmax allows explicit temperature scheduling during training.

See: `src/filters/differentiable_particle_filter.py`, `src/filters/gradient_diagnostics.py`

---

## 7. Bayesian Parameter Inference

### 7.1 HMC with Differentiable PF

Hamiltonian Monte Carlo uses the gradient of the log posterior to guide proposals:

log p(theta | y_{1:T}) proportional to log p(y_{1:T} | theta) + log p(theta)

With a differentiable PF, the log marginal likelihood log p(y_{1:T} | theta) and its gradient w.r.t. theta are available via automatic differentiation. The leapfrog integrator simulates Hamiltonian dynamics for efficient exploration of the parameter space.

**Challenge**: The PF's stochastic internals add noise to the gradient. Higher N reduces this noise (gradient variance is O(1/N)), but increases cost.

See: `src/inference/hmc.py`

### 7.2 Particle MCMC (PMMH, Particle Gibbs)

For models where differentiable PF gradients are unreliable:
- **PMMH** (Andrieu et al. 2010): Use the PF's likelihood estimate in a Metropolis-Hastings acceptance step. No gradients needed.
- **Particle Gibbs** (Zheng 2017): Conditional SMC for jointly sampling state trajectories and parameters.

See: `src/inference/pmmh.py`, `src/inference/particle_gibbs.py`

---

## 8. Models in This Library

| Model | Dynamics | Observation | Reference |
|-------|----------|-------------|-----------|
| LGSSM | x_t = F x_{t-1} + q_t | y_t = H x_t + r_t | Doucet (2009) |
| SV | x_t = alpha x_{t-1} + sigma v_t | y_t ~ N(0, beta^2 exp(x_t)) | Doucet (2009) |
| Nonlinear | x_t = x/2 + 25x/(1+x^2) + 8cos(1.2t) + v_t | y_t = x_t^2/20 + w_t | Andrieu (2010) |
| Bearing-only | x_t = x_{t-1} (static) | z_t = atan2(x - s_k) | Dai (2022) |
| SS-LSTM | LSTM cell dynamics | Learned emission | Zheng (2017) |

See: `src/models/`

---

## 9. Test Design Philosophy

Tests are designed from the underlying mathematics, not as a checklist:

- **Unit tests** verify mathematical necessary conditions (e.g., log-likelihood matches analytic Gaussian density, ESS formula at known weight vectors, stationary distribution moments).
- **Integration tests** verify complete pipeline behavior (e.g., PF RMSE decreases with N, KF is optimal for linear systems).
- **Gradient tests** verify that differentiable resampling produces usable gradients (existence, SNR, variance reduction with N).
- **Convergence tests** verify asymptotic properties (PF consistency, DPF-PF equivalence).

See: `tests/`

---

## References

1. Doucet, A. & Johansen, A. (2009). A tutorial on particle filtering and smoothing.
2. Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov chain Monte Carlo methods. JRSS-B.
3. Daum, F. & Huang, J. (2010). Exact particle flow for nonlinear filters.
4. Li, C. & Coates, M. (2017). Particle filtering with invertible particle flow.
5. Dai, D. (2021, 2022). Stochastic particle flow and stiffness mitigation.
6. Corenflos, A. et al. (2021). Differentiable particle filtering via entropy-regularized optimal transport. AISTATS.
7. Neal, R. (2011). MCMC using Hamiltonian dynamics. Handbook of MCMC.
8. Zheng, Y. et al. (2017). State-space LSTM models with particle MCMC inference.
9. Chen, L. et al. (2023). An overview of differentiable particle filters.
