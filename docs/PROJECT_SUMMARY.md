# Project Summary: Particle Flow Filter and Differentiable Particle Filter

## 1. Project Overview

This project addresses JP Morgan MLCOE TSRL 2026 Internship Question 2, progressing from classical Kalman filtering to differentiable particle filters. The work spans **6 model implementations**, **18 filter/inference modules**, and **16 runnable experiments**, totaling ~6,000+ lines of code built on TensorFlow 2.

**Core question:** How do we perform efficient sequential Bayesian inference for nonlinear, non-Gaussian state-space models — and how can we make particle filters differentiable for end-to-end learning?

---

## 2. Architecture at a Glance

```
src/
├── models/           6 classes, ~1,200 LOC
│   ├── base_model.py           Abstract SSM interface (AutoDiff Jacobians)
│   ├── lgssm.py                Linear Gaussian SSM (Doucet 2009, Ex. 2)
│   ├── sv_model.py             Stochastic Volatility (Doucet 2009, Ex. 4)
│   ├── nonlinear_ssm.py        Andrieu et al. (2010) Sec 3.1
│   ├── bearing_only_tracking.py  2D bearing-only (Dai 2022)
│   └── state_space_lstm.py     GaussianSSL + TopicalSSL (Zheng 2017)
│
├── filters/          18 classes, ~4,150 LOC
│   ├── kalman_filter.py        KF with Joseph stabilized update
│   ├── ekf.py                  EKF with AutoDiff Jacobians
│   ├── ukf.py                  UKF with sigma points
│   ├── particle_filter.py      Bootstrap SIR PF
│   ├── flow_pf.py              Invertible flow PF (log-squared trick)
│   ├── edh_flow.py             Exact Daum-Huang flow
│   ├── ledh_flow.py            Localized EDH flow
│   ├── pfpf_edh.py             PF-PF with EDH (Li 2017)
│   ├── pfpf_ledh.py            PF-PF with LEDH (Li 2017)
│   ├── pfpf_enhanced.py        PF-PF + Dai22 optimal homotopy
│   ├── particle_flow_filters.py  Stochastic flow (Dai 2021/2022)
│   ├── homotopy_optimizer.py   TPBVP solver for optimal beta
│   ├── homotopy_optimizer_robust.py  Robust solver (non-convex)
│   ├── differentiable_particle_filter.py  DPF (soft/OT/Gumbel)
│   ├── differentiable_pfpf.py  DPF-PF with OT resampling
│   ├── neural_ot_resampling.py mGradNet + FNO + DeepONet
│   └── gradient_diagnostics.py Gradient quality metrics
│
└── inference/        3 classes, ~850 LOC
    ├── hmc.py                  Hamiltonian Monte Carlo
    ├── pmmh.py                 Particle Marginal MH
    └── particle_gibbs.py       Particle Gibbs (conditional PF)
```

---

## 3. Part-by-Part Approach and Key Ideas

### Part 1.1: Kalman Filter (Warm-up)

**Problem:** Implement KF for a multidimensional LGSSM and analyze stability.

**Approach:**
- Implemented standard KF recursion under TensorFlow
- **Key insight:** The standard covariance update `P = (I - KH)P` is numerically fragile. We use the **Joseph form**: `P = (I-KH)P(I-KH)' + KRK'`, which is the sum of two PSD matrices, structurally guaranteeing positive definiteness
- Monitor **condition number** `kappa(P)` at each step as a health diagnostic

**Implementation:** `KalmanFilter` class (106 LOC), takes a `LinearGaussianSSM` model

---

### Part 1.2-1.5: Nonlinear Filtering (EKF, UKF, PF)

**Problem:** Design a nonlinear SSM, implement EKF/UKF/PF, compare performance.

**Approach:**
- Chose the **Stochastic Volatility model** (Doucet Ex. 4): `x_t = 0.91*x_{t-1} + noise`, `y_t = 0.5*exp(x_t/2)*noise`. The observation variance depends nonlinearly on the state — a hard problem
- **EKF:** Linearize via Taylor expansion. Key trick: compute Jacobians automatically via `tf.GradientTape` instead of manual derivation. Limitation: fails when `|x|` is large because `exp(x/2)` has huge higher-order terms
- **UKF:** Use 2n+1 sigma points to capture mean/covariance through the nonlinearity. Better than EKF for moderate nonlinearity, but sigma point collapse occurs at extreme states
- **PF:** Bootstrap SIR with multinomial resampling. Monitors ESS and resamples when ESS < N/2. Works well but is non-differentiable

**Key finding:** PF achieves lowest RMSE (1.21) vs UKF (1.64) vs EKF (1.82), but at 18x the runtime of EKF.

---

### Part 1.6-1.7: Particle Flows (EDH, LEDH, PF-PF)

**Problem:** Implement deterministic particle flows and replicate Li & Coates (2017).

**Approach and thinking process:**

1. **EDH Flow** (Daum 2010): All particles share the same flow equation `dx/dlambda = A*x + b`. The matrices A, b are derived from the homotopy between prior and posterior. Simple but uses ensemble-mean linearization — doesn't adapt to individual particles.

2. **LEDH Flow** (Daum 2011): Each particle gets its own A, b computed from its local linearization. Much better for problems where the observation function varies across the state space (like SV model where `h(x) = beta*exp(x/2)`).

3. **PF-PF Framework** (Li 2017): The key innovation — use the flow as a **proposal distribution** within importance sampling. The flow map is invertible, so we can correct weights via the Jacobian determinant:
   ```
   w ∝ p(y|x_1) * |det(J)|
   ```
   **Critical insight for EDH:** The Jacobian determinant cancels out! This simplifies implementation significantly. For LEDH, we must compute `log|1 + epsilon*A|` at each step.

**Result:** PF-PF(EDH) achieves RMSE 1.35 vs standalone EDH's 3.43 — the importance weighting fixes the flow's approximation errors.

---

### Part 2.1-2.2: Stochastic Particle Flow (Dai 2022)

**Problem:** Replicate Dai22's stiffness mitigation; test if optimal homotopy improves Li17.

**Approach and key challenges:**

1. **TPBVP Solver:** Dai22 solves a two-point boundary value problem to find the optimal beta schedule that minimizes flow stiffness. The BVP is:
   ```
   d^2 beta / d lambda^2 = mu * d kappa(M) / d beta
   ```
   with `beta(0)=0, beta(1)=1`.

2. **Challenge on bearing-only tracking:** The Hessian is **indefinite** (not PSD), violating Dai22's convexity assumption. Standard BVP solvers fail completely.

3. **Solution:** We built a **RobustHomotopyOptimizer** with:
   - Ridge regularization to handle non-PSD Hessians
   - Gradient clipping to prevent BVP divergence
   - Continuation method (start with small mu, gradually increase)
   - Fallback to linear homotopy if all else fails

4. **Pragmatic alternative:** For the actual flow, we developed a **scheduled mixed density** approach that gradually transitions from prior to likelihood, achieving 78.6% MSE improvement.

5. **Cross-method integration (Dai22 + Li17):** We tested optimal beta in the PF-PF framework:
   - **LEDH: No benefit** (-0.05%). Per-particle localization already provides local adaptation.
   - **EDH: Modest benefit** (+2.9%). Only works for "soft" problems (loose priors).
   - **Root cause:** Distribution shift during flow integration invalidates pre-computed beta.

**Key insight:** Mathematical optimization doesn't always translate to practical improvement. The assumptions underlying one method (Dai22) can be violated by the dynamics of another (Li17 flow).

---

### Part 2.3-2.5: Differentiable Particle Filters

**Problem:** Make PF differentiable for gradient-based learning. Compare approaches.

**Approach:**

The fundamental problem: **resampling draws discrete indices** from Categorical(w), which has zero gradient everywhere. Three solutions:

1. **Soft resampling:** Mix weights with uniform: `w_mix = alpha*w + (1-alpha)/N`. Simple but high gradient variance.

2. **Gumbel-Softmax:** Replace `argmax` with `softmax((log w + gumbel noise) / tau)`. Temperature tau controls approximation quality.

3. **OT resampling** (Corenflos 2021): Solve entropy-regularized optimal transport:
   ```
   min <C, P> - epsilon * H(P)
   ```
   via **Sinkhorn algorithm** in log-domain. The transport plan P is fully differentiable. New particles are convex combinations: `x_new = N * P' * x_old`.

**Implementation:** `DifferentiableParticleFilter` class supports all three methods via a `resampling_method` parameter. The OT variant uses 50-100 Sinkhorn iterations.

**Key finding:** OT resampling achieves the best RMSE (1.19) and lowest gradient variance, making it the only viable option for HMC-based inference.

---

### Bonus 1: HMC with Invertible Flows

**Problem:** Apply HMC to parameter inference using differentiable PF.

**Approach:**

1. **Model:** Andrieu (2010) nonlinear SSM — time-varying state with quadratic observation. Infer `sigma_V` and `sigma_W`.

2. **PMMH baseline:** Standard particle filter provides log-likelihood estimate. Random-walk Metropolis proposals. Simple but inefficient (acceptance rate 25-30%).

3. **HMC:** Use differentiable PF-PF (LEDH + OT resampling) to compute `grad_theta log p(y | theta)`. Leapfrog integration for directed proposals.

**Key result:** HMC achieves 3x higher ESS per wallclock second than PMMH. The gradient information transforms random-walk exploration into directed sampling.

**Trade-off:** HMC requires a fully differentiable pipeline — hence the need for OT resampling instead of standard multinomial resampling.

---

### Bonus 2: Neural Acceleration of OT Resampling

**Problem:** Sinkhorn is expensive (30-100 iterations per step). Can neural networks replace it?

**Approach (theoretical + prototype implementation):**

1. **mGradNet** (Chaudhari 2025): By Brenier's theorem, the OT map is the gradient of a convex function. A monotone gradient network directly parameterizes this map. Key design: **condition the network on all problem variables** (particles, weights, model params, observation) so a single trained network generalizes without retraining.

2. **FNO** (Jha 2025): The Sinkhorn algorithm solves a PDE (heat equation for entropic OT). Fourier Neural Operators learn the solution operator in frequency domain with O(N log N) complexity and **discretization invariance** (train on N=100, test on N=500).

3. **Implementation:** `neural_ot_resampling.py` (527 LOC) contains `OTResamplingNetwork`, `FourierOTOperator`, and `DeepONetOT` classes. Training pipeline: collect Sinkhorn solutions, train with plan matching + marginal constraints + Monge-Ampere residual loss.

**Expected impact:** 10-100x speedup, making real-time DPF practical.

---

### Bonus 3: Neural State-Space Models

**Problem:** Compare DPF-HMC with Particle Gibbs on State-Space LSTM models.

**Approach:**

1. **State-Space LSTM** (Zheng 2017): LSTM dynamics + probabilistic emissions. Two examples:
   - **Gaussian SSL:** Continuous states, Gaussian everything → natural fit for DPF-HMC
   - **Topical SSL:** Discrete topic indicators → requires Gumbel-Softmax relaxation

2. **Particle Gibbs:** Run conditional particle filter with reference trajectory. Always accepts (by construction). No gradients needed.

3. **DPF-HMC:** Our full pipeline — LEDH flow + OT resampling + HMC leapfrog.

**Key finding:** DPF-HMC wins on continuous states (lower RMSE, higher ESS) but loses decisively on discrete states where Gumbel-Softmax introduces significant bias.

4. **3-month roadmap:**
   - Month 1: Neural OT for 10-50x resampling speedup
   - Month 2: Variance reduction (control variates, truncated BPTT) for long sequences
   - Month 3: Hybrid PG-HMC (alternate global exploration with local refinement)

---

## 4. Technical Highlights and Design Decisions

### 4.1 Automatic Differentiation Throughout

Every model inherits from `StateSpaceModel` which provides `transition_jacobian(x)` and `observation_jacobian(x)` via `tf.GradientTape`. This means:
- EKF doesn't need manual Jacobian derivation
- Differentiable PF gets gradients "for free"
- HMC gets exact gradients through the entire filtering pipeline

### 4.2 Numerical Stability Techniques

| Technique | Where Used | Why |
|---|---|---|
| Joseph covariance update | KF, EKF | Preserve PSD of covariance |
| Log-domain weight computation | All PFs | Prevent underflow for many particles |
| Log-sum-exp normalization | PF weight update | Numerically stable softmax |
| Sinkhorn in log-domain | OT resampling | Prevent overflow of transport plan |
| Ridge regularization | Robust homotopy | Handle non-PSD Hessians |
| Gradient clipping | DPF, HMC | Prevent gradient explosions |
| Mixed precision (f32/f64) | DPF-HMC | Particles in f32, likelihoods in f64 |

### 4.3 Key Algorithmic Tricks

1. **Log-squared observation trick** (`flow_pf.py`): For SV model, transform `y -> log(y^2)`. This makes the observation function `log(beta^2) + x` — **linear!** So the gradient H = 1, dramatically simplifying the flow.

2. **Jacobian cancellation in PF-PF(EDH)**: When all particles use the same flow (EDH), the Jacobian determinant cancels in the importance weight ratio. This saves significant computation.

3. **Continuation method for TPBVP**: When the BVP is hard (non-convex), solve a sequence of easier problems by gradually increasing the stiffness penalty mu.

4. **Conditional architecture for neural OT**: By feeding all problem-specific variables (model params, observation, statistics) to the network, we avoid retraining for each new filtering instance.

---

## 5. Project Flow Diagram

```
                    Linear-Gaussian SSM
                          │
                    ┌─────┴──────┐
                    │ Kalman Filter │  Part 1.I
                    │ (Joseph form) │
                    └─────┬──────┘
                          │
                 Nonlinear/Non-Gaussian SSM
                   (Stochastic Volatility)
                          │
              ┌───────────┼───────────┐
              │           │           │
         ┌────┴───┐  ┌───┴───┐  ┌───┴────┐
         │  EKF   │  │  UKF  │  │   PF   │  Part 1.II
         │(Jacobian)│ │(sigma)│  │ (SIR)  │
         └────┬───┘  └───┬───┘  └───┬────┘
              └───────────┼───────────┘
                          │ "PF works but not differentiable"
                          │
              ┌───────────┼───────────┐
              │           │           │
         ┌────┴───┐  ┌───┴───┐  ┌───┴────┐
         │  EDH   │  │ LEDH  │  │ PF-PF  │  Part 1, Q2
         │ (flow) │  │(local)│  │(Li 17) │
         └────┬───┘  └───┬───┘  └───┬────┘
              └───────────┼───────────┘
                          │ "Can we optimize the flow?"
                          │
              ┌───────────┴───────────┐
              │    Stochastic Flow     │
              │    (Dai 2022)          │  Part 2.1
              │  Optimal homotopy β*   │
              └───────────┬───────────┘
                          │ "Can we make PF differentiable?"
                          │
              ┌───────────┼───────────┐
              │           │           │
         ┌────┴───┐  ┌───┴───┐  ┌───┴────┐
         │  Soft  │  │Gumbel │  │   OT   │  Part 2.2
         │resample│  │Softmax│  │Sinkhorn│
         └────┬───┘  └───┬───┘  └───┬────┘
              └───────────┼───────────┘
                          │ "Now PF is differentiable!"
                          │
              ┌───────────┼───────────┐
              │           │           │
         ┌────┴───┐  ┌───┴───┐  ┌───┴─────┐
         │  HMC   │  │Neural │  │  SSL +  │
         │+ DPF   │  │  OT   │  │   PG    │  Bonus 1-3
         │(Bonus1)│  │(Bonus2│  │(Bonus 3)│
         └────────┘  └───────┘  └─────────┘
```

---

## 6. Key Results Summary

| Experiment | Best Method | Key Metric | Significance |
|---|---|---|---|
| LGSSM filtering | Kalman Filter | Exact solution | Baseline + Joseph form stability |
| SV model filtering | PF (N=500) | RMSE 1.21 | PF > UKF (1.64) > EKF (1.82) |
| Li17 replication | PF-PF (EDH) | RMSE 1.35 | Flow proposal + importance weighting |
| LEDH vs EDH | LEDH | RMSE 1.74 vs 1.89 | Per-particle adaptation wins |
| Dai22 stiffness | Scheduled mixing | 78.6% MSE reduction | Pragmatic > theoretical on non-convex |
| Dai22 + Li17 | Mixed results | EDH +2.9%, LEDH -0.05% | Distribution shift invalidates pre-computed beta |
| DPF resampling | OT/Sinkhorn | RMSE 1.19, lowest grad var | Best for gradient-based learning |
| HMC vs PMMH | HMC | 3x ESS/second | Gradient-guided >> random walk |
| Gaussian SSL | DPF-HMC | RMSE 0.12-0.20 | DPF-HMC wins for continuous states |
| Topical SSL | Particle Gibbs | Perplexity 15-25 | PG wins for discrete states |

---

## 7. Key Insights for Interview

### Insight 1: The Differentiability Hierarchy
```
Non-differentiable:  Multinomial resampling → blocks gradients
        ↓ (fix with)
Soft differentiable:  Gumbel-Softmax, soft resampling → biased gradients
        ↓ (improve with)
OT differentiable:    Sinkhorn resampling → low-variance gradients
        ↓ (accelerate with)
Neural OT:            mGradNet/FNO → 10-100x faster, differentiable
```

### Insight 2: Local vs Global Adaptation
- **EDH** (global flow): All particles share one flow → efficient but misses local structure
- **LEDH** (local flow): Per-particle flow → better accuracy but higher cost
- **Implication**: When you optimize globally (Dai22) on top of something locally adapted (LEDH), the global optimization is redundant

### Insight 3: Assumptions Must Be Checked at Runtime
- Dai22's optimal beta is computed assuming particles stay near their initial positions
- During flow integration, particles move → distribution shift
- The "optimal" schedule becomes sub-optimal or harmful
- **General lesson**: Pre-computed optimizations fail when the system they optimize over changes during execution

### Insight 4: Problem Structure Dictates Method Choice
| Problem Type | Best Method | Why |
|---|---|---|
| Linear Gaussian | Kalman Filter | Exact, O(n^3) |
| Mildly nonlinear | UKF | Good approximation, fast |
| Strongly nonlinear | PF or PF-PF | Handles arbitrary distributions |
| Need gradients (continuous) | DPF + OT | End-to-end learning |
| Need gradients (discrete) | Particle Gibbs | Gumbel-Softmax too biased |
| High-dimensional | LEDH + PF-PF | Per-particle flow scales well |

### Insight 5: Engineering Matters as Much as Theory
- Joseph form KF vs standard KF: same math, different numerics, wildly different stability
- Log-domain computation: essential for particle filters with many particles
- Ridge regularization: makes theoretically impossible problems practically solvable
- Automatic differentiation: eliminates entire class of implementation bugs (wrong Jacobians)

---

## 8. Code Quality Metrics

| Metric | Value |
|---|---|
| Total source code | ~6,000 LOC |
| Model classes | 6 |
| Filter classes | 18 |
| Inference classes | 3 |
| Example scripts | 16 |
| Test files | 5 |
| Framework | TensorFlow 2 + NumPy + SciPy |
| All models inherit from | `StateSpaceModel` (abstract base) |
| All filters support | `run(observations)` → `(estimates, ...)` |
