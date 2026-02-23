# Robust Optimal Homotopy for Bearing-Only Tracking

## Problem Background

In **Bearing-only Tracking**, a highly non-convex and nonlinear scenario, the likelihood Hessian matrix $M_h = -\nabla\nabla^T \log h$ may **not be positive definite**, which leads to:

1. Condition number $\kappa(M)$ and its gradient $\partial\kappa/\partial\beta$ exhibiting numerical jumps
2. BVP solver (`scipy.integrate.solve_bvp`) failing to converge or crashing
3. Standard Dai22 implementation failing

## Solution: Robust Homotopy Optimizer

We implemented the `RobustHomotopyOptimizer` class with the following key improvements:

### 1. Nuclear Norm Condition Number (Dai22 Formula 28)

**Do not use eigenvalue ratio**, as eigenvalues can cross zero in non-convex regions.

Use **nuclear norm condition number**:

$$\kappa^*(M) = \text{tr}(M) \cdot \text{tr}(M^{-1})$$

with gradient:

$$\frac{\partial\kappa^*}{\partial\beta} = \text{tr}(M_h) \cdot \text{tr}(M^{-1}) - \text{tr}(M) \cdot \text{tr}(M^{-2} M_h)$$

**Note**: The second term has a **negative sign** (the paper is misleading).

### 2. Ridge Regression Regularization

Regularize $M(\beta) = M_0 + \beta M_h$ to ensure it remains invertible and positive definite throughout pseudo-time $\lambda \in [0,1]$:

$$M_{\text{reg}}(\beta) = M(\beta) + \lambda_{\text{reg}} I$$

where $\lambda_{\text{reg}}$ is between $10^{-2}$ and $10^{-1}$.

**Code implementation**:

```python
M_reg = M + self.ridge_reg * np.eye(M.shape[0])
```

### 3. Gradient Clipping

To prevent BVP solver from reducing step size too much due to gradient spikes:

$$\frac{\partial\kappa^*}{\partial\beta} \leftarrow \text{clip}\left(\frac{\partial\kappa^*}{\partial\beta}, -C, C\right)$$

where $C = 100$.

**Code implementation**:

```python
d_kappa_clipped = np.clip(d_kappa, -self.grad_clip, self.grad_clip)
```

### 4. Continuation Method (Relaxation Factor)

If direct computation of $\mu$ doesn't converge (due to excessive nonlinearity), use **progressive relaxation**:

1. Start with smaller $\mu$ (e.g., $\mu = 0.02$), where solution is close to linear $\beta(\lambda) = \lambda$
2. Gradually increase $\mu$: $0.02 \to 0.05 \to 0.1 \to 0.2$
3. Use previous solution as initial guess for next iteration

**Code implementation**:

```python
def solve_bvp_continuation(self, mu_schedule=[0.02, 0.05, 0.1, 0.2]):
    """Progressive relaxation"""
    for mu_current in mu_schedule:
        self.mu = mu_current
        # Solve BVP with previous solution as initial guess
        solution = solve_bvp(...)
```

### 5. Improved Initial Guess

Don't provide just two points. Give an array with **50 points**, initialized to linear growth:

```python
lambda_grid = np.linspace(0, 1, 50)
beta_guess = lambda_grid.copy()
u_guess = np.ones_like(lambda_grid)  # β'(λ) = 1
y_guess = np.vstack([beta_guess, u_guess])
```

### 6. BVP Solver Configuration

```python
solution = solve_bvp(
    self.ode_system,
    self.boundary_conditions,
    lambda_grid,
    y_guess,
    tol=1e-2,        # Relaxed tolerance (nonlinear problem)
    max_nodes=5000,  # Increased maximum grid points
    verbose=0
)
```

## Usage

### Basic Usage

```python
from filters.homotopy_optimizer_robust import RobustHomotopyOptimizer

# Initialize optimizer
optimizer = RobustHomotopyOptimizer(
    mu=0.2,           # Dai22 recommended value
    ridge_reg=1e-2,   # Ridge regularization
    grad_clip=100.0   # Gradient clipping
)

# Solve optimal homotopy
beta_func = optimizer.solve_optimal_homotopy(
    M0=M0,  # Prior Hessian (positive definite)
    Mh=Mh,  # Likelihood Hessian (may be non-positive-definite)
    method='continuation'  # Use continuation method
)

# beta_func returns (α, β, α', β')
alpha, beta, alpha_dot, beta_dot = beta_func(lambda_val)
```

### Complete Example

See `examples/replicate_dai22_robust.py`, which includes:

1. Sample particles from prior, compute ensemble mean
2. Compute Hessian at ensemble mean
3. Use robust optimizer to solve optimal homotopy
4. Run particle flow filter
5. Compare with linear homotopy

## Test Results

Run test script:

```bash
python test_robust_homotopy.py
```

### Success Indicators

1. **BVP Convergence**: All methods (direct solve, continuation) successful
   ```
   ✓ solve_bvp succeeded!
   β(0) = -0.000000, β(1) = 1.000000
   ```

2. **Condition Number Reduction**: Optimal path has smaller condition number than linear path
   ```
   Condition number comparison:
     Optimal: max κ* = 22.76, mean = 4.38
     Linear:  max κ* = 51.84, mean = 4.66
     Reduction: 56.1%
   ```

3. **Convex Shape**: $\beta^*(\lambda)$ shows lower convex curve
   - When $\lambda$ is small (prior dominant), $\beta'(\lambda) < 1$, slow growth
   - When $\lambda$ is large (near likelihood), $\beta'(\lambda) > 1$, accelerated growth
   
   **Physical Meaning**: Delays introduction of ill-conditioned likelihood information, giving particle flow more time to align under the relatively "smooth" prior guidance.

## Key Technical Details

### Hessian Sign Convention

Dai22 uses **energy Hessian** (minimization problem):

$$M = -\nabla\nabla^T \log p$$

When computing, need to negate `model.hessian_log_prior()` and `model.hessian_log_likelihood()`:

```python
M0 = -model.hessian_log_prior()
Mh = -model.hessian_log_likelihood_numerical(ensemble_mean, z)
```

### Boundary Conditions

TPBVP boundary conditions:

$$\beta^*(0) = 0, \quad \beta^*(1) = 1$$

Code implementation:

```python
def boundary_conditions(self, ya, yb):
    return np.array([ya[0] - 0.0, yb[0] - 1.0])
```

### ODE System

$$\frac{d\beta}{d\lambda} = u$$

$$\frac{du}{d\lambda} = \mu \frac{\partial\kappa^*}{\partial\beta}$$

Code implementation:

```python
def ode_system(self, lam, y):
    beta = y[0]
    u = y[1]
    d_kappa_d_beta = self.derivative_condition_number(beta)
    dbeta_dlam = u
    du_dlam = self.mu * d_kappa_d_beta
    return np.array([dbeta_dlam, du_dlam])
```

## Expected Physical Effect

Successful `solve_bvp` should produce a **lower convex curve** (similar to $\beta(\lambda) \approx \lambda^2$):

```
  β*(λ)
   1.0 |                    ___/
       |                ___/
       |            ___/
   0.5 |        ___/
       |    ___/
       | __/
   0.0 |/___________________
       0.0       0.5      1.0   λ
```

This corresponds to:
- $\lambda \in [0, 0.5]$: $\beta'(\lambda) < 1$ (slow growth)
- $\lambda \in [0.5, 1]$: $\beta'(\lambda) > 1$ (fast growth)

## Comparison with Standard Implementation

| Feature | Standard Implementation | Robust Implementation |
|---------|------------------------|----------------------|
| Non-PSD Hessian | ✗ Crashes | ✓ Handles |
| Condition number formula | Eigenvalue ratio | Nuclear norm |
| Regularization | None | Ridge regression |
| Gradient handling | Direct use | Clipping |
| Initialization | 2-point linear | 50-point linear |
| Solve strategy | Direct solve | Continuation |
| BVP tolerance | 1e-6 | 1e-2 |
| Max grid points | 1000 | 5000 |

## Troubleshooting

If BVP solver still fails:

1. **Increase ridge_reg**: Try $0.1$ or $0.5$
2. **Increase grad_clip**: Try $200$ or $500$
3. **Use gentler continuation schedule**:
   ```python
   [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
   ```
4. **Relax BVP tolerance**: `tol=1e-1`
5. **Check Hessian computation**: Use numerical Hessian instead of analytical Hessian

## References

1. Daum & Huang (2022), "Optimal Importance Densities via Gromov's Method with Applications to Particle Filtering"
2. Dai et al. (2022), "Two-Point Boundary Value Problems for Optimal Homotopy"

## File Structure

```
src/filters/
  └── homotopy_optimizer_robust.py  # Robust optimizer implementation

examples/
  └── replicate_dai22_robust.py     # Complete Dai22 replication example

test_robust_homotopy.py             # Unit tests
```

## Summary

Through the above improvements, we upgrade Dai22's "Scheduled" approach to "**Adaptive Optimal Scheduling**", truly realizing the core academic value of the paper:

**In highly nonlinear, non-convex Bearing-only Tracking scenarios, by solving TPBVP to obtain optimal homotopy path $\beta^*(\lambda)$, significantly reduce condition number, and improve numerical stability and estimation accuracy of particle flow filters.**
