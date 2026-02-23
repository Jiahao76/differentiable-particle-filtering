"""
Robust Optimal Homotopy Solver for Highly Nonlinear Problems (Dai22 Section 3).

This implementation is specifically designed for bearing-only tracking and other
non-convex, highly nonlinear scenarios where the likelihood Hessian M_h may not
be positive definite.

Key improvements over standard implementation:
1. Nuclear norm condition number κ*(M) = tr(M)·tr(M^{-1})
2. Ridge regularization: M_reg = M + λ_reg·I to ensure positive definiteness
3. Gradient clipping to prevent BVP solver divergence
4. Continuation method (progressive relaxation)
5. Improved initial guess with 50+ points
6. Adaptive scaling to handle eigenvalue imbalance
"""

import tensorflow as tf
import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline


class RobustHomotopyOptimizer:
    """
    Solves for optimal β*(λ) that minimizes condition number of M(λ).
    
    TPBVP from Dai22 Theorem 3.1:
        d²β*/dλ² = μ * ∂κ*(M)/∂β
        β*(0) = 0, β*(1) = 1
    
    where M(λ) = M_0 + β(λ) * M_h (with ridge regularization)
          M_0 = -∇∇^T log p_0 (prior Hessian)
          M_h = -∇∇^T log h (likelihood Hessian)
    
    Nuclear norm condition number (Dai22 Eq. 28):
        κ*(M) = tr(M) · tr(M^{-1})
        ∂κ*/∂β = tr(M_h)·tr(M^{-1}) - tr(M)·tr(M^{-2}·M_h)
    """
    
    def __init__(self, mu=0.2, ridge_reg=1e-2, grad_clip=100.0):
        """
        Args:
            mu: Weight for condition number penalty in objective
            ridge_reg: Ridge regularization parameter (λ_reg)
            grad_clip: Maximum absolute value for gradient clipping
        """
        self.mu = mu
        self.ridge_reg = ridge_reg
        self.grad_clip = grad_clip
        self.M0 = None
        self.Mh = None
        self.lambda_grid = None
        self.beta_vals = None
        self.u_vals = None
    
    def condition_number_nuclear(self, M):
        """
        Compute nuclear norm condition number κ*(M) = tr(M)·tr(M^{-1}).
        
        Args:
            M: Matrix (numpy array)
            
        Returns:
            κ*(M): Nuclear norm condition number
        """
        M_np = M if isinstance(M, np.ndarray) else M.numpy()
        
        # Apply ridge regularization
        M_reg = M_np + self.ridge_reg * np.eye(M_np.shape[0])
        
        try:
            M_inv = np.linalg.inv(M_reg)
            tr_M = np.trace(M_reg)
            tr_M_inv = np.trace(M_inv)
            kappa = tr_M * tr_M_inv
            return float(kappa)
        except np.linalg.LinAlgError:
            # Matrix is still singular even with regularization
            return 1e10
    
    def derivative_condition_number(self, beta):
        """
        Compute ∂κ*(M)/∂β using Dai22 Eq. (28).
        
        For nuclear norm:
            ∂κ*/∂β = tr(M_h)·tr(M^{-1}) - tr(M)·tr(M^{-2}·M_h)
        
        Args:
            beta: Homotopy parameter β ∈ [0, 1]
            
        Returns:
            ∂κ*/∂β: Derivative of condition number w.r.t. β
        """
        beta_val = float(beta)
        
        # Clip beta to valid range
        beta_val = np.clip(beta_val, -0.1, 1.5)
        
        # Compute M(β) = M_0 + β·M_h with ridge regularization
        M = self.M0 + beta_val * self.Mh
        M_reg = M + self.ridge_reg * np.eye(M.shape[0])
        
        try:
            # Compute required matrix operations
            M_inv = np.linalg.inv(M_reg)
            M_inv_sq = M_inv @ M_inv
            
            # Compute traces
            tr_M = np.trace(M_reg)
            tr_M_inv = np.trace(M_inv)
            tr_Mh = np.trace(self.Mh)
            tr_M_inv_sq_Mh = np.trace(M_inv_sq @ self.Mh)
            
            # Derivative formula (Dai22 Eq. 28)
            # Note: Corrected sign (minus, not plus)
            d_kappa = tr_Mh * tr_M_inv - tr_M * tr_M_inv_sq_Mh
            
            # Clip gradient to prevent BVP solver divergence
            d_kappa_clipped = np.clip(d_kappa, -self.grad_clip, self.grad_clip)
            
            return float(d_kappa_clipped)
            
        except np.linalg.LinAlgError as e:
            print(f"Warning: LinAlgError at β={beta_val:.4f}: {e}")
            return 0.0
        except Exception as e:
            print(f"Warning: Error in derivative_condition_number at β={beta_val:.4f}: {e}")
            return 0.0
    
    def ode_system(self, lam, y):
        """
        ODE system for TPBVP in scipy.integrate.solve_bvp format.
        
        System:
            dβ/dλ = u
            du/dλ = μ · ∂κ*/∂β
        
        Args:
            lam: λ values (array or scalar)
            y: State [β, u] (shape: (2, n_points) or (2,))
            
        Returns:
            dy/dλ: Derivatives [dβ/dλ, du/dλ]
        """
        # Handle both scalar and array inputs
        is_scalar = np.isscalar(lam)
        
        if is_scalar:
            # Single point evaluation
            beta = y[0]
            u = y[1]
            
            d_kappa_d_beta = self.derivative_condition_number(beta)
            
            dbeta_dlam = float(u)
            du_dlam = float(self.mu * d_kappa_d_beta)
            
            return np.array([dbeta_dlam, du_dlam], dtype=np.float64)
        else:
            # Multiple points (vectorized)
            n_points = lam.shape[0] if hasattr(lam, 'shape') else len(lam)
            dy = np.zeros((2, n_points), dtype=np.float64)
            
            for i in range(n_points):
                beta = y[0, i]
                u = y[1, i]
                
                d_kappa_d_beta = self.derivative_condition_number(beta)
                
                dy[0, i] = float(u)
                dy[1, i] = float(self.mu * d_kappa_d_beta)
            
            return dy
    
    def boundary_conditions(self, ya, yb):
        """
        Boundary conditions for TPBVP:
            β(0) = 0
            β(1) = 1
        
        Args:
            ya: State at λ=0
            yb: State at λ=1
            
        Returns:
            Residuals for boundary conditions
        """
        return np.array([ya[0] - 0.0, yb[0] - 1.0], dtype=np.float64)
    
    def solve_bvp_direct(self, n_initial_points=50):
        """
        Solve TPBVP directly using scipy.integrate.solve_bvp with improved initial guess.
        
        Args:
            n_initial_points: Number of points in initial guess
            
        Returns:
            (lambda_grid, beta_vals, u_vals) or (None, None, None) if failed
        """
        # Create fine initial grid
        lambda_grid = np.linspace(0, 1, n_initial_points)
        
        # Initial guess: linear interpolation β(λ) = λ, u(λ) = 1
        beta_guess = lambda_grid.copy()
        u_guess = np.ones_like(lambda_grid)
        y_guess = np.vstack([beta_guess, u_guess])
        
        print(f"  Attempting solve_bvp with {n_initial_points} initial points...")
        print(f"  Ridge regularization: λ_reg = {self.ridge_reg}")
        print(f"  Gradient clipping: |∂κ/∂β| ≤ {self.grad_clip}")
        
        try:
            solution = solve_bvp(
                self.ode_system,
                self.boundary_conditions,
                lambda_grid,
                y_guess,
                tol=1e-2,  # Relaxed tolerance for nonlinear problems
                max_nodes=5000,
                verbose=0
            )
            
            if not solution.success:
                print(f"  ✗ solve_bvp failed: {solution.message}")
                return None, None, None
            
            # Extract solution on fine grid
            lambda_fine = np.linspace(0, 1, 201)
            y_fine = solution.sol(lambda_fine)
            beta_vals = y_fine[0]
            u_vals = y_fine[1]
            
            # Verify boundary conditions
            bc_error = max(abs(beta_vals[0] - 0.0), abs(beta_vals[-1] - 1.0))
            if bc_error > 0.05:
                print(f"  ✗ Boundary conditions not satisfied: error = {bc_error:.4f}")
                return None, None, None
            
            print(f"  ✓ solve_bvp succeeded!")
            print(f"    β(0) = {beta_vals[0]:.6f}, β(1) = {beta_vals[-1]:.6f}")
            print(f"    β' range: [{u_vals.min():.4f}, {u_vals.max():.4f}]")
            
            return lambda_fine, beta_vals, u_vals
            
        except Exception as e:
            print(f"  ✗ solve_bvp raised exception: {e}")
            return None, None, None
    
    def solve_bvp_continuation(self, mu_schedule=[0.02, 0.05, 0.1, 0.2]):
        """
        Solve TPBVP using continuation method (progressive relaxation).
        
        Start with small μ (weak penalty on condition number), then gradually
        increase μ to final value. Use previous solution as initial guess.
        
        Args:
            mu_schedule: List of μ values to try in sequence
            
        Returns:
            (lambda_grid, beta_vals, u_vals) or (None, None, None) if failed
        """
        print("  Using continuation method with μ schedule:", mu_schedule)
        
        # Start with linear homotopy
        lambda_grid = np.linspace(0, 1, 50)
        beta_vals = lambda_grid.copy()
        u_vals = np.ones_like(lambda_grid)
        
        mu_original = self.mu
        
        for i, mu_current in enumerate(mu_schedule):
            self.mu = mu_current
            print(f"\n  Stage {i+1}/{len(mu_schedule)}: μ = {mu_current}")
            
            # Use previous solution as initial guess
            y_guess = np.vstack([beta_vals, u_vals])
            
            try:
                solution = solve_bvp(
                    self.ode_system,
                    self.boundary_conditions,
                    lambda_grid,
                    y_guess,
                    tol=1e-2,
                    max_nodes=5000,
                    verbose=0
                )
                
                if not solution.success:
                    print(f"    ✗ Failed at μ = {mu_current}: {solution.message}")
                    self.mu = mu_original
                    return None, None, None
                
                # Extract solution
                lambda_fine = np.linspace(0, 1, 201)
                y_fine = solution.sol(lambda_fine)
                beta_vals = y_fine[0]
                u_vals = y_fine[1]
                lambda_grid = lambda_fine
                
                # Verify boundary conditions
                bc_error = max(abs(beta_vals[0] - 0.0), abs(beta_vals[-1] - 1.0))
                if bc_error > 0.05:
                    print(f"    ✗ Boundary conditions not satisfied: error = {bc_error:.4f}")
                    self.mu = mu_original
                    return None, None, None
                
                print(f"    ✓ Success: β(0)={beta_vals[0]:.6f}, β(1)={beta_vals[-1]:.6f}")
                
            except Exception as e:
                print(f"    ✗ Exception at μ = {mu_current}: {e}")
                self.mu = mu_original
                return None, None, None
        
        # Restore original μ
        self.mu = mu_original
        
        print(f"\n  ✓ Continuation method succeeded!")
        return lambda_grid, beta_vals, u_vals
    
    def solve_optimal_homotopy(self, M0, Mh, method='continuation'):
        """
        Main interface: solve for optimal β*(λ).
        
        Args:
            M0: Prior Hessian -∇∇^T log p_0 (positive semi-definite)
            Mh: Likelihood Hessian -∇∇^T log h (may not be PSD)
            method: 'direct', 'continuation', or 'auto'
            
        Returns:
            beta_func: Function β(λ) that returns (α, β, α', β')
        """
        # Convert to numpy
        self.M0 = M0.numpy() if isinstance(M0, tf.Tensor) else M0
        self.Mh = Mh.numpy() if isinstance(Mh, tf.Tensor) else Mh
        
        print(f"\nSolving optimal homotopy with robust BVP solver:")
        print(f"  μ = {self.mu}")
        print(f"  Ridge regularization: λ_reg = {self.ridge_reg}")
        print(f"  Gradient clipping: {self.grad_clip}")
        print(f"  Method: {method}")
        
        # Check Hessian properties
        print(f"\nHessian diagnostics:")
        print(f"  ||M0||_F = {np.linalg.norm(self.M0):.4e}")
        print(f"  ||Mh||_F = {np.linalg.norm(self.Mh):.4e}")
        
        eigvals_M0 = np.linalg.eigvalsh(self.M0)
        eigvals_Mh = np.linalg.eigvalsh(self.Mh)
        print(f"  M0 eigenvalues: {eigvals_M0}")
        print(f"  Mh eigenvalues: {eigvals_Mh}")
        
        # Check if Mh is near-zero (degenerate case)
        if np.linalg.norm(self.Mh) < 1e-8:
            print("\n⚠️  Mh has near-zero norm. Using linear homotopy.")
            return self._linear_homotopy()
        
        # Try to solve TPBVP
        lambda_grid, beta_vals, u_vals = None, None, None
        
        if method == 'continuation' or method == 'auto':
            lambda_grid, beta_vals, u_vals = self.solve_bvp_continuation()
            
            if lambda_grid is None and method == 'auto':
                print("\n  Continuation method failed, trying direct solve...")
                lambda_grid, beta_vals, u_vals = self.solve_bvp_direct()
        
        elif method == 'direct':
            lambda_grid, beta_vals, u_vals = self.solve_bvp_direct()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if beta_vals is None:
            print("\n⚠️  TPBVP solver failed. Using linear homotopy.")
            return self._linear_homotopy()
        
        # Compute objective values for comparison
        J_optimal = self._compute_objective(lambda_grid, beta_vals, u_vals)
        J_linear = self._compute_objective(lambda_grid, lambda_grid, np.ones_like(lambda_grid))
        
        print(f"\n✓ Optimal homotopy solved successfully!")
        print(f"  J(optimal) = {J_optimal:.6e}")
        print(f"  J(linear)  = {J_linear:.6e}")
        
        if J_optimal < J_linear:
            improvement = (J_linear - J_optimal) / J_linear * 100
            print(f"  Improvement: {improvement:.2f}%")
        else:
            print(f"  ⚠️  Optimal solution has higher cost (numerical issues)")
        
        # Store solution
        self.lambda_grid = lambda_grid
        self.beta_vals = beta_vals
        self.u_vals = u_vals
        
        # Create interpolated function
        return self._create_beta_function(lambda_grid, beta_vals, u_vals)
    
    def _compute_objective(self, lambda_grid, beta_vals, u_vals):
        """
        Compute objective J = ∫[0.5·u² + μ·κ*(M)] dλ
        """
        integrand = np.zeros(len(beta_vals))
        
        for i, (beta, u) in enumerate(zip(beta_vals, u_vals)):
            M = self.M0 + beta * self.Mh
            kappa = self.condition_number_nuclear(M)
            integrand[i] = 0.5 * u**2 + self.mu * kappa
        
        # Trapezoidal integration
        J = np.trapz(integrand, lambda_grid)
        return J
    
    def _create_beta_function(self, lambda_grid, beta_vals, u_vals):
        """
        Create TensorFlow function for β(λ) using cubic spline interpolation.
        """
        # Create cubic spline interpolators
        cs_beta = CubicSpline(lambda_grid, beta_vals, bc_type='natural')
        cs_u = CubicSpline(lambda_grid, u_vals, bc_type='natural')
        
        def beta_func(lam):
            """
            Interpolated β(λ) function.
            
            Returns:
                (α, β, α', β') where α(λ) = 1 - β(λ)
            """
            lam_val = float(lam) if isinstance(lam, tf.Tensor) else lam
            
            # Cubic spline interpolation
            beta = float(cs_beta(lam_val))
            u = float(cs_u(lam_val))
            
            # Ensure beta stays in valid range
            beta = np.clip(beta, 0.0, 1.0)
            
            # Compute α and derivatives
            alpha = 1.0 - beta
            alpha_dot = -u
            beta_dot = u
            
            return (
                tf.constant(alpha, dtype=tf.float32),
                tf.constant(beta, dtype=tf.float32),
                tf.constant(alpha_dot, dtype=tf.float32),
                tf.constant(beta_dot, dtype=tf.float32)
            )
        
        return beta_func
    
    def _linear_homotopy(self):
        """Fallback to linear homotopy β(λ) = λ"""
        print("  Using linear homotopy: β(λ) = λ")
        
        def beta_func(lam):
            lam_t = tf.cast(lam, tf.float32)
            alpha = 1.0 - lam_t
            beta = lam_t
            alpha_dot = tf.constant(-1.0, dtype=tf.float32)
            beta_dot = tf.constant(1.0, dtype=tf.float32)
            return alpha, beta, alpha_dot, beta_dot
        
        return beta_func
