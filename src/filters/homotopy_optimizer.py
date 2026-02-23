"""
Optimal Homotopy Solver for Stiffness Mitigation (Dai22 Section 3).
Solves the Two-Point Boundary Value Problem (TPBVP) for optimal β*(λ).
"""

import tensorflow as tf
import numpy as np
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.interpolate import CubicSpline


class HomotopyOptimizer:
    """
    Solves for optimal β*(λ) that minimizes condition number of M(λ).
    
    TPBVP from Dai22 Theorem 3.1:
        d²β*/dλ² = μ * ∂κ(M)/∂β
        β*(0) = 0, β*(1) = 1
    
    where M(λ) = M_0 + β(λ) * M_h
          M_0 = -∇∇^T log p_0 (prior Hessian)  [converted to positive semi-definite]
          M_h = -∇∇^T log h (likelihood Hessian) [converted to positive semi-definite]
    
    Energy Hessians are used (negative log-density Hessians for minimization).
    """
    
    def __init__(self, mu=0.2, norm_type='nuclear'):
        self.mu = mu
        self.norm_type = norm_type
        self.M0 = None
        self.Mh = None
        self.lambda_grid = None
        self.beta_vals = None
        self.u_vals = None
        self.beta_interp = None
    
    def condition_number(self, M):
        """
        Compute κ(M) using specified norm.
        
        For nuclear norm (Dai22 Remark 3.2):
            κ*(M) = tr(M) * tr(M^{-1})
        """
        M_np = M.numpy() if isinstance(M, tf.Tensor) else M
        
        # Add regularization for numerical stability
        M_reg = M_np + 1e-6 * np.eye(M_np.shape[0])
        
        if self.norm_type == 'nuclear':
            # Nuclear norm condition number
            tr_M = np.trace(M_reg)
            tr_M_inv = np.trace(np.linalg.inv(M_reg))
            return tr_M * tr_M_inv
        
        elif self.norm_type == 'spectral':
            # Spectral (L2) norm condition number
            eigvals = np.linalg.eigvalsh(M_reg)
            return eigvals[-1] / eigvals[0]
        
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}")
    
    def derivative_condition_number(self, beta):
        """
        Compute ∂κ(M)/∂β using analytical formula.
        
        For nuclear norm (Dai22 Eq. 28):
            ∂κ*/∂β = tr(M_h) * tr(M^{-1}) + tr(M) * tr(M^{-2} * M_h)
        
        Note: Paper uses -Hessian convention, so M = -∇∇^T log p
        """
        beta_val = float(beta)
        
        # Compute M(β) = M_0 + β * M_h
        M = self.M0 + beta_val * self.Mh
        
        # Add larger regularization for numerical stability
        reg = 1e-4 * np.eye(M.shape[0])
        M_reg = M + reg
        
        try:
            if self.norm_type == 'nuclear':
                # Compute inverse with regularization
                M_inv = np.linalg.inv(M_reg)
                M_inv_sq = M_inv @ M_inv
                
                # Compute traces
                tr_M = np.trace(M_reg)
                tr_M_inv = np.trace(M_inv)
                tr_Mh = np.trace(self.Mh)
                tr_M_inv_sq_Mh = np.trace(M_inv_sq @ self.Mh)
                
                # Derivative formula from Dai22 Eq. 28 - CORRECTED SIGN
                # ∂κ*/∂β = tr(M_h)*tr(M^{-1}) - tr(M)*tr(M^{-2}*M_h)
                # Original plus sign was incorrect; should be minus (verified by scalar test)
                d_kappa = tr_Mh * tr_M_inv - tr_M * tr_M_inv_sq_Mh
                
                return float(d_kappa)
            
            else:
                # Numerical derivative for other norms
                eps = 1e-5
                kappa_plus = self.condition_number(self.M0 + (beta_val + eps) * self.Mh)
                kappa_minus = self.condition_number(self.M0 + (beta_val - eps) * self.Mh)
                d_kappa = (kappa_plus - kappa_minus) / (2 * eps)
                return float(d_kappa)
        
        except np.linalg.LinAlgError as e:
            # Matrix is singular or near-singular
            print(f"Warning: LinAlgError at β={beta_val:.4f}: {e}")
            return 0.0
        
        except Exception as e:
            print(f"Warning: Unexpected error in derivative_condition_number at β={beta_val:.4f}: {e}")
            return 0.0
    
    def ode_system(self, y, lam):
        """
        ODE system for TPBVP (scipy.odeint format: func(y, t)).
        
        System:
            dy[0]/dλ = y[1]           (β' = u)
            dy[1]/dλ = μ * ∂κ/∂β      (u' = μ * ∂κ/∂β)
        
        Args:
            y: [β, u] state vector (numpy array)
            lam: λ value (scalar, not used directly but required by odeint)
            
        Returns:
            dy/dλ: [β', u'] derivatives as numpy array
        """
        # Ensure y is a numpy array
        y = np.asarray(y, dtype=np.float64)
        
        # Extract beta and u
        beta = float(y[0])
        u = float(y[1])
        
        # Clip beta to valid range to avoid numerical issues
        beta = np.clip(beta, -0.1, 1.5)
        
        try:
            # Compute derivative of condition number
            d_kappa_d_beta = self.derivative_condition_number(beta)
            
            # Ensure scalar output
            if isinstance(d_kappa_d_beta, np.ndarray):
                d_kappa_d_beta = float(d_kappa_d_beta)
            
            # Return derivatives as numpy array
            dbeta_dlam = float(u)
            du_dlam = float(self.mu * d_kappa_d_beta)
            
            return np.array([dbeta_dlam, du_dlam], dtype=np.float64)
        
        except Exception as e:
            # If derivative computation fails, return zero derivatives
            print(f"Warning: ODE derivative computation failed at β={beta:.4f}: {e}")
            return np.array([0.0, 0.0], dtype=np.float64)
    
    def shooting_objective(self, u0, lambda_grid):
        """
        Shooting method objective: returns β(1) - 1.
        
        Integrate ODE from λ=0 with initial condition [0, u0],
        then check if β(1) = 1. Uses solve_ivp with LSODA for stiff ODEs.
        """
        y0 = np.array([0.0, u0], dtype=np.float64)
        
        try:
            # Use solve_ivp with LSODA method for better stiff ODE handling
            def ode_func(lam, y):
                return self.ode_system(y, lam)
            
            solution = solve_ivp(
                ode_func,
                [lambda_grid[0], lambda_grid[-1]],
                y0,
                t_eval=lambda_grid,
                method='LSODA',
                rtol=1e-6,
                atol=1e-8,
                max_step=0.1,
                dense_output=False
            )
            
            if not solution.success:
                raise ValueError(f"Integration failed: {solution.message}")
            
            # Extract beta at final lambda
            beta_final = solution.y[0, -1]
            
            # Return error at λ=1
            return float(beta_final) - 1.0
        
        except Exception as e:
            print(f"Warning: ODE integration failed with u0={u0}: {e}")
            return np.inf
    
    def _ode_for_bvp(self, lam, y):
        """
        ODE system in standard form for solve_bvp: dy/dλ = f(λ, y).
        y = [β, β']
        Returns dy/dλ = [β', β'']
        
        This version handles BOTH scalar and array inputs from solve_bvp.
        """
        # Handle both scalar and array inputs
        if np.isscalar(lam):
            # Single point evaluation
            beta = y[0]
            u = y[1]
            
            # Clip beta to valid range
            beta = np.clip(beta, -0.1, 1.5)
            
            try:
                d_kappa_d_beta = self.derivative_condition_number(beta)
                dbeta_dlam = float(u)
                du_dlam = float(self.mu * d_kappa_d_beta)
                
                return np.array([dbeta_dlam, du_dlam], dtype=np.float64)
            except Exception as e:
                return np.array([0.0, 0.0], dtype=np.float64)
        else:
            # Multiple points (array input)
            n_points = len(lam) if hasattr(lam, '__len__') else 1
            dy = np.zeros((2, n_points), dtype=np.float64)
            
            for i in range(n_points):
                lam_i = lam[i] if hasattr(lam, '__getitem__') else lam
                y_i = y[:, i] if y.ndim > 1 else y
                
                beta = y_i[0]
                u = y_i[1]
                
                beta = np.clip(beta, -0.1, 1.5)
                
                try:
                    d_kappa_d_beta = self.derivative_condition_number(beta)
                    dy[0, i] = float(u)
                    dy[1, i] = float(self.mu * d_kappa_d_beta)
                except Exception:
                    dy[:, i] = 0.0
            
            return dy
    
    def _bc_for_bvp(self, ya, yb):
        """
        Boundary conditions for solve_bvp:
        ya[0] = β(0) = 0
        yb[0] = β(1) = 1
        """
        return np.array([ya[0] - 0.0, yb[0] - 1.0], dtype=np.float64)
    
    def solve_bvp_method(self, lambda_grid=None):
        """
        Solve TPBVP using scipy.integrate.solve_bvp (more robust than shooting).
        
        Returns:
            lambda_grid, beta_vals, u_vals (or None, None, None if failed)
        """
        if lambda_grid is None:
            lambda_grid = np.linspace(0, 1, 101)
        
        # Initial guess: linear homotopy
        x_guess = np.array([lambda_grid, np.ones_like(lambda_grid)])
        
        try:
            # Solve BVP
            solution = solve_bvp(
                self._ode_for_bvp,
                self._bc_for_bvp,
                lambda_grid,
                x_guess,
                max_nodes=10000,
                tol=1e-6,
                verbose=0
            )
            
            if not solution.success:
                print(f"Warning: solve_bvp failed: {solution.message}")
                return None, None, None
            
            # Extract solution on fine grid
            lambda_fine = np.linspace(0, 1, 201)
            y_fine = solution.sol(lambda_fine)
            
            beta_vals = y_fine[0]
            u_vals = y_fine[1]
            
            # Verify boundary conditions
            if abs(beta_vals[0] - 0.0) > 0.01 or abs(beta_vals[-1] - 1.0) > 0.01:
                print(f"Warning: Boundary conditions not satisfied: β(0)={beta_vals[0]:.4f}, β(1)={beta_vals[-1]:.4f}")
                return None, None, None
            
            return lambda_fine, beta_vals, u_vals
        
        except Exception as e:
            print(f"Warning: solve_bvp raised exception: {e}")
            return None, None, None
    
    def solve_shooting_method(self, lambda_grid=None, tol=1e-6):
        """
        Solve TPBVP using shooting method with bisection.
        
        Find u0 = β'(0) such that β(1) = 1.
        """
        if lambda_grid is None:
            lambda_grid = np.linspace(0, 1, 101)
        
        # Bisection search for u0 - reasonable initial range
        u0_low, u0_high = -100.0, 100.0
        
        # Ensure we bracket the solution
        max_attempts = 15
        for attempt in range(max_attempts):
            f_low = self.shooting_objective(u0_low, lambda_grid)
            f_high = self.shooting_objective(u0_high, lambda_grid)
            
            if not np.isfinite(f_low) or not np.isfinite(f_high):
                print(f"Warning: Non-finite values in shooting method (attempt {attempt + 1})")
                u0_low *= 0.5
                u0_high *= 0.5
                continue
            
            if f_low * f_high < 0:
                # Successfully bracketed
                print(f"  Successfully bracketed solution: f({u0_low:.2e})={f_low:.6f}, f({u0_high:.2e})={f_high:.6f}")
                break
            
            # Expand search range
            u0_low *= 2
            u0_high *= 2
            
            if abs(u0_low) > 1e5:  # Reasonable max to prevent infinite expansion
                print("Warning: Failed to bracket solution in shooting method")
                print(f"  Tried range [{u0_low:.2e}, {u0_high:.2e}]")
                return None, None, None
        
        # Bisection
        iteration = 0
        max_iter = 50
        
        while abs(u0_high - u0_low) > tol and iteration < max_iter:
            u0_mid = (u0_low + u0_high) / 2
            f_mid = self.shooting_objective(u0_mid, lambda_grid)
            f_low = self.shooting_objective(u0_low, lambda_grid)
            
            if not np.isfinite(f_mid):
                print(f"Warning: Non-finite value at iteration {iteration}")
                break
            
            if f_mid * f_low < 0:
                u0_high = u0_mid
            else:
                u0_low = u0_mid
            
            iteration += 1
        
        u0_optimal = (u0_low + u0_high) / 2
        
        # Integrate with optimal initial condition using solve_ivp for stability
        y0 = np.array([0.0, u0_optimal])
        
        def ode_func(lam, y):
            return self.ode_system(y, lam)
        
        solution = solve_ivp(
            ode_func,
            [lambda_grid[0], lambda_grid[-1]],
            y0,
            t_eval=lambda_grid,
            method='LSODA',
            rtol=1e-6,
            atol=1e-8,
            max_step=0.1
        )
        
        if not solution.success:
            print(f"Warning: Final ODE integration failed: {solution.message}")
            return None, None, None
        
        beta_optimal = solution.y[0]
        u_optimal = solution.y[1]
        
        # Verify boundary conditions
        if abs(beta_optimal[-1] - 1.0) > 0.01:
            print(f"Warning: β(1) = {beta_optimal[-1]:.6f}, expected 1.0")
            return None, None, None
        
        return lambda_grid, beta_optimal, u_optimal
    
    def solve_optimal_homotopy(self, M0, Mh, method='bvp'):
        """
        Main interface: solve for optimal β*(λ).
        
        Args:
            M0: Energy Hessian -∇∇^T log p_0 (positive semi-definite)
            Mh: Energy Hessian -∇∇^T log h (positive semi-definite)
            method: 'bvp' (default, more robust) or 'shooting'
            
        Returns:
            beta_func: Function β(λ) that returns (α, β, α', β')
        """
        # Convert to numpy
        self.M0 = M0.numpy() if isinstance(M0, tf.Tensor) else M0
        self.Mh = Mh.numpy() if isinstance(Mh, tf.Tensor) else Mh
        
        # Check if Mh is near-zero (degenerate case)
        if np.linalg.norm(self.Mh) < 1e-8:
            print("Warning: Mh has near-zero norm. Using linear homotopy.")
            return self._linear_homotopy()
        
        # Check if M0 is positive definite
        eigvals_M0 = np.linalg.eigvalsh(self.M0)
        if np.any(eigvals_M0 <= 0):
            print(f"Warning: M0 not positive definite (eigvals: {eigvals_M0}). Using linear homotopy.")
            return self._linear_homotopy()
        
        print(f"Solving optimal homotopy with μ={self.mu}, norm={self.norm_type}, method={method}...")
        
        # Try solve_bvp first (more robust), fall back to shooting
        lambda_grid, beta_vals, u_vals = None, None, None
        
        if method in ['bvp', 'auto']:
            print("  Attempting solve_bvp...")
            lambda_grid, beta_vals, u_vals = self.solve_bvp_method()
            
            if lambda_grid is not None:
                print("  ✓ solve_bvp succeeded!")
                method_used = 'bvp'
            elif method == 'auto':
                print("  solve_bvp failed, trying shooting method...")
                lambda_grid, beta_vals, u_vals = self.solve_shooting_method()
                method_used = 'shooting'
            else:
                print("  ✗ solve_bvp failed and not in auto mode")
                method_used = None
        else:
            print("  Attempting shooting method...")
            lambda_grid, beta_vals, u_vals = self.solve_shooting_method()
            method_used = 'shooting'
        
        if beta_vals is None:
            print("Warning: TPBVP solver failed. Using linear homotopy.")
            return self._linear_homotopy()
        
        # Compute objective function value
        J_optimal = self.compute_objective(lambda_grid, beta_vals, u_vals)
        J_linear = self.compute_objective(lambda_grid, lambda_grid, np.ones_like(lambda_grid))
        
        print(f"✓ Optimal homotopy solved successfully with {method_used}!")
        print(f"  J(optimal) = {J_optimal:.6f}")
        print(f"  J(linear)  = {J_linear:.6f}")
        print(f"  Improvement: {(J_linear - J_optimal) / J_linear * 100:.2f}%")
        
        # Store solution for later reference
        self.lambda_grid = lambda_grid
        self.beta_vals = beta_vals
        self.u_vals = u_vals
        
        # Create interpolated function
        return self._create_beta_function(lambda_grid, beta_vals, u_vals)
    
    def compute_objective(self, lambda_grid, beta_vals, u_vals):
        """
        Compute objective J = ∫[0.5*u² + μ*κ(M)] dλ
        """
        integrand = np.zeros(len(beta_vals))
        
        for i, (beta, u) in enumerate(zip(beta_vals, u_vals)):
            M = self.M0 + beta * self.Mh
            kappa = self.condition_number(M)
            integrand[i] = 0.5 * u**2 + self.mu * kappa
        
        # Trapezoidal integration
        J = np.trapz(integrand, lambda_grid)
        return J
    
    def _create_beta_function(self, lambda_grid, beta_vals, u_vals):
        """
        Create TensorFlow function for β(λ) using cubic spline interpolation.
        This provides smooth derivatives needed for accurate particle flow.
        """
        # Store for reference
        lambda_grid_np = lambda_grid.copy()
        beta_vals_np = beta_vals.copy()
        u_vals_np = u_vals.copy()
        
        # Create cubic spline interpolators
        cs_beta = CubicSpline(lambda_grid_np, beta_vals_np, bc_type='natural')
        cs_u = CubicSpline(lambda_grid_np, u_vals_np, bc_type='natural')
        
        def beta_func(lam):
            """
            Interpolated β(λ) function with smooth derivatives.
            Returns: (α, β, α', β')
            
            where α(λ) = 1 - β(λ)
                  α'(λ) = -β'(λ)
                  β'(λ) = u(λ)
            """
            lam_val = float(lam) if isinstance(lam, tf.Tensor) else lam
            
            # Cubic spline interpolation (smooth derivatives)
            beta = float(cs_beta(lam_val))
            u = float(cs_u(lam_val))
            
            # Ensure beta stays in valid range
            beta = np.clip(beta, 0.0, 1.0)
            
            # α(λ) = 1 - β(λ) for normalized homotopy
            alpha = 1.0 - beta
            alpha_dot = -u
            beta_dot = u
            
            # Return as TensorFlow constants
            return (
                tf.constant(alpha, dtype=tf.float32),
                tf.constant(beta, dtype=tf.float32),
                tf.constant(alpha_dot, dtype=tf.float32),
                tf.constant(beta_dot, dtype=tf.float32)
            )
        
        return beta_func
    
    def _linear_homotopy(self):
        """Fallback to linear homotopy β(λ) = λ"""
        def beta_func(lam):
            lam_t = tf.cast(lam, tf.float32)
            alpha = 1.0 - lam_t
            beta = lam_t
            alpha_dot = tf.constant(-1.0, dtype=tf.float32)
            beta_dot = tf.constant(1.0, dtype=tf.float32)
            return alpha, beta, alpha_dot, beta_dot
        
        return beta_func