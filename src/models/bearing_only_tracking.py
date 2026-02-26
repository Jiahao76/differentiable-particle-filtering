"""
Bearing-only tracking model for Dai22 Section 4 replication.
Two passive infrared sensors measuring angles to a stationary target.
"""

import tensorflow as tf
import numpy as np
from src.models.base_model import StateSpaceModel


class BearingOnlyTrackingModel(StateSpaceModel):
    """
    2D Bearing-only tracking scenario from Dai22 Section 4.
    
    State: x = [x_pos, y_pos]^T (2D position)
    Observation: z = [bearing_1, bearing_2]^T (angles from two sensors)
    
    Key parameters from paper [Dai22, Section 4]:
    - Sensor 1: (3.5, 0)
    - Sensor 2: (-3.5, 0)  
    - True target: (4.0, 4.0)
    - Prior mean: [3.0, 5.0]  # Note: NOT [3.0, 0.0]!
    - Prior cov: diag(1000.0, 2.0)
    - Measurement noise: R = diag(0.04, 0.04)
    """
    
    def __init__(self):
        """Initialize the Bearing-Only Tracking Model (state_dim=2, obs_dim=2)."""
        super().__init__(state_dim=2, obs_dim=2)

        # Sensor locations [Dai22, Fig 1]
        self.sensors = tf.constant([[3.5, 0.0], [-3.5, 0.0]], dtype=tf.float32)
        
        # Prior distribution [Dai22, p.11]
        # CRITICAL FIX: Prior mean is [3.0, 5.0], not [3.0, 0.0]
        self.prior_mean = tf.constant([3.0, 5.0], dtype=tf.float32)
        self.prior_cov = tf.constant([[1000.0, 0.0], [0.0, 2.0]], dtype=tf.float32)
        self.inv_prior_cov = tf.linalg.inv(self.prior_cov)
        
        # Measurement noise [Dai22, p.11-12]
        self.R = tf.constant([[0.04, 0.0], [0.0, 0.04]], dtype=tf.float32)
        self.inv_R = tf.linalg.inv(self.R)
        
        # True target location (for evaluation)
        self.target_truth = tf.constant([4.0, 4.0], dtype=tf.float32)

    # --- StateSpaceModel interface ---

    def transition(self, x, noise=None):
        """
        Static target: state does not change (x_t = x_{t-1}).

        For the Dai22 static bearing-only problem, there is no dynamics.
        The model is used as a single-step prior-to-posterior update.

        Args:
            x (tf.Tensor): Current state of shape (N, 2).
            noise (tf.Tensor, optional): Unused.

        Returns:
            tf.Tensor: Same state of shape (N, 2).
        """
        return x

    def observation(self, x, noise=None):
        """
        Generate noisy bearing observation: z = h(x) + noise.

        Args:
            x (tf.Tensor): State of shape (N, 2).
            noise (tf.Tensor, optional): Normal noise of shape (N, 2).

        Returns:
            tf.Tensor: Noisy bearings of shape (N, 2).
        """
        h_x = self.observation_mean(x)
        if noise is None:
            noise = tf.random.normal(tf.shape(h_x), dtype=tf.float32)
        chol_R = tf.linalg.cholesky(self.R)
        return h_x + tf.linalg.matvec(chol_R, noise)

    def observation_mean(self, x):
        """
        Deterministic observation h(x) = [atan2(...), atan2(...)].

        Alias for measurement_function.

        Args:
            x (tf.Tensor): State of shape (..., 2).

        Returns:
            tf.Tensor: Predicted bearings of shape (..., 2).
        """
        return self.measurement_function(x)

    def measurement_function(self, x):
        """
        h(x) = [atan2(y - y1, x - x1), atan2(y - y2, x - x2)]^T
        
        CRITICAL: Use atan2, not atan, to handle all quadrants correctly
        """
        xt, yt = x[..., 0], x[..., 1]
        
        # Compute bearings using atan2 (handles all quadrants)
        h1 = tf.math.atan2(yt - self.sensors[0, 1], xt - self.sensors[0, 0])
        h2 = tf.math.atan2(yt - self.sensors[1, 1], xt - self.sensors[1, 0])
        
        return tf.stack([h1, h2], axis=-1)
    
    def log_prior(self, x):
        """Log-prior density: log p_0(x) = -0.5 * (x - mu)^T Σ^{-1} (x - mu) + const"""
        diff = x - self.prior_mean
        quad_form = tf.reduce_sum(diff * tf.linalg.matvec(self.inv_prior_cov, diff), axis=-1)
        return -0.5 * quad_form
    
    def log_likelihood(self, y, x):
        """
        Log-likelihood: log p(y|x) = -0.5 * (y - h(x))^T R^{-1} (y - h(x)) + const

        Args:
            y: Observations (bearings) of shape (..., 2)
            x: States (positions) of shape (..., 2)

        Note: Handles angle wrapping for residuals
        """
        h_x = self.measurement_function(x)
        residual = y - h_x
        
        # Handle angle wrapping: map to [-π, π]
        residual = tf.math.atan2(tf.sin(residual), tf.cos(residual))
        
        quad_form = tf.reduce_sum(residual * tf.linalg.matvec(self.inv_R, residual), axis=-1)
        return -0.5 * quad_form
    
    def hessian_log_prior(self):
        """
        Hessian of log-prior is constant: ∇∇^T log p_0 = -Σ^{-1}
        """
        return -self.inv_prior_cov
    
    def gradient_log_likelihood(self, x, z):
        """
        Gradient of log-likelihood using automatic differentiation.
        ∇_x log p(z|x) = H^T R^{-1} (z - h(x))
        where H is the Jacobian of h(x)
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            log_lik = self.log_likelihood(z, x)

        grad = tape.gradient(log_lik, x)
        return grad

    def hessian_log_likelihood_numerical(self, x, z, eps=1e-4):
        """
        Numerical Hessian of log-likelihood using finite differences.
        More stable than automatic differentiation for this nonlinear problem.
        
        Returns: ∇∇^T log h(x)  [shape: (2, 2)]
        """
        x_np = x.numpy() if isinstance(x, tf.Tensor) else x
        n = len(x_np)
        hess = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Four-point stencil for second derivative
                x_pp = x_np.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                
                x_pm = x_np.copy()
                x_pm[i] += eps
                x_pm[j] -= eps
                
                x_mp = x_np.copy()
                x_mp[i] -= eps
                x_mp[j] += eps
                
                x_mm = x_np.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps
                
                # Evaluate log-likelihood at four points
                f_pp = self.log_likelihood(z, tf.constant(x_pp, dtype=tf.float32)).numpy()
                f_pm = self.log_likelihood(z, tf.constant(x_pm, dtype=tf.float32)).numpy()
                f_mp = self.log_likelihood(z, tf.constant(x_mp, dtype=tf.float32)).numpy()
                f_mm = self.log_likelihood(z, tf.constant(x_mm, dtype=tf.float32)).numpy()
                
                # Central difference formula for second derivative
                hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
        
        return tf.constant(hess, dtype=tf.float32)
    
    def hessian_log_likelihood_analytical(self, x, z):
        """
        Analytical Hessian approximation using Gauss-Newton:
        ∇∇^T log p(z|x) ≈ -H^T R^{-1} H
        
        where H is the Jacobian of h(x) at x.
        
        For bearing-only tracking:
        h_i = atan2(y - y_i, x - x_i)
        
        ∂h_i/∂x = (-(y - y_i) / r_i^2, (x - x_i) / r_i^2)
        where r_i^2 = (x - x_i)^2 + (y - y_i)^2
        """
        # Handle both numpy arrays and tensors
        if isinstance(x, tf.Tensor):
            x_val = x.numpy()
        else:
            x_val = x
        
        # Ensure x is 1D array
        if x_val.ndim == 0:
            raise ValueError("x must be a vector, not a scalar")
        
        xt, yt = float(x_val[0]), float(x_val[1])
        
        jacobian = []
        for i in range(self.sensors.shape[0]):
            sx = float(self.sensors[i, 0])
            sy = float(self.sensors[i, 1])
            dx = xt - sx
            dy = yt - sy
            r_sq = dx**2 + dy**2
            
            # Jacobian row: [∂h_i/∂x, ∂h_i/∂y]
            dh_dx = -dy / r_sq
            dh_dy = dx / r_sq
            jacobian.append([dh_dx, dh_dy])
        
        H = tf.constant(jacobian, dtype=tf.float32)
        
        # Gauss-Newton Hessian approximation
        hess = -tf.linalg.matmul(tf.transpose(H), tf.linalg.matmul(self.inv_R, H))
        
        return hess
    
    def generate_measurement(self, add_noise=True, seed=None):
        """
        Generate measurement from true target location.
        Returns the sample measurement used in Dai22 paper.
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
        
        # True measurement (noiseless)
        z_true = self.measurement_function(self.target_truth)
        
        if add_noise:
            noise = tf.random.normal([2], mean=0.0, stddev=tf.sqrt(0.04), seed=seed)
            z = z_true + noise
        else:
            z = z_true
        
        return z


class BearingOnlyScenario:
    """
    Complete experimental setup from Dai22 Section 4.
    Provides convenient methods for running experiments.
    """
    
    def __init__(self):
        self.model = BearingOnlyTrackingModel()
        
        # Sample measurement from paper [Dai22, p.12]
        self.z_sample = tf.constant([0.4754, 1.1868], dtype=tf.float32)
    
    def sample_prior(self, n_samples=1, seed=None):
        """Sample particles from prior distribution"""
        if seed is not None:
            tf.random.set_seed(seed)
        
        samples = tf.random.normal(
            [n_samples, 2],
            mean=self.model.prior_mean,
            stddev=tf.sqrt(tf.linalg.diag_part(self.model.prior_cov)),
            seed=seed
        )
        
        return samples
    
    def compute_metrics(self, particles):
        """
        Compute performance metrics: MSE and trace(P)
        
        Args:
            particles: [n_particles, 2] tensor
            
        Returns:
            mse: Mean squared error
            trace_cov: Trace of covariance matrix
        """
        # Posterior mean
        posterior_mean = tf.reduce_mean(particles, axis=0)
        
        # MSE relative to true target
        error = particles - self.model.target_truth
        mse = tf.reduce_mean(tf.reduce_sum(tf.square(error), axis=1))
        
        # Covariance matrix
        centered = particles - posterior_mean
        n_particles = tf.cast(tf.shape(particles)[0], tf.float32)
        cov = tf.linalg.matmul(centered, centered, transpose_a=True) / (n_particles - 1.0)
        trace_cov = tf.linalg.trace(cov)
        
        return mse.numpy(), trace_cov.numpy(), posterior_mean.numpy()