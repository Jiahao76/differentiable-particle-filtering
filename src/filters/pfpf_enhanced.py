"""
Enhanced PFPF (Particle Flow Particle Filter) with Dai22 Optimal Homotopy Support

This module provides improved PF-PF implementations that support:
1. Standard linear homotopy: β(λ) = λ (Li 2017 baseline)
2. Optimal homotopy: β*(λ) from solving TPBVP (Dai 2022)

The key innovation: Using Dai22's optimal homotopy as the proposal distribution
for Li17's PF-PF framework should improve performance by providing a better-conditioned
particle flow with lower variance in the proposal distribution.
"""

import tensorflow as tf
import numpy as np


class PFPF_LEDH_Enhanced:
    """
    Enhanced PF-PF with LEDH Flow supporting optional Dai22 optimal homotopy.
    
    Compared to standard PFPF_LEDH:
    - Supports both linear β(λ) = λ and optimal β*(λ)
    - Automatic integration of optimal homotopy from Dai22 optimizer
    - Performance comparison infrastructure
    """
    
    def __init__(self, model, num_particles=100, flow_steps=20, step_size=0.05,
                 obs_noise_var=1.0):
        """Initialize the enhanced PFPF-LEDH filter.

        Args:
            model: State-space model with ``observation_mean``.
            num_particles: Number of particles.
            flow_steps: Number of discretization steps.
            step_size: Euler step size.
            obs_noise_var: Observation noise variance *R*.
        """
        self.model = model
        self.num_particles = num_particles
        self.flow_steps = flow_steps
        self.epsilon = step_size
        self.R = obs_noise_var

        # Track beta function used
        self.beta_func_used = None

    def run(self, observations, beta_func=None):
        """Run PF-PF (LEDH) with optional optimal homotopy.

        Args:
            observations: Tensor of shape ``(T, 1)``.
            beta_func: Optional homotopy function returning
                ``(alpha, beta, alpha_dot, beta_dot)``.

        Returns:
            Tuple of ``(estimates, metadata)``.
        """
        # Use linear homotopy if beta_func not provided
        if beta_func is None:
            beta_func = self._linear_homotopy
        
        self.beta_func_used = beta_func
        
        T = tf.shape(observations)[0]
        state_dim = 1
        
        # Initialize
        particles = tf.random.normal((self.num_particles, state_dim), dtype=tf.float32)
        weights = tf.ones((self.num_particles,), dtype=tf.float32) / float(self.num_particles)
        estimates = tf.TensorArray(dtype=tf.float32, size=T, clear_after_read=False)
        
        # Track statistics
        metadata = {
            'ess_history': [],
            'weight_variance': [],
            'particles_std': []
        }
        
        for t in range(T):
            observation = observations[t]
            
            # ===== PREDICTION =====
            particles_0 = self.model.transition(particles)
            
            # Shared predictive covariance P
            eta_mean = tf.reduce_mean(particles_0, axis=0, keepdims=True)
            P = tf.reduce_mean((particles_0 - eta_mean)**2)
            
            # ===== PARTICLE FLOW WITH OPTIONAL OPTIMAL HOMOTOPY =====
            current_particles = tf.identity(particles_0)
            log_det_sum = tf.zeros((self.num_particles,), dtype=tf.float32)
            
            # Keep P fixed from initial state
            P_fixed = P
            
            # Integrate particle flow
            for j in range(self.flow_steps):
                lam = float(j + 1) / float(self.flow_steps)

                # Get homotopy parameters from beta_func
                alpha, beta, alpha_dot, beta_dot = beta_func(tf.constant(lam, dtype=tf.float32))
                beta_scalar = float(beta.numpy())
                
                # Compute local H_i for each particle
                with tf.GradientTape() as tape:
                    tape.watch(current_particles)
                    h_val = self.model.observation_mean(current_particles)

                H = tape.gradient(h_val, current_particles)
                
                # Flow parameters with Dai22 optimal homotopy
                denom = beta_scalar * (H**2) * P_fixed + self.R
                A = -0.5 * P_fixed * (H**2) / (denom + 1e-8)
                
                e = h_val - H * current_particles
                innovation = observation - e
                
                # Note: Using beta_scalar from optimal homotopy instead of lam
                factor1 = (1.0 + 2.0 * beta_scalar * A)
                factor2 = (1.0 + beta_scalar * A)
                factor3 = P_fixed * H / (self.R + 1e-8)
                b = factor1 * factor2 * factor3 * innovation + A * current_particles
                
                # Jacobian determinant
                log_det_sum += tf.math.log(
                    tf.abs(1.0 + self.epsilon * tf.reshape(A, [-1])) + 1e-8
                )
                
                # Euler step
                particles_new = current_particles + self.epsilon * (A * current_particles + b)
                particles_new = tf.clip_by_value(particles_new, -10.0, 10.0)
                current_particles = particles_new
            
            particles_1 = current_particles
            
            # ===== WEIGHT UPDATE (Li 2017) =====
            log_lik = self.model.log_likelihood(observation, particles_1)
            log_lik = tf.reshape(log_lik, (-1,))
            
            prior_1 = self.model.transition_log_pdf(particles_1, particles)
            prior_0 = self.model.transition_log_pdf(particles_0, particles)
            prior_ratio = prior_1 - prior_0
            
            log_weights = (tf.math.log(weights + 1e-16) + 
                          prior_ratio + log_lik + log_det_sum)
            
            # Normalize weights
            max_log_w = tf.reduce_max(log_weights)
            log_weights_shifted = log_weights - max_log_w
            weights_exp = tf.exp(log_weights_shifted)
            weights = weights_exp / (tf.reduce_sum(weights_exp) + 1e-16)
            
            # Estimate
            estimate = tf.reduce_sum(particles_1 * tf.reshape(weights, (-1, 1)), axis=0)
            estimates = estimates.write(t, estimate)
            
            # Track statistics
            ess = 1.0 / tf.reduce_sum(weights ** 2)
            metadata['ess_history'].append(float(ess.numpy()))
            metadata['weight_variance'].append(float(tf.reduce_mean((weights - 1.0/self.num_particles)**2).numpy()))
            metadata['particles_std'].append(float(tf.math.reduce_std(particles_1).numpy()))
            
            # Resample if ESS too low
            if ess < self.num_particles / 2:
                indices = tf.random.categorical(
                    tf.math.log(tf.reshape(weights, (1, -1)) + 1e-16),
                    self.num_particles,
                    dtype=tf.int32
                )[0]
                particles = tf.gather(particles_1, indices)
                weights = tf.ones((self.num_particles,), dtype=tf.float32) / float(self.num_particles)
            else:
                particles = particles_1
        
        return estimates.stack(), metadata
    
    @staticmethod
    def _linear_homotopy(lam):
        """Default linear homotopy β(λ) = λ"""
        lam_t = tf.cast(lam, tf.float32)
        alpha = 1.0 - lam_t
        beta = lam_t
        alpha_dot = tf.constant(-1.0, dtype=tf.float32)
        beta_dot = tf.constant(1.0, dtype=tf.float32)
        return alpha, beta, alpha_dot, beta_dot


class PFPF_EDH_Enhanced:
    """
    Enhanced PF-PF with EDH Flow supporting optional Dai22 optimal homotopy.
    
    For EDH, the Jacobian determinant cancels out in weight update,
    so the main benefit of optimal homotopy is improved numerical stability
    and better proposal distribution conditioning.
    """
    
    def __init__(self, model, num_particles=100, flow_steps=20, step_size=0.05,
                 obs_noise_var=1.0):
        """Initialize the enhanced PFPF-EDH filter.

        Args:
            model: State-space model with ``observation_mean``.
            num_particles: Number of particles.
            flow_steps: Number of discretization steps.
            step_size: Euler step size.
            obs_noise_var: Observation noise variance *R*.
        """
        self.model = model
        self.num_particles = num_particles
        self.flow_steps = flow_steps
        self.epsilon = step_size
        self.R = obs_noise_var

        self.beta_func_used = None
    
    def run(self, observations, beta_func=None):
        """
        Run PF-PF (EDH) with optional optimal homotopy.
        
        Args:
            observations: Tensor of shape (T, 1)
            beta_func: Optional function β(λ) that returns (α, β, α_dot, β_dot)
                      If None, uses linear homotopy β(λ) = λ
            
        Returns:
            estimates: Estimated states
            metadata: Dict with filter statistics
        """
        if beta_func is None:
            beta_func = self._linear_homotopy
        
        self.beta_func_used = beta_func
        
        T = tf.shape(observations)[0]
        state_dim = 1
        
        # Initialize
        particles = tf.random.normal((self.num_particles, state_dim), dtype=tf.float32)
        weights = tf.ones((self.num_particles,), dtype=tf.float32) / float(self.num_particles)
        estimates = tf.TensorArray(dtype=tf.float32, size=T, clear_after_read=False)
        
        metadata = {
            'ess_history': [],
            'weight_variance': [],
            'particles_std': []
        }
        
        for t in range(T):
            observation = observations[t]
            
            # ===== PREDICTION =====
            particles_0 = self.model.transition(particles)
            
            # Ensemble statistics
            eta_mean = tf.reduce_mean(particles_0, axis=0, keepdims=True)
            P = tf.reduce_mean((particles_0 - eta_mean)**2)
            
            # ===== PARTICLE FLOW (all particles use same flow parameters) =====
            current_particles = tf.identity(particles_0)
            
            for j in range(self.flow_steps):
                lam = float(j + 1) / float(self.flow_steps)

                # Get optimal homotopy parameters
                alpha, beta, alpha_dot, beta_dot = beta_func(tf.constant(lam, dtype=tf.float32))
                beta_scalar = float(beta.numpy())
                
                # Compute flow parameters at ensemble mean (EDH)
                with tf.GradientTape() as tape:
                    tape.watch(eta_mean)
                    h_mean = self.model.observation_mean(eta_mean)
                
                H = tape.gradient(h_mean, eta_mean)
                H_scalar = tf.reshape(H, [])
                P_scalar = tf.reshape(P, [])
                
                # Flow parameters using optimal beta
                denom = beta_scalar * (H_scalar**2) * P_scalar + self.R
                A = -0.5 * P_scalar * (H_scalar**2) / (denom + 1e-8)
                
                e_beta = h_mean - H_scalar * eta_mean
                innovation = observation - e_beta
                
                factor1 = (1.0 + 2.0 * beta_scalar * A)
                factor2 = (1.0 + beta_scalar * A)
                factor3 = P_scalar * H_scalar / self.R
                
                b = factor1 * factor2 * factor3 * innovation + A * eta_mean
                
                # Euler step for all particles
                particles_new = (current_particles + 
                               self.epsilon * (A * current_particles + b))
                particles_new = tf.clip_by_value(particles_new, -10.0, 10.0)
                current_particles = particles_new
                
                # Update ensemble mean
                eta_mean = tf.reduce_mean(current_particles, axis=0, keepdims=True)
            
            particles_1 = current_particles
            
            # ===== WEIGHT UPDATE (EDH: Jacobian cancels) =====
            log_lik = self.model.log_likelihood(observation, particles_1)
            log_lik = tf.reshape(log_lik, (-1,))
            
            prior_1 = self.model.transition_log_pdf(particles_1, particles)
            prior_0 = self.model.transition_log_pdf(particles_0, particles)
            prior_ratio = prior_1 - prior_0
            
            # No Jacobian term for EDH (cancels out)
            log_weights = tf.math.log(weights + 1e-16) + prior_ratio + log_lik
            
            # Normalize
            max_log_w = tf.reduce_max(log_weights)
            log_weights_shifted = log_weights - max_log_w
            weights_exp = tf.exp(log_weights_shifted)
            weights = weights_exp / (tf.reduce_sum(weights_exp) + 1e-16)
            
            # Estimate
            estimate = tf.reduce_sum(particles_1 * tf.reshape(weights, (-1, 1)), axis=0)
            estimates = estimates.write(t, estimate)
            
            # Statistics
            ess = 1.0 / tf.reduce_sum(weights ** 2)
            metadata['ess_history'].append(float(ess.numpy()))
            metadata['weight_variance'].append(float(tf.reduce_mean((weights - 1.0/self.num_particles)**2).numpy()))
            metadata['particles_std'].append(float(tf.math.reduce_std(particles_1).numpy()))
            
            # Resample
            if ess < self.num_particles / 2:
                indices = tf.random.categorical(
                    tf.math.log(tf.reshape(weights, (1, -1)) + 1e-16),
                    self.num_particles,
                    dtype=tf.int32
                )[0]
                particles = tf.gather(particles_1, indices)
                weights = tf.ones((self.num_particles,), dtype=tf.float32) / float(self.num_particles)
            else:
                particles = particles_1
        
        return estimates.stack(), metadata
    
    @staticmethod
    def _linear_homotopy(lam):
        """Default linear homotopy β(λ) = λ"""
        lam_t = tf.cast(lam, tf.float32)
        alpha = 1.0 - lam_t
        beta = lam_t
        alpha_dot = tf.constant(-1.0, dtype=tf.float32)
        beta_dot = tf.constant(1.0, dtype=tf.float32)
        return alpha, beta, alpha_dot, beta_dot
