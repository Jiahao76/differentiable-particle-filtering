"""
Exact Daum-Huang (EDH) Flow Filter
Based on Daum & Huang (2010, 2011)

Reference:
[Daum(10)] Daum, Fred, Jim Huang, and Arjang Noushin. 
"Exact particle flow for nonlinear filters." SPIE, 2010.
"""
import tensorflow as tf
import numpy as np


class EDHFlowFilter:
    """Exact Daum-Huang (EDH) Particle Flow Filter.

    All particles use the SAME flow parameters computed at the ensemble mean.
    Generic: works with any model that provides ``observation_mean()``.
    """

    def __init__(self, model, num_particles=100, flow_steps=20, step_size=0.05,
                 obs_noise_var=1.0):
        """Initialize the EDH flow filter.

        Args:
            model: State-space model with ``transition`` and ``observation_mean``.
            num_particles: Number of particles.
            flow_steps: Number of discretization steps in pseudo-time.
            step_size: Euler step size.
            obs_noise_var: Observation noise variance *R* used in the flow.
        """
        self.model = model
        self.num_particles = num_particles
        self.flow_steps = flow_steps
        self.epsilon = step_size
        self.R = obs_noise_var
    
    def compute_flow_parameters(self, eta_mean, P, observation, lambda_val):
        """
        Compute EDH flow parameters A(λ) and b(λ)
        
        Equations (10)-(11) from Daum & Huang (2010):
        A(λ) = -0.5 * P * H^T * (λ*H*P*H^T + R)^{-1} * H
        b(λ) = (I + 2λA)(I + λA) * P * H^T * R^{-1} * (z - e(λ)) + A * η̄
        
        where H = ∂h/∂x evaluated at η̄, and e(λ) = h(η̄) - H*η̄
        
        Args:
            eta_mean: Ensemble mean (where to linearize)
            P: Predictive covariance
            observation: Current observation z
            lambda_val: Current pseudo-time λ
        
        Returns:
            A: Flow coefficient matrix
            b: Flow offset vector
        """
        with tf.GradientTape() as tape:
            tape.watch(eta_mean)
            h_mean = self.model.observation_mean(eta_mean)

        H = tape.gradient(h_mean, eta_mean)
        
        # For 1D case, simplify matrix operations
        H_scalar = tf.reshape(H, [])
        P_scalar = tf.reshape(P, [])
        
        # Compute A(λ)
        denominator = lambda_val * (H_scalar ** 2) * P_scalar + self.R
        A = -0.5 * P_scalar * (H_scalar ** 2) / (denominator + 1e-8)
        
        # Compute b(λ)
        # e(λ) = h(η̄) - H*η̄
        e_lambda = h_mean - H_scalar * eta_mean
        
        # b = (I + 2λA)(I + λA) * P * H^T * R^{-1} * (z - e) + A*η̄
        innovation = observation - e_lambda
        factor1 = (1.0 + 2.0 * lambda_val * A) * (1.0 + lambda_val * A)
        factor2 = P_scalar * H_scalar / self.R
        b = factor1 * factor2 * innovation + A * eta_mean
        
        return A, b
    
    def run(self, observations):
        """
        Run EDH flow filter for a sequence of observations
        
        Args:
            observations: Tensor of shape (T, 1)
        
        Returns:
            estimates: Estimated states at each time step
        """
        T = tf.shape(observations)[0]
        state_dim = 1
        
        # Initialize particles
        particles = tf.random.normal((self.num_particles, state_dim), dtype=tf.float32)
        estimates = tf.TensorArray(dtype=tf.float32, size=T, clear_after_read=False)
        
        for t in range(T):
            observation = observations[t]
            
            # ===== PREDICTION =====
            particles = self.model.transition(particles)
            
            # Compute ensemble statistics
            eta_mean = tf.reduce_mean(particles, axis=0, keepdims=True)
            centered = particles - eta_mean
            P = tf.reduce_mean(centered ** 2)  # Scalar covariance for 1D
            
            # ===== PARTICLE FLOW =====
            # All particles use the SAME flow parameters (computed at ensemble mean)
            current_particles = tf.identity(particles)
            current_mean = tf.identity(eta_mean)
            
            lambda_val = 0.0
            for j in range(self.flow_steps):
                lambda_val += self.epsilon
                
                # Compute flow parameters at current ensemble mean
                A, b = self.compute_flow_parameters(current_mean, P, observation, lambda_val)
                
                # Flow update: η_{j+1} = η_j + ε * (A * η_j + b)
                # All particles move with the SAME A and b
                current_particles = current_particles + self.epsilon * (A * current_particles + b)
                current_mean = current_mean + self.epsilon * (A * current_mean + b)
            
            # ===== ESTIMATION =====
            # Simple mean estimate (no weights needed for pure flow filter)
            estimate = tf.reduce_mean(current_particles, axis=0)
            estimates = estimates.write(t, estimate)
            
            # Update particles for next iteration
            particles = current_particles
        
        return estimates.stack()