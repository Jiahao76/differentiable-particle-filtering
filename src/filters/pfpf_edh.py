"""
Particle Flow Particle Filter (PF-PF) with EDH Flow
Based on Li & Coates (2017)

Reference:
[Li(17)] Li, Yunpeng, and Mark Coates. 
"Particle filtering with invertible particle flow." 
IEEE Transactions on Signal Processing, 2017.

Key contribution: Invertible mapping property allows efficient weight update
"""
import tensorflow as tf
import numpy as np


class PFPF_EDH:
    """
    PF-PF with EDH (Exact Daum-Huang) Flow
    
    Key innovations:
    1. EDH flow creates proposal distribution close to posterior
    2. Invertible mapping allows efficient weight update
    3. For EDH, Jacobian determinant CANCELS OUT in weight update!
    
    Weight update (Equation 37 in Li 2017):
    w_k ∝ [p(η₁|x_{k-1}) * p(z_k|η₁)] / p(η₀|x_{k-1}) * w_{k-1}
    
    No Jacobian determinant calculation needed!
    """
    
    def __init__(self, model, num_particles=100, flow_steps=20, step_size=0.05,
                 obs_noise_var=1.0):
        self.model = model
        self.num_particles = num_particles
        self.flow_steps = flow_steps
        self.epsilon = step_size
        self.R = obs_noise_var
    
    def compute_flow_parameters(self, eta_mean, P, observation, lambda_val):
        """Compute EDH flow parameters (same as EDH filter)."""
        with tf.GradientTape() as tape:
            tape.watch(eta_mean)
            h_mean = self.model.observation_mean(eta_mean)
        
        H = tape.gradient(h_mean, eta_mean)
        H_scalar = tf.reshape(H, [])
        P_scalar = tf.reshape(P, [])
        
        # A(λ)
        denom = lambda_val * (H_scalar ** 2) * P_scalar + self.R
        A = -0.5 * P_scalar * (H_scalar ** 2) / (denom + 1e-8)
        
        # b(λ)
        e_lambda = h_mean - H_scalar * eta_mean
        innovation = observation - e_lambda
        factor1 = (1.0 + 2.0 * lambda_val * A) * (1.0 + lambda_val * A)
        factor2 = P_scalar * H_scalar / self.R
        b = factor1 * factor2 * innovation + A * eta_mean
        
        return A, b
    
    def run(self, observations):
        """
        Run PF-PF (EDH) filter
        
        Args:
            observations: Tensor of shape (T, 1)
        
        Returns:
            estimates: Estimated states
        """
        T = tf.shape(observations)[0]
        state_dim = 1
        
        # Initialize
        particles = tf.random.normal((self.num_particles, state_dim), dtype=tf.float32)
        weights = tf.ones((self.num_particles,), dtype=tf.float32) / float(self.num_particles)
        estimates = tf.TensorArray(dtype=tf.float32, size=T, clear_after_read=False)
        ess_history = []  # Track ESS for analysis
        
        for t in range(T):
            observation = observations[t]
            
            # ===== PREDICTION =====
            particles_before_flow = self.model.transition(particles)
            
            # Compute statistics
            eta_mean = tf.reduce_mean(particles_before_flow, axis=0, keepdims=True)
            centered = particles_before_flow - eta_mean
            P = tf.reduce_mean(centered ** 2)
            
            # ===== PARTICLE FLOW (Proposal Generation) =====
            current_particles = tf.identity(particles_before_flow)
            current_mean = tf.identity(eta_mean)
            
            lambda_val = 0.0
            for j in range(self.flow_steps):
                lambda_val += self.epsilon
                
                A, b = self.compute_flow_parameters(current_mean, P, observation, lambda_val)
                
                # Flow update (all particles use same A, b)
                current_particles = current_particles + self.epsilon * (A * current_particles + b)
                current_mean = current_mean + self.epsilon * (A * current_mean + b)
            
            particles_after_flow = current_particles
            
            # ===== WEIGHT UPDATE (Li 2017, Eq. 37) =====
            # For PF-PF (EDH), Jacobian determinant cancels!
            # w_k ∝ p(η₁|x_{k-1}) * p(z_k|η₁) / p(η₀|x_{k-1})
            
            # Log-likelihood at flowed positions
            log_likelihood = self.model.log_likelihood(observation, particles_after_flow)
            log_likelihood = tf.reshape(log_likelihood, (-1,))
            
            # Compute prior terms: p(η₁|x_prev) / p(η₀|x_prev)
            # For SV model: these are transition log-probabilities
            prior_1 = self.model.transition_log_pdf(particles_after_flow, particles)  # p(η_1|x_prev)
            prior_0 = self.model.transition_log_pdf(particles_before_flow, particles)  # p(η_0|x_prev)
            prior_ratio = prior_1 - prior_0
            
            # Update weights with full weight formula including prior ratio
            log_weights = tf.math.log(weights + 1e-16) + prior_ratio + log_likelihood
            
            # Normalize with numerical stability
            log_max = tf.reduce_max(log_weights)
            log_weights_stable = log_weights - log_max
            weights = tf.exp(log_weights_stable)
            weights = weights / (tf.reduce_sum(weights) + 1e-16)
            
            # ===== ESTIMATION =====
            estimate = tf.reduce_sum(particles_after_flow * tf.reshape(weights, (-1, 1)), axis=0)
            estimates = estimates.write(t, estimate)
            
            # ===== RESAMPLING =====
            eff_N = 1.0 / tf.reduce_sum(weights ** 2)
            ess_history.append(eff_N.numpy())  # Record ESS
            if eff_N < self.num_particles / 2.0:
                indices = tf.random.categorical(
                    tf.reshape(tf.math.log(weights + 1e-16), (1, -1)),
                    self.num_particles
                )
                particles = tf.gather(particles_after_flow, indices[0])
                weights = tf.ones_like(weights) / float(self.num_particles)
            else:
                particles = particles_after_flow
        
        avg_ess = np.mean(ess_history) if ess_history else np.nan
        return estimates.stack(), avg_ess