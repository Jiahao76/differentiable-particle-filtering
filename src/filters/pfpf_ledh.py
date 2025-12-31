"""
Particle Flow Particle Filter (PF-PF) with LEDH Flow
Reference: Li & Coates (2017) "Particle Filtering with Invertible Particle Flow"
"""
import tensorflow as tf
import numpy as np

class PFPF_LEDH:
    def __init__(self, model, num_particles=100, flow_steps=20, step_size=0.05):
        self.model = model
        self.num_particles = num_particles
        self.flow_steps = flow_steps
        self.epsilon = step_size
        self.beta = model.beta
        self.R = 1.0  # Observation noise variance
    
    def run(self, observations):
        T = tf.shape(observations)[0]
        state_dim = 1
        
        # Initialize particles and weights
        particles = tf.random.normal((self.num_particles, state_dim), dtype=tf.float32)
        weights = tf.ones((self.num_particles,), dtype=tf.float32) / float(self.num_particles)
        estimates = tf.TensorArray(dtype=tf.float32, size=T)
        ess_history = []  # Track ESS for analysis
        
        for t in range(T):
            observation = observations[t]
            
            # ===== PREDICTION =====
            # eta_0 in Li(2017)
            particles_0 = self.model.transition(particles)
            
            # Shared predictive covariance P
            eta_mean = tf.reduce_mean(particles_0, axis=0)
            P = tf.reduce_mean((particles_0 - eta_mean)**2)
            
            # ===== INVERTIBLE PARTICLE FLOW (Proposal Generation) =====
            current_particles = tf.identity(particles_0)
            # Log of product of determinants: sum(log|det(I + eps*A)|) 
            log_det_sum = tf.zeros((self.num_particles,), dtype=tf.float32)
            
            # CRITICAL: Compute P ONCE and KEEP IT FIXED
            # P is based on initial (eta_0) particles
            eta_mean_0 = tf.reduce_mean(particles_0, axis=0)
            P_fixed = tf.reduce_mean((particles_0 - eta_mean_0)**2)
            
            lambda_val = 0.0
            for j in range(self.flow_steps):
                lambda_val += self.epsilon
                
                # Vectorized computation of local H_i for each particle
                # H computed at CURRENT particle position (local linearization)
                with tf.GradientTape() as tape:
                    tape.watch(current_particles)
                    h_val = self.beta * tf.exp(current_particles / 2.0)
                
                H = tape.gradient(h_val, current_particles) # Shape: (N, 1)
                
                # A_i(lambda) for each particle - using FIXED P
                denom = lambda_val * (H**2) * P_fixed + self.R
                A = -0.5 * P_fixed * (H**2) / (denom + 1e-8)
                
                # b_i(lambda) for each particle
                e = h_val - H * current_particles
                innovation = observation - e
                factor1 = (1.0 + 2.0 * lambda_val * A)
                factor2 = (1.0 + lambda_val * A)
                factor3 = P_fixed * H / (self.R + 1e-8)
                b = factor1 * factor2 * factor3 * innovation + A * current_particles
                
                # Accumulate Jacobian determinant for 1D: det(I + eps*A) = 1 + eps*A
                log_det_sum += tf.math.log(tf.abs(1.0 + self.epsilon * tf.reshape(A, [-1])) + 1e-8)
                
                # Euler integration step
                particles_new = current_particles + self.epsilon * (A * current_particles + b)
                
                # Clipping for numerical stability
                particles_new = tf.clip_by_value(particles_new, -10.0, 10.0)
                current_particles = particles_new
            
            # η_1 in Li(2017)
            particles_1 = current_particles
            
            # ===== WEIGHT UPDATE (Li 2017, Eq. 18 & 20) =====
            # p(z|η_1)
            log_lik = self.model.log_likelihood(observation, particles_1)
            log_lik = tf.reshape(log_lik, (-1,))
            
            # Compute prior terms: p(η_1|x_prev) / p(η_0|x_prev)
            # For SV model: these are transition log-probabilities
            # Note: For random walk (alpha ~ 1), prior_ratio should be close to 0
            prior_1 = self.model.transition_log_pdf(particles_1, particles)  # p(η_1|x_prev)
            prior_0 = self.model.transition_log_pdf(particles_0, particles)  # p(η_0|x_prev)
            prior_ratio = prior_1 - prior_0
            
            # Final log weight: log(w_prev) + (prior_1 - prior_0) + log_likelihood + log_jacobian_det
            # This is the complete weight update from Li(2017) [Eq. 18, 20]
            log_weights = tf.math.log(weights + 1e-16) + prior_ratio + log_lik + log_det_sum
            
            # Normalize weights with numerical stability
            log_max = tf.reduce_max(log_weights)
            log_weights_stable = log_weights - log_max
            weights = tf.exp(log_weights_stable)
            weights = weights / (tf.reduce_sum(weights) + 1e-16)
            
            # ===== ESTIMATION & RESAMPLING =====
            estimate = tf.reduce_sum(particles_1 * tf.reshape(weights, (-1, 1)), axis=0)
            estimates = estimates.write(t, estimate)
            
            # Effective Sample Size (ESS) [cite: 987]
            ess = 1.0 / tf.reduce_sum(weights**2)
            ess_history.append(ess.numpy())  # Record ESS
            if ess < self.num_particles / 2.0:
                indices = tf.random.categorical(tf.math.log(weights[None, :] + 1e-16), self.num_particles)
                particles = tf.gather(particles_1, indices[0])
                weights = tf.ones((self.num_particles,)) / float(self.num_particles)
            else:
                particles = particles_1
                
        avg_ess = np.mean(ess_history) if ess_history else np.nan
        return estimates.stack(), avg_ess