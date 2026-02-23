import tensorflow as tf
import numpy as np
from src.models.base_model import StateSpaceModel

class InvertibleFlowParticleFilter:
    """
    Invertible Particle Flow-based Particle Filter (Optimized for SV Model).
    
    KEY FIX: Uses 'Log-Squared' observation transformation for the Flow step.
    This linearizes the highly non-linear SV observation (y = beta*exp(x/2)*v),
    stabilizing gradients and preventing particle overshooting.
    """
    
    def __init__(self, model: StateSpaceModel, num_particles=1000, flow_steps=20, step_size=0.05):
        self.model = model
        self.num_particles = num_particles
        self.flow_steps = flow_steps
        self.epsilon = step_size

    def compute_gradients_log_domain(self, particles):
        """
        Compute Jacobian H using the Log-Squared transform.
        Original: y = beta * exp(x/2)
        Log-Sq:   log(y^2) = log(beta^2) + x
        
        This makes the gradient H = 1 (constant), which is extremely stable.
        """
        beta = self.model.beta

        # Theoretical h(x) in log-squared domain: h(x) = log(beta^2) + x
        # Note: We ignore the noise log(v^2) term for the deterministic gradient direction
        h_x_log = tf.math.log(beta**2 + 1e-8) + particles

        # Gradient is trivially 1.0, but we use AutoDiff to be generic
        with tf.GradientTape() as tape:
            tape.watch(particles)
            output = tf.math.log(beta**2 + 1e-8) + particles
        H = tape.gradient(output, particles)
        
        return h_x_log, H

    @tf.function
    def apply_edh_flow(self, particles, P, R, z_meas):
        """
        Executes one step of the EDH Flow in Log-Squared Domain.
        """
        # 1. Transform Measurement to Log-Squared Domain
        # z_meas is raw y. We need log(y^2).
        z_log = tf.math.log(z_meas**2 + 1e-8)
        
        # 2. Compute Gradients in Log Domain
        h_x, H = self.compute_gradients_log_domain(particles)
        
        # 3. Error term in Log Domain
        error = h_x - z_log
        
        # 4. Standard EDH Velocity Calculation
        # H is (N, 1), P is (1, 1) scalar cov
        H_mean = tf.reduce_mean(H, axis=0)
        H_mean = tf.reshape(H_mean, [1, 1])
        
        # In log domain, H is approx 1. This term is very stable.
        term_inv = 1.0 / (R + H_mean**2 * P)
        
        # Velocity v = -0.5 * P * H * inv * error
        velocity = -0.5 * P * H * term_inv * error
        
        return velocity

    def run(self, observations):
        T = tf.shape(observations)[0]
        state_dim = 1
        
        particles = tf.random.normal((self.num_particles, state_dim), dtype=tf.float32)
        weights = tf.ones((self.num_particles,), dtype=tf.float32) / float(self.num_particles)
        estimates = tf.TensorArray(dtype=tf.float32, size=T)
        
        # R_proxy: Variance of log(chi-square noise). 
        # For SV, log(v^2) is log(chi^2_1). Variance is approx 4.93.
        # We use a tuned value for stability.
        R_proxy = tf.constant([[4.93]], dtype=tf.float32) 

        for t in range(T):
            z = tf.reshape(observations[t], [1, 1])
            
            # 1. Prediction
            particles = self.model.transition(particles)
            
            # 2. Particle Flow (Log-Domain)
            mean_p = tf.reduce_mean(particles, axis=0)
            diff = particles - mean_p
            P_matrix = tf.matmul(diff, diff, transpose_a=True) / (self.num_particles - 1)
            
            curr_particles = particles
            log_jacobian_det = tf.zeros((self.num_particles, 1))
            
            for k in range(self.flow_steps):
                velo = self.apply_edh_flow(curr_particles, P_matrix, R_proxy, z)
                curr_particles = curr_particles + self.epsilon * velo
                
                # Divergence for weight update
                with tf.GradientTape() as tape_div:
                    tape_div.watch(curr_particles)
                    v_temp = self.apply_edh_flow(curr_particles, P_matrix, R_proxy, z)
                div_v = tape_div.gradient(v_temp, curr_particles)
                log_jacobian_det -= div_v * self.epsilon

            flowed_particles = curr_particles
            
            # 3. Update Weights (Standard Likelihood on Flowed Particles)
            # We must use the ORIGINAL likelihood function (raw y), not the log-proxy
            likelihoods = self.model.log_likelihood(observations[t], flowed_particles)
            likelihoods = tf.reshape(likelihoods, (-1, 1))
            
            log_weights = tf.math.log(weights + 1e-16)
            log_weights = tf.reshape(log_weights, (-1, 1))
            
            unnormalized_log_w = log_weights + likelihoods + log_jacobian_det
            
            w_max = tf.reduce_max(unnormalized_log_w)
            weights = tf.exp(unnormalized_log_w - w_max)
            weights = weights / tf.reduce_sum(weights)
            weights = tf.reshape(weights, (-1,))
            
            # 4. Estimate
            estimate = tf.reduce_sum(flowed_particles * tf.reshape(weights, (-1, 1)), axis=0)
            estimates = estimates.write(t, estimate)
            
            # Resampling
            eff_N = 1.0 / tf.reduce_sum(weights**2)
            if eff_N < self.num_particles / 2.0:
                 indices = tf.random.categorical(tf.reshape(tf.math.log(weights), (1, -1)), self.num_particles)
                 particles = tf.gather(flowed_particles, indices[0])
                 weights = tf.ones_like(weights) / float(self.num_particles)
            else:
                 particles = flowed_particles
            
        return estimates.stack()