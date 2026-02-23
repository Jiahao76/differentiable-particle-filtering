"""
Differentiable Particle Flow Particle Filter (DPF-PF)

Combines:
1. Li & Coates (2017): Invertible particle flow with LEDH for proposal
2. Corenflos et al. (2021): Entropy-regularized OT resampling for differentiability

This enables gradient-based inference (e.g., HMC) for particle filter models.
"""
import tensorflow as tf
import numpy as np
from src.models.base_model import StateSpaceModel


class DifferentiablePFPF:
    """
    Differentiable Particle Flow Particle Filter.
    
    Key features:
    - Uses LEDH flow as proposal distribution (Li 2017)
    - Computes importance weights accounting for flow Jacobian
    - Uses entropy-regularized OT for differentiable resampling (Corenflos 2021)
    - Fully differentiable w.r.t. model parameters
    """
    
    def __init__(
        self,
        model: StateSpaceModel,
        num_particles: int = 100,
        flow_steps: int = 20,
        step_size: float = 0.05,
        ot_epsilon: float = 0.5,
        ot_iterations: int = 50,
        resample_threshold: float = 0.5,
    ):
        self.model = model
        self.num_particles = num_particles
        self.flow_steps = flow_steps
        self.epsilon = step_size
        self.ot_epsilon = ot_epsilon
        self.ot_iterations = ot_iterations
        self.resample_threshold = resample_threshold
    
    def compute_observation_gradient(self, observation, particles):
        """
        Compute gradient of observation function h(x) w.r.t. particles.
        
        For general models, uses automatic differentiation.
        Model-specific implementations can override for efficiency.
        
        Args:
            observation: Current observation
            particles: Current particle states [N, state_dim]
        
        Returns:
            h_val: h(particles) [N, obs_dim]
            H: Jacobian dh/dx [N, obs_dim, state_dim]
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(particles)
            
            h_val = self.model.observation_mean(particles)
        
        # Compute Jacobian
        H = tape.gradient(h_val, particles)
        
        if H is None:
            # If gradient is None, use identity (works for linear observations)
            H = tf.eye(self.model.state_dim, batch_shape=[self.num_particles, self.model.obs_dim])
        
        del tape
        return h_val, H
    
    def ledh_flow_step(self, particles, observation, lambda_val, P):
        """
        Single LEDH flow step.
        
        Each particle uses its own local linearization.
        
        Args:
            particles: Current particles [N, state_dim]
            observation: Current observation [obs_dim]
            lambda_val: Flow parameter (0 to 1)
            P: Ensemble covariance (scalar for 1D, matrix for multi-D)
        
        Returns:
            new_particles: Updated particles [N, state_dim]
            A: Flow matrix for Jacobian computation [N, state_dim, state_dim]
        """
        # For 1D case (most common in examples)
        if self.model.state_dim == 1 and self.model.obs_dim == 1:
            return self._ledh_flow_step_1d(particles, observation, lambda_val, P)
        else:
            return self._ledh_flow_step_nd(particles, observation, lambda_val, P)
    
    def _ledh_flow_step_1d(self, particles, observation, lambda_val, P):
        """LEDH flow for 1D state and observation."""
        with tf.GradientTape() as tape:
            tape.watch(particles)
            
            # Compute h(x) using the observation function
            # For NonlinearSSM: h(x) = x^2/20
            # We need model-specific h(x) - let's use a generic approach
            
            h_val = self.model.observation_mean(particles)
        
        H = tape.gradient(h_val, particles)
        
        if H is None:
            H = tf.ones_like(particles)
        
        # Observation noise variance (model-specific)
        if hasattr(self.model, 'sigma_W'):
            R = self.model.sigma_W ** 2
        elif hasattr(self.model, 'R'):
            R = self.model.R
        else:
            R = 1.0
        
        # LEDH flow parameters
        denom = lambda_val * (H**2) * P + R
        A = -0.5 * P * (H**2) / (denom + 1e-8)
        
        # Innovation
        e = h_val - H * particles
        innovation = observation - e
        
        # Flow velocity
        factor1 = 1.0 + 2.0 * lambda_val * A
        factor2 = 1.0 + lambda_val * A
        factor3 = P * H / (R + 1e-8)
        b = factor1 * factor2 * factor3 * innovation + A * particles
        
        # Update
        new_particles = particles + self.epsilon * (A * particles + b)
        
        return new_particles, A
    
    def _ledh_flow_step_nd(self, particles, observation, lambda_val, P):
        """LEDH flow for multi-dimensional state/observation."""
        # TODO: Implement multi-dimensional version
        raise NotImplementedError("Multi-dimensional LEDH flow not yet implemented")
    
    def sinkhorn_transport(self, weights: tf.Tensor, cost: tf.Tensor):
        """
        Compute entropy-regularized optimal transport matrix via Sinkhorn algorithm.
        
        Args:
            weights: Source distribution [N]
            cost: Cost matrix [N, N]
        
        Returns:
            Transport matrix [N, N]
        """
        N = self.num_particles
        
        log_a = tf.math.log(weights + 1e-20)
        log_b = tf.math.log(tf.fill((N,), 1.0 / float(N)))
        log_K = -cost / self.ot_epsilon
        
        log_u = tf.zeros_like(log_a)
        log_v = tf.zeros_like(log_b)
        
        for _ in range(self.ot_iterations):
            log_u = log_a - tf.reduce_logsumexp(log_K + tf.reshape(log_v, (1, -1)), axis=1)
            log_v = log_b - tf.reduce_logsumexp(log_K + tf.reshape(log_u, (-1, 1)), axis=0)
        
        log_P = log_K + tf.reshape(log_u, (-1, 1)) + tf.reshape(log_v, (1, -1))
        return tf.exp(log_P)
    
    def ot_resample(self, particles: tf.Tensor, weights: tf.Tensor):
        """
        Differentiable resampling via optimal transport.
        
        Args:
            particles: Current particles [N, state_dim]
            weights: Normalized weights [N]
        
        Returns:
            new_particles: Resampled particles [N, state_dim]
            new_weights: Uniform weights [N]
        """
        # Compute cost matrix (squared Euclidean distance)
        diff = tf.expand_dims(particles, 1) - tf.expand_dims(particles, 0)  # [N, N, state_dim]
        cost = tf.reduce_sum(tf.square(diff), axis=-1)  # [N, N]
        
        # Normalize cost
        cost = cost / (tf.reduce_mean(cost) + 1e-8)
        
        # Compute transport matrix
        transport = self.sinkhorn_transport(weights, cost)
        
        # Barycentric projection
        new_particles = tf.matmul(transport, particles, transpose_a=True) * float(self.num_particles)
        new_weights = tf.ones((self.num_particles,), dtype=tf.float32) / float(self.num_particles)
        
        return new_particles, new_weights
    
    def run(self, observations: tf.Tensor, initial_particles: tf.Tensor = None):
        """
        Run the differentiable PF-PF.
        
        Args:
            observations: Observation sequence [T, obs_dim]
            initial_particles: Optional initial particles [N, state_dim]
        
        Returns:
            estimates: State estimates [T, state_dim]
            log_marginal_likelihood: Estimate of log p(y_{1:T} | theta)
            ess_history: ESS at each time step [T]
        """
        T = tf.shape(observations)[0]
        
        # Initialize particles
        if initial_particles is None:
            particles = tf.random.normal((self.num_particles, self.model.state_dim), dtype=tf.float32)
        else:
            particles = initial_particles
        
        weights = tf.ones((self.num_particles,), dtype=tf.float32) / float(self.num_particles)
        
        estimates = []
        log_marginal_likelihood = 0.0
        ess_history = []
        
        for t in range(T):
            observation = observations[t:t+1]  # [1, obs_dim] or just [obs_dim]
            if len(observation.shape) == 1:
                observation = tf.reshape(observation, [1, -1])
            observation = observation[0]  # [obs_dim]
            
            # ===== PREDICTION =====
            if hasattr(self.model, 'reset_time'):
                # For time-varying models like NonlinearSSM
                if t == 0:
                    self.model.reset_time()
            
            if hasattr(self.model, 'time_step'):
                particles_0 = self.model.transition(particles, time_step=t)
            else:
                particles_0 = self.model.transition(particles)
            
            # Compute predictive covariance
            eta_mean = tf.reduce_mean(particles_0, axis=0, keepdims=True)
            P = tf.reduce_mean((particles_0 - eta_mean)**2)
            P_fixed = tf.identity(P)  # Fixed covariance for flow
            
            # ===== INVERTIBLE PARTICLE FLOW =====
            current_particles = tf.identity(particles_0)
            log_det_sum = tf.zeros((self.num_particles,), dtype=tf.float32)
            
            lambda_val = 0.0
            for j in range(self.flow_steps):
                lambda_val += self.epsilon
                
                # LEDH flow step
                new_particles, A = self.ledh_flow_step(current_particles, observation, lambda_val, P_fixed)
                
                # Accumulate Jacobian determinant (1D: det(I + eps*A) = 1 + eps*A)
                if self.model.state_dim == 1:
                    log_det_sum += tf.math.log(tf.abs(1.0 + self.epsilon * tf.reshape(A, [-1])) + 1e-8)
                
                # Clip for stability
                new_particles = tf.clip_by_value(new_particles, -10.0, 10.0)
                current_particles = new_particles
            
            particles_1 = current_particles
            
            # ===== WEIGHT UPDATE (Li 2017) =====
            # p(y | eta_1)
            log_lik = self.model.log_likelihood(observation, particles_1)
            log_lik = tf.reshape(log_lik, (-1,))
            
            # Jacobian term: log |det(J_T)|
            log_jacobian = log_det_sum
            
            # Prior ratio: log p(eta_1 | x_prev) - log p(eta_0 | x_prev)
            # For time-varying models, need to pass time_step
            if hasattr(self.model, 'transition_log_pdf'):
                # Assuming we can compute this (may need special handling for time-varying models)
                # For simplicity, approximate as Gaussian with learned variance
                log_prior_1 = -0.5 * ((particles_1 - eta_mean)**2) / (P_fixed + 1e-8)
                log_prior_0 = -0.5 * ((particles_0 - eta_mean)**2) / (P_fixed + 1e-8)
                log_prior_ratio = tf.reshape(tf.reduce_sum(log_prior_1 - log_prior_0, axis=-1), (-1,))
            else:
                log_prior_ratio = 0.0
            
            # Total log weight
            log_weights = log_lik + log_jacobian + log_prior_ratio
            
            # Normalize and compute estimate
            log_max = tf.reduce_max(log_weights)
            log_weights_stable = log_weights - log_max
            weights = tf.exp(log_weights_stable)
            weights = weights / (tf.reduce_sum(weights) + 1e-16)
            
            estimate = tf.reduce_sum(particles_1 * tf.reshape(weights, (-1, 1)), axis=0)
            estimates.append(estimate)
            
            # Update log marginal likelihood
            log_marginal_increment = log_max + tf.math.log(tf.reduce_mean(tf.exp(log_weights_stable)))
            log_marginal_likelihood += log_marginal_increment
            
            # Compute ESS
            ess = 1.0 / tf.reduce_sum(weights ** 2)
            ess_history.append(ess)
            
            # ===== DIFFERENTIABLE RESAMPLING =====
            if ess < self.num_particles * self.resample_threshold:
                particles, weights = self.ot_resample(particles_1, weights)
            else:
                particles = particles_1
        
        estimates = tf.stack(estimates)
        ess_history = tf.stack(ess_history)
        
        return estimates, log_marginal_likelihood, ess_history
