import tensorflow as tf
import numpy as np
from src.models.base_model import StateSpaceModel

class StandardParticleFilter:
    """
    A Generic Standard Particle Filter (SIR) implementation in TensorFlow.
    
    This filter works with any model that inherits from `src.models.base_model.StateSpaceModel`.
    It implements the Sequential Importance Resampling (SIR) algorithm.
    
    Attributes:
        model (StateSpaceModel): The state-space model defining transition and likelihood.
        num_particles (int): Number of particles to use for estimation.
    """
    
    def __init__(self, model: StateSpaceModel, num_particles: int = 1000):
        """
        Initialize the Particle Filter.

        Args:
            model: An instance of a class inheriting from StateSpaceModel.
            num_particles: Number of Monte Carlo samples (particles).
        """
        self.model = model
        self.N = num_particles
        
    def initialize_particles(self, initial_dist_std: float = 1.0):
        """
        Initialize particles around zero (or based on model specific logic).
        For SV model, we typically initialize from the stationary distribution.
        """
        # Note: In a fully generic filter, we might accept a distribution object.
        # Here we assume a simple normal initialization for demonstration.
        return tf.random.normal(
            (self.N, self.model.state_dim), 
            mean=0.0, 
            stddev=initial_dist_std,
            dtype=tf.float32
        )

    def _resample(self, particles: tf.Tensor, log_weights: tf.Tensor):
        """
        Perform Multinomial Resampling based on log-weights.
        
        Args:
            particles: Current states [N, state_dim]
            log_weights: Unnormalized log weights [N]
            
        Returns:
            resampled_particles: [N, state_dim]
            reset_log_weights: [N] (all zeros)
        """
        # tf.random.categorical expects logits (unnormalized log-probs)
        # It returns indices of shape [1, N]
        indices = tf.random.categorical(tf.reshape(log_weights, (1, -1)), self.N)
        indices = tf.reshape(indices, (-1,)) # Flatten to [N]
        
        # Gather particles based on selected indices
        new_particles = tf.gather(particles, indices)
        
        # Reset weights to uniform (log(1/N) is constant, so we can use 0 for unnormalized)
        # We start fresh after resampling
        new_log_weights = tf.zeros((self.N,), dtype=tf.float32)
        
        return new_particles, new_log_weights

    def run(self, observations: tf.Tensor, verbose: bool = True):
        """
        Run the particle filter over a sequence of observations.
        """
        T = tf.shape(observations)[0]
        
        # 1. Initialization
        particles = self.initialize_particles(initial_dist_std=1.0)
        
        # Log weights initialized to 0 (shape: [N])
        log_weights = tf.zeros((self.N,), dtype=tf.float32)
        
        estimates_list = []
        ess_list = []
        
        if verbose:
            print(f"Starting Particle Filter with {self.N} particles...")
            
        for t in range(T):
            y_curr = observations[t]
            
            # --- A. Transition (Predict) ---
            particles = self.model.transition(particles)
            
            # --- B. Weight Update (Correct) ---
            log_likelihoods = self.model.log_likelihood(y_curr, particles)
            
            # [FIX] Reshape to (N,) to match log_weights shape and prevent 
            # accidental broadcasting into (N, N) matrix
            log_likelihoods = tf.reshape(log_likelihoods, (self.N,))
            
            # Update cumulative weights
            log_weights = log_weights + log_likelihoods
            
            # Normalize Log-Weights
            log_weights_norm = log_weights - tf.reduce_logsumexp(log_weights)
            weights = tf.exp(log_weights_norm)
            
            # --- C. Estimation ---
            # Weighted Mean
            # Reshape weights to [N, 1] for broadcasting against particles [N, state_dim]
            estimate = tf.reduce_sum(particles * tf.reshape(weights, (-1, 1)), axis=0)
            estimates_list.append(estimate)
            
            # --- D. Resampling Check ---
            ess = 1.0 / tf.reduce_sum(tf.square(weights))
            ess_list.append(ess)
            
            if ess < (self.N / 2.0):
                particles, log_weights = self._resample(particles, log_weights)
        
        estimates = tf.stack(estimates_list)
        ess_history = tf.stack(ess_list)
        
        return estimates, ess_history