"""
Particle Gibbs (PG) Sampler for State Space Models

Implementation of Particle Gibbs algorithm for joint posterior sampling
in state space models, particularly for State Space LSTM models.

Reference:
    Andrieu, C., Doucet, A., & Holenstein, R. (2010).
    "Particle Markov chain Monte Carlo methods"
    Journal of the Royal Statistical Society: Series B, 72(3), 269-342.
    
    Zheng, Z., et al. (2017). "State Space LSTM Models with Particle MCMC Inference"
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Callable, Dict, List
import time


class ParticleGibbs:
    """
    Particle Gibbs (PG) sampler using Conditional Particle Filter.
    
    PG is a Particle MCMC method that samples from the joint posterior:
    p(z_{1:T}, θ | x_{1:T})
    
    Key idea:
    1. Condition on a reference trajectory z^* (from previous iteration)
    2. Run Conditional SMC to sample new z_{1:T} | θ, x_{1:T}, z^*
    3. Sample parameters θ | z_{1:T}, x_{1:T} (often via Gibbs step)
    
    The conditioning ensures the Markov chain is ergodic.
    """
    
    def __init__(
        self,
        num_particles: int = 50,
        resampling_threshold: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Args:
            num_particles: Number of particles (excluding reference trajectory)
            resampling_threshold: ESS threshold for resampling (relative to N)
            seed: Random seed for reproducibility
        """
        self.N = num_particles
        self.resample_threshold = resampling_threshold
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
    
    def effective_sample_size(self, log_weights: np.ndarray) -> float:
        """
        Compute effective sample size (ESS).
        
        ESS = 1 / sum(w_i^2) where w_i are normalized weights
        
        Args:
            log_weights: Log weights [N]
            
        Returns:
            ess: Effective sample size
        """
        log_weights_norm = log_weights - np.max(log_weights)
        weights = np.exp(log_weights_norm)
        weights = weights / np.sum(weights)
        ess = 1.0 / np.sum(weights ** 2)
        return ess
    
    def systematic_resampling(
        self, 
        log_weights: np.ndarray,
        reference_idx: int = 0
    ) -> np.ndarray:
        """
        Systematic resampling with reference trajectory preservation.
        
        The reference trajectory (index 0) is always kept.
        
        Args:
            log_weights: Log weights [N+1] (including reference)
            reference_idx: Index of reference trajectory (default: 0)
            
        Returns:
            indices: Resampled indices [N+1]
        """
        N = len(log_weights)
        
        # Normalize weights
        log_weights_norm = log_weights - np.max(log_weights)
        weights = np.exp(log_weights_norm)
        weights = weights / np.sum(weights)
        
        # Reserve first position for reference trajectory
        indices = np.zeros(N, dtype=np.int32)
        indices[reference_idx] = reference_idx
        
        # Systematic resampling for remaining particles
        u = np.random.uniform(0, 1.0 / (N - 1))
        cumsum = np.cumsum(weights)
        
        j = 0
        for i in range(N):
            if i == reference_idx:
                continue
            
            threshold = u + (i if i < reference_idx else i - 1) / (N - 1)
            while j < N and cumsum[j] < threshold:
                j += 1
            indices[i] = min(j, N - 1)
        
        return indices
    
    def multinomial_resampling(
        self,
        log_weights: np.ndarray,
        reference_idx: int = 0
    ) -> np.ndarray:
        """
        Multinomial resampling with reference trajectory preservation.
        
        Args:
            log_weights: Log weights [N+1]
            reference_idx: Index of reference trajectory
            
        Returns:
            indices: Resampled indices [N+1]
        """
        N = len(log_weights)
        
        # Normalize weights
        log_weights_norm = log_weights - np.max(log_weights)
        weights = np.exp(log_weights_norm)
        weights = weights / np.sum(weights)
        
        # Sample N-1 particles (excluding reference)
        indices = np.random.choice(N, size=N-1, replace=True, p=weights)
        
        # Insert reference at position 0
        indices = np.insert(indices, reference_idx, reference_idx)
        
        return indices
    
    def conditional_particle_filter(
        self,
        model,
        observations: np.ndarray,
        reference_trajectory: Optional[np.ndarray] = None,
        return_trajectory: bool = True,
        return_weights: bool = False
    ) -> Dict:
        """
        Conditional Particle Filter (CPF).
        
        Runs a particle filter conditioned on a reference trajectory.
        The reference trajectory is always kept as particle 0.
        
        Args:
            model: State space model with methods:
                - reset_lstm_state(batch_size)
                - sample_transition(z_prev, training=True)
                - log_likelihood(x, z)
            observations: Observations [T, obs_dim]
            reference_trajectory: Reference trajectory [T, state_dim]
                If None, runs standard particle filter
            return_trajectory: Whether to return sampled trajectory
            return_weights: Whether to return particle weights
            
        Returns:
            result: Dictionary containing:
                - log_marginal_lik: Log marginal likelihood estimate
                - trajectory: Sampled trajectory [T, state_dim] (if requested)
                - weights_history: Weights at each time step (if requested)
                - ess_history: ESS at each time step
                - num_resamples: Number of resampling steps
        """
        T = len(observations)
        state_dim = model.state_dim
        N = self.N + 1  # +1 for reference trajectory
        
        # Initialize particles
        # Particle 0 is always the reference (if provided)
        particles = np.zeros((N, state_dim))
        
        if reference_trajectory is not None:
            particles[0] = reference_trajectory[0]
            # Initialize other particles from prior
            particles[1:] = np.random.randn(N - 1, state_dim) * 0.5
        else:
            particles = np.random.randn(N, state_dim) * 0.5
        
        log_weights = np.zeros(N)
        
        # Storage for trajectory and diagnostics
        trajectory = np.zeros((T, state_dim))
        ancestor_indices = np.zeros((T, N), dtype=np.int32)
        ess_history = []
        weights_history = [] if return_weights else None
        num_resamples = 0
        
        log_marginal_lik = 0.0
        
        # Reset model's LSTM state
        model.reset_lstm_state(batch_size=N)
        
        for t in range(T):
            # === Propagation step ===
            particles_prev = particles.copy()
            
            # Propagate particles through transition
            particles_tf = tf.convert_to_tensor(particles, dtype=tf.float32)
            new_particles_tf, _ = model.sample_transition(particles_tf, training=True)
            new_particles = new_particles_tf.numpy()
            
            # If we have a reference, overwrite particle 0
            if reference_trajectory is not None:
                new_particles[0] = reference_trajectory[t]
            
            particles = new_particles
            
            # === Weighting step ===
            obs = observations[t]
            obs_tf = tf.tile(
                tf.reshape(obs, (1, -1)),
                [N, 1]
            )
            
            particles_tf = tf.convert_to_tensor(particles, dtype=tf.float32)
            log_liks = model.log_likelihood(obs_tf, particles_tf).numpy()
            
            log_weights = log_weights + log_liks
            
            # Normalize weights for ESS and marginal likelihood
            log_weights_norm = log_weights - np.max(log_weights)
            weights = np.exp(log_weights_norm)
            sum_weights = np.sum(weights)
            weights = weights / sum_weights
            
            # Update marginal likelihood estimate
            log_marginal_lik += np.log(sum_weights) + np.max(log_weights) - np.log(N)
            
            # Compute ESS
            ess = 1.0 / np.sum(weights ** 2)
            ess_history.append(ess)
            
            if return_weights:
                weights_history.append(weights.copy())
            
            # === Resampling step ===
            if ess < self.resample_threshold * N and t < T - 1:
                indices = self.systematic_resampling(log_weights, reference_idx=0)
                particles = particles[indices]
                ancestor_indices[t] = indices
                log_weights = np.zeros(N)
                num_resamples += 1
            else:
                ancestor_indices[t] = np.arange(N)
            
            # Store for trajectory reconstruction
            if t == T - 1 and return_trajectory:
                # Sample final trajectory according to weights
                final_idx = np.random.choice(N, p=weights)
                trajectory[t] = particles[final_idx]
        
        # Reconstruct trajectory by tracing back ancestors
        if return_trajectory:
            current_idx = final_idx if reference_trajectory is None else 0
            for t in range(T - 1, -1, -1):
                trajectory[t] = particles[current_idx] if t == T - 1 else trajectory[t]
                if t > 0:
                    current_idx = ancestor_indices[t][current_idx]
        
        result = {
            'log_marginal_lik': log_marginal_lik,
            'ess_history': np.array(ess_history),
            'num_resamples': num_resamples
        }
        
        if return_trajectory:
            result['trajectory'] = trajectory
        
        if return_weights:
            result['weights_history'] = weights_history
        
        return result
    
    def particle_gibbs_sampler(
        self,
        model,
        observations: np.ndarray,
        num_iterations: int = 100,
        burn_in: int = 50,
        parameter_update_fn: Optional[Callable] = None,
        initial_trajectory: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run Particle Gibbs sampler.
        
        Args:
            model: State space model
            observations: Observations [T, obs_dim]
            num_iterations: Number of PG iterations
            burn_in: Number of burn-in iterations
            parameter_update_fn: Optional function to update parameters
                Signature: new_params = fn(trajectory, observations, current_params)
            initial_trajectory: Initial reference trajectory [T, state_dim]
            verbose: Whether to print progress
            
        Returns:
            results: Dictionary containing:
                - trajectories: Sampled trajectories [num_samples, T, state_dim]
                - log_marginal_liks: Log marginal likelihoods [num_iterations]
                - acceptance_rate: Acceptance rate (always 1.0 for PG)
                - time_per_iter: Average time per iteration
                - parameters: Parameter history (if parameter_update_fn provided)
        """
        T = len(observations)
        state_dim = model.state_dim
        
        # Initialize reference trajectory
        if initial_trajectory is None:
            # Run initial particle filter
            result = self.conditional_particle_filter(
                model, observations, 
                reference_trajectory=None,
                return_trajectory=True
            )
            reference_trajectory = result['trajectory']
        else:
            reference_trajectory = initial_trajectory
        
        # Storage
        trajectories = []
        log_marginal_liks = []
        parameters = [] if parameter_update_fn is not None else None
        times = []
        
        if verbose:
            print(f"Running Particle Gibbs sampler for {num_iterations} iterations...")
            print(f"Burn-in: {burn_in}, Particles: {self.N}")
        
        for iteration in range(num_iterations):
            start_time = time.time()
            
            # === Step 1: Sample trajectory | parameters, observations ===
            result = self.conditional_particle_filter(
                model,
                observations,
                reference_trajectory=reference_trajectory,
                return_trajectory=True
            )
            
            new_trajectory = result['trajectory']
            log_marginal_lik = result['log_marginal_lik']
            
            # === Step 2: Update parameters | trajectory, observations ===
            if parameter_update_fn is not None:
                model = parameter_update_fn(new_trajectory, observations, model)
                if iteration >= burn_in:
                    parameters.append(model.get_parameters())  # Assume model has this method
            
            # Update reference trajectory
            reference_trajectory = new_trajectory
            
            # Store results
            if iteration >= burn_in:
                trajectories.append(new_trajectory)
            
            log_marginal_liks.append(log_marginal_lik)
            
            iter_time = time.time() - start_time
            times.append(iter_time)
            
            if verbose and (iteration + 1) % 10 == 0:
                avg_time = np.mean(times[-10:])
                print(f"Iteration {iteration + 1}/{num_iterations} | "
                      f"Log-lik: {log_marginal_lik:.2f} | "
                      f"ESS (mean): {np.mean(result['ess_history']):.1f} | "
                      f"Time: {avg_time:.3f}s")
        
        results = {
            'trajectories': np.array(trajectories),
            'log_marginal_liks': np.array(log_marginal_liks),
            'acceptance_rate': 1.0,  # PG always accepts
            'time_per_iter': np.mean(times),
            'total_time': np.sum(times)
        }
        
        if parameters is not None:
            results['parameters'] = parameters
        
        if verbose:
            print(f"\nParticle Gibbs completed!")
            print(f"Total time: {results['total_time']:.2f}s")
            print(f"Average time per iteration: {results['time_per_iter']:.3f}s")
        
        return results
