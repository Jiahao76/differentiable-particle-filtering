"""
Hamiltonian Monte Carlo (HMC) for parameter inference with particle filters.

Based on Neal (2011) "MCMC using Hamiltonian dynamics"

Uses differentiable particle filters to enable gradient-based MCMC.
"""
import tensorflow as tf
import numpy as np
from typing import Callable, Dict, Any
import time


class HMC:
    """
    Hamiltonian Monte Carlo sampler.
    
    Uses leapfrog integration to propose new parameter values,
    with Metropolis acceptance step.
    """
    
    def __init__(
        self,
        log_posterior_fn: Callable,
        gradient_fn: Callable,
        step_size: float = 0.01,
        num_leapfrog_steps: int = 10,
        mass_matrix: np.ndarray = None,
    ):
        """
        Args:
            log_posterior_fn: Function that computes log p(theta | data)
            gradient_fn: Function that computes gradient of log posterior
            step_size: Leapfrog step size (epsilon)
            num_leapfrog_steps: Number of leapfrog steps (L)
            mass_matrix: Mass matrix M for momentum (default: identity)
        """
        self.log_posterior_fn = log_posterior_fn
        self.gradient_fn = gradient_fn
        self.epsilon = step_size
        self.L = num_leapfrog_steps
        self.mass_matrix = mass_matrix
        self.inv_mass_matrix = None if mass_matrix is None else np.linalg.inv(mass_matrix)
    
    def kinetic_energy(self, p: np.ndarray) -> float:
        """
        Compute kinetic energy K(p) = 0.5 * p^T * M^{-1} * p
        
        Args:
            p: Momentum vector
        
        Returns:
            Kinetic energy
        """
        if self.inv_mass_matrix is not None:
            return 0.5 * np.dot(p, np.dot(self.inv_mass_matrix, p))
        else:
            return 0.5 * np.sum(p ** 2)
    
    def sample_momentum(self, dim: int) -> np.ndarray:
        """
        Sample momentum from N(0, M).
        
        Args:
            dim: Dimension of parameter space
        
        Returns:
            Momentum vector
        """
        if self.mass_matrix is not None:
            # Sample from N(0, M) using Cholesky decomposition
            L = np.linalg.cholesky(self.mass_matrix)
            return np.dot(L, np.random.randn(dim))
        else:
            return np.random.randn(dim)
    
    def leapfrog(self, q: np.ndarray, p: np.ndarray) -> tuple:
        """
        Leapfrog integrator for Hamiltonian dynamics.
        
        Args:
            q: Current position (parameters)
            p: Current momentum
        
        Returns:
            (new_q, new_p): Updated position and momentum
        """
        # Make copies to avoid modifying originals
        q = np.copy(q)
        p = np.copy(p)
        
        # Half step for momentum
        grad = self.gradient_fn(q)
        p = p + 0.5 * self.epsilon * grad
        
        # Full steps for position and momentum
        for i in range(self.L):
            # Full step for position
            if self.inv_mass_matrix is not None:
                q = q + self.epsilon * np.dot(self.inv_mass_matrix, p)
            else:
                q = q + self.epsilon * p
            
            # Full step for momentum (except at last step)
            if i < self.L - 1:
                grad = self.gradient_fn(q)
                p = p + self.epsilon * grad
        
        # Half step for momentum at end
        grad = self.gradient_fn(q)
        p = p + 0.5 * self.epsilon * grad
        
        # Negate momentum for reversibility
        p = -p
        
        return q, p
    
    def step(self, q_current: np.ndarray, log_posterior_current: float = None) -> tuple:
        """
        Single HMC step.
        
        Args:
            q_current: Current parameter values
            log_posterior_current: Optional pre-computed log posterior
        
        Returns:
            (q_new, log_posterior_new, accepted): New parameter values, log posterior, and acceptance flag
        """
        # Sample momentum
        p_current = self.sample_momentum(len(q_current))
        
        # Compute current Hamiltonian
        if log_posterior_current is None:
            log_posterior_current = self.log_posterior_fn(q_current)
        
        H_current = -log_posterior_current + self.kinetic_energy(p_current)
        
        # Leapfrog integration
        try:
            q_proposed, p_proposed = self.leapfrog(q_current, p_current)
        except Exception as e:
            print(f"Leapfrog integration failed: {e}")
            return q_current, log_posterior_current, False
        
        # Compute proposed Hamiltonian
        log_posterior_proposed = self.log_posterior_fn(q_proposed)
        H_proposed = -log_posterior_proposed + self.kinetic_energy(p_proposed)
        
        # Metropolis acceptance
        log_accept_ratio = -(H_proposed - H_current)
        
        if np.log(np.random.rand()) < log_accept_ratio:
            return q_proposed, log_posterior_proposed, True
        else:
            return q_current, log_posterior_current, False
    
    def sample(
        self,
        initial_params: np.ndarray,
        num_samples: int,
        burn_in: int = 0,
        thin: int = 1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run HMC sampling.
        
        Args:
            initial_params: Starting parameter values
            num_samples: Number of samples to draw (after burn-in and thinning)
            burn_in: Number of initial samples to discard
            thin: Keep every thin-th sample
            verbose: Print progress
        
        Returns:
            Dictionary with samples, acceptance rate, and diagnostics
        """
        samples = []
        log_posteriors = []
        acceptance_count = 0
        total_iterations = burn_in + num_samples * thin
        
        q = np.copy(initial_params)
        log_post = self.log_posterior_fn(q)
        
        if verbose:
            print(f"Starting HMC sampling...")
            print(f"  Total iterations: {total_iterations}")
            print(f"  Burn-in: {burn_in}")
            print(f"  Thin: {thin}")
            print(f"  Target samples: {num_samples}")
            print(f"  Initial log posterior: {log_post:.4f}")
        
        start_time = time.time()
        
        for i in range(total_iterations):
            q, log_post, accepted = self.step(q, log_post)
            
            if accepted:
                acceptance_count += 1
            
            # Store sample after burn-in and thinning
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(np.copy(q))
                log_posteriors.append(log_post)
            
            if verbose and (i + 1) % 100 == 0:
                accept_rate = acceptance_count / (i + 1)
                elapsed = time.time() - start_time
                print(f"  Iteration {i+1}/{total_iterations}: "
                      f"accept_rate={accept_rate:.3f}, "
                      f"log_post={log_post:.4f}, "
                      f"elapsed={elapsed:.1f}s")
        
        elapsed_time = time.time() - start_time
        acceptance_rate = acceptance_count / total_iterations
        
        if verbose:
            print(f"\nHMC sampling completed!")
            print(f"  Final acceptance rate: {acceptance_rate:.3f}")
            print(f"  Total time: {elapsed_time:.1f}s")
            print(f"  Time per sample: {elapsed_time/num_samples:.3f}s")
        
        return {
            'samples': np.array(samples),
            'log_posteriors': np.array(log_posteriors),
            'acceptance_rate': acceptance_rate,
            'time': elapsed_time,
            'step_size': self.epsilon,
            'num_leapfrog_steps': self.L,
        }


# Re-export for backwards compatibility
from src.inference.utils import compute_ess  # noqa: F401
