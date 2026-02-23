"""
Particle Marginal Metropolis-Hastings (PMMH)

Based on Andrieu et al. (2010) "Particle Markov chain Monte Carlo methods"

Uses particle filter to estimate marginal likelihood for MCMC.
"""
import tensorflow as tf
import numpy as np
from typing import Callable, Dict, Any
import time


class PMMH:
    """
    Particle Marginal Metropolis-Hastings sampler.
    
    Uses particle filter to obtain unbiased estimates of marginal likelihood,
    which are then used in Metropolis-Hastings accept/reject step.
    """
    
    def __init__(
        self,
        particle_filter: Callable,
        log_prior_fn: Callable,
        proposal_fn: Callable,
        proposal_log_density_fn: Callable = None,
        symmetric_proposal: bool = True,
    ):
        """
        Args:
            particle_filter: Function that runs PF and returns log likelihood estimate
                            Signature: particle_filter(theta, data) -> log_lik_estimate
            log_prior_fn: Function that computes log prior p(theta)
            proposal_fn: Function that proposes new theta given current theta
                        Signature: proposal_fn(theta_current) -> theta_proposed
            proposal_log_density_fn: Log density q(theta_proposed | theta_current)
                                    Required if proposal is not symmetric
            symmetric_proposal: Whether proposal is symmetric (e.g., random walk)
        """
        self.particle_filter = particle_filter
        self.log_prior_fn = log_prior_fn
        self.proposal_fn = proposal_fn
        self.proposal_log_density_fn = proposal_log_density_fn
        self.symmetric_proposal = symmetric_proposal
        
        if not symmetric_proposal and proposal_log_density_fn is None:
            raise ValueError("proposal_log_density_fn required for asymmetric proposals")
    
    def log_posterior_unnormalized(self, theta: np.ndarray, data: Any) -> float:
        """
        Compute unnormalized log posterior: log p(theta | y) ∝ log p(y | theta) + log p(theta)
        
        Args:
            theta: Parameter values
            data: Observation data
        
        Returns:
            Unnormalized log posterior
        """
        log_prior = self.log_prior_fn(theta)
        
        if not np.isfinite(log_prior) or log_prior < -1e10:
            # Prior probability is essentially zero
            return -np.inf
        
        try:
            log_likelihood = self.particle_filter(theta, data)
        except Exception as e:
            print(f"Particle filter failed: {e}")
            return -np.inf
        
        return log_likelihood + log_prior
    
    def step(
        self,
        theta_current: np.ndarray,
        log_posterior_current: float,
        data: Any,
    ) -> tuple:
        """
        Single PMMH step.
        
        Args:
            theta_current: Current parameter values
            log_posterior_current: Current log posterior (or -inf if not computed)
            data: Observation data
        
        Returns:
            (theta_new, log_posterior_new, accepted): New parameters, log posterior, acceptance flag
        """
        # Propose new parameters
        theta_proposed = self.proposal_fn(theta_current)
        
        # Compute proposed log posterior
        log_posterior_proposed = self.log_posterior_unnormalized(theta_proposed, data)
        
        # Compute acceptance ratio
        if self.symmetric_proposal:
            log_accept_ratio = log_posterior_proposed - log_posterior_current
        else:
            # Include proposal ratio for asymmetric proposals
            log_q_forward = self.proposal_log_density_fn(theta_proposed, theta_current)
            log_q_backward = self.proposal_log_density_fn(theta_current, theta_proposed)
            log_accept_ratio = (log_posterior_proposed - log_posterior_current +
                               log_q_backward - log_q_forward)
        
        # Metropolis acceptance
        if np.log(np.random.rand()) < log_accept_ratio:
            return theta_proposed, log_posterior_proposed, True
        else:
            return theta_current, log_posterior_current, False
    
    def sample(
        self,
        initial_params: np.ndarray,
        data: Any,
        num_samples: int,
        burn_in: int = 0,
        thin: int = 1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run PMMH sampling.
        
        Args:
            initial_params: Starting parameter values
            data: Observation data
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
        
        theta = np.copy(initial_params)
        
        if verbose:
            print(f"Computing initial log posterior...")
        
        log_post = self.log_posterior_unnormalized(theta, data)
        
        if verbose:
            print(f"Starting PMMH sampling...")
            print(f"  Total iterations: {total_iterations}")
            print(f"  Burn-in: {burn_in}")
            print(f"  Thin: {thin}")
            print(f"  Target samples: {num_samples}")
            print(f"  Initial log posterior: {log_post:.4f}")
        
        start_time = time.time()
        
        for i in range(total_iterations):
            theta, log_post, accepted = self.step(theta, log_post, data)
            
            if accepted:
                acceptance_count += 1
            
            # Store sample after burn-in and thinning
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(np.copy(theta))
                log_posteriors.append(log_post)
            
            if verbose and (i + 1) % 10 == 0:
                accept_rate = acceptance_count / (i + 1)
                elapsed = time.time() - start_time
                print(f"  Iteration {i+1}/{total_iterations}: "
                      f"accept_rate={accept_rate:.3f}, "
                      f"log_post={log_post:.4f}, "
                      f"elapsed={elapsed:.1f}s")
        
        elapsed_time = time.time() - start_time
        acceptance_rate = acceptance_count / total_iterations
        
        if verbose:
            print(f"\nPMMH sampling completed!")
            print(f"  Final acceptance rate: {acceptance_rate:.3f}")
            print(f"  Total time: {elapsed_time:.1f}s")
            print(f"  Time per sample: {elapsed_time/num_samples:.3f}s")
        
        return {
            'samples': np.array(samples),
            'log_posteriors': np.array(log_posteriors),
            'acceptance_rate': acceptance_rate,
            'time': elapsed_time,
        }


def random_walk_proposal(theta: np.ndarray, step_size: float = 0.1) -> np.ndarray:
    """
    Symmetric random walk proposal for PMMH.
    
    Args:
        theta: Current parameter values
        step_size: Standard deviation of proposal
    
    Returns:
        Proposed parameter values
    """
    return theta + np.random.randn(len(theta)) * step_size


# Re-export for backwards compatibility
from src.inference.utils import compute_ess  # noqa: F401
