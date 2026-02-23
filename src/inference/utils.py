"""Shared utilities for Bayesian inference algorithms."""
import numpy as np


def compute_ess(samples: np.ndarray, max_lag: int = None) -> float:
    """Compute Effective Sample Size (ESS) for an MCMC chain.

    Uses the initial positive sequence estimator (Geyer, 1992).

    Args:
        samples: MCMC samples of shape ``(num_samples,)`` or
            ``(num_samples, dim)``.
        max_lag: Maximum lag for autocorrelation.  Defaults to
            ``num_samples // 2``.

    Returns:
        ESS averaged over dimensions.
    """
    if len(samples.shape) == 1:
        samples = samples.reshape(-1, 1)

    n_samples, n_dims = samples.shape

    if max_lag is None:
        max_lag = n_samples // 2

    ess_values = []

    for d in range(n_dims):
        x = samples[:, d]
        x = x - np.mean(x)

        var = np.var(x)
        if var < 1e-16:
            ess_values.append(float(n_samples))
            continue

        acf = np.correlate(x, x, mode='full')[n_samples - 1:] / (n_samples * var)
        acf = acf[:max_lag]

        tau = 1.0
        for k in range(1, len(acf)):
            if acf[k] > 0:
                tau += 2 * acf[k]
            else:
                break

        ess_values.append(n_samples / tau)

    return float(np.mean(ess_values))
