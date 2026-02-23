"""
Gradient quality diagnostics for differentiable particle filters.

Provides metrics to assess whether gradients through the DPF pipeline
are informative enough for gradient-based inference (HMC, optimization).

Metrics:
    - estimate_gradient_variance: Var[grad] across random seeds
    - gradient_agreement: cosine similarity between analytic and finite-diff grads
    - estimate_gradient_snr: signal-to-noise ratio E[grad]^2 / Var[grad]
    - estimate_condition_number: cond(Hessian) at a point
    - estimate_lipschitz: empirical Lipschitz constant of gradient map

Reference:
    Mohamed et al. (2020), "Monte Carlo Gradient Estimation in Machine Learning"
"""
import tensorflow as tf
import numpy as np


def _compute_gradient(loss_fn, params):
    """Compute gradient of loss_fn w.r.t. params using GradientTape."""
    with tf.GradientTape() as tape:
        tape.watch(params)
        loss = loss_fn(params)
    grad = tape.gradient(loss, params)
    return grad, loss


def estimate_gradient_variance(loss_fn, params, num_samples=20):
    """
    Estimate variance of stochastic gradients across multiple evaluations.

    The variance arises from the stochastic particle filter internals
    (random resampling, random transitions). Lower variance means more
    stable gradient signal for optimization or HMC.

    Args:
        loss_fn: Callable(params) -> scalar loss (e.g., negative log ML)
        params: tf.Variable of model parameters
        num_samples: Number of independent gradient evaluations

    Returns:
        dict with:
            'per_param_variance': tf.Tensor of shape [param_dim]
            'mean_variance': scalar mean across parameters
            'gradients': tf.Tensor of shape [num_samples, param_dim]
    """
    grads = []
    for i in range(num_samples):
        tf.random.set_seed(i * 1000 + 7)
        grad, _ = _compute_gradient(loss_fn, params)
        if grad is None:
            raise ValueError("Gradient is None — loss_fn may not depend on params")
        grads.append(tf.reshape(grad, [-1]))

    grads_tensor = tf.stack(grads)  # [num_samples, param_dim]
    per_param_var = tf.math.reduce_variance(grads_tensor, axis=0)
    mean_var = tf.reduce_mean(per_param_var)

    return {
        'per_param_variance': per_param_var,
        'mean_variance': mean_var,
        'gradients': grads_tensor,
    }


def gradient_agreement(loss_fn, params, epsilon=1e-4, num_avg=5):
    """
    Compare analytic gradient (tape) with central finite differences.

    This is the gold-standard check for gradient correctness. If the tape
    gradient disagrees with finite differences, the differentiable
    resampling is not propagating gradients correctly.

    Args:
        loss_fn: Callable(params) -> scalar loss
        params: tf.Variable of model parameters
        epsilon: Perturbation size for finite differences
        num_avg: Number of runs to average (reduces PF stochasticity)

    Returns:
        dict with:
            'cosine_similarity': scalar in [-1, 1]
            'relative_error': scalar ||analytic - fd|| / ||fd||
            'analytic_grad': tf.Tensor
            'finite_diff_grad': tf.Tensor
    """
    param_flat = tf.reshape(params, [-1])
    dim = param_flat.shape[0]

    # Average analytic gradient over multiple runs
    analytic_grads = []
    for i in range(num_avg):
        tf.random.set_seed(i * 2000 + 13)
        grad, _ = _compute_gradient(loss_fn, params)
        if grad is None:
            raise ValueError("Gradient is None")
        analytic_grads.append(tf.reshape(grad, [-1]))
    analytic_grad = tf.reduce_mean(tf.stack(analytic_grads), axis=0)

    # Central finite differences (averaged over runs)
    fd_grad = tf.zeros([dim], dtype=tf.float32)
    for d in range(dim):
        e_d = tf.one_hot(d, dim, dtype=tf.float32)

        fwd_losses = []
        bwd_losses = []
        for i in range(num_avg):
            tf.random.set_seed(i * 3000 + 17)
            params_fwd = tf.Variable(tf.reshape(param_flat + epsilon * e_d, params.shape))
            loss_fwd = loss_fn(params_fwd)
            fwd_losses.append(loss_fwd)

            tf.random.set_seed(i * 3000 + 17)
            params_bwd = tf.Variable(tf.reshape(param_flat - epsilon * e_d, params.shape))
            loss_bwd = loss_fn(params_bwd)
            bwd_losses.append(loss_bwd)

        avg_fwd = tf.reduce_mean(fwd_losses)
        avg_bwd = tf.reduce_mean(bwd_losses)
        fd_d = (avg_fwd - avg_bwd) / (2.0 * epsilon)

        indices = tf.constant([[d]])
        fd_grad = tf.tensor_scatter_nd_update(fd_grad, indices, [fd_d])

    # Cosine similarity
    dot = tf.reduce_sum(analytic_grad * fd_grad)
    norm_a = tf.norm(analytic_grad)
    norm_fd = tf.norm(fd_grad)
    cosine_sim = dot / (norm_a * norm_fd + 1e-12)

    # Relative error
    rel_error = tf.norm(analytic_grad - fd_grad) / (norm_fd + 1e-12)

    return {
        'cosine_similarity': cosine_sim,
        'relative_error': rel_error,
        'analytic_grad': analytic_grad,
        'finite_diff_grad': fd_grad,
    }


def estimate_gradient_snr(loss_fn, params, num_samples=20):
    """
    Estimate gradient signal-to-noise ratio.

    SNR = E[grad]^2 / Var[grad] per parameter.
    SNR < 1 means noise dominates the gradient signal, making
    optimization or HMC unreliable.

    Args:
        loss_fn: Callable(params) -> scalar loss
        params: tf.Variable of model parameters
        num_samples: Number of gradient evaluations

    Returns:
        dict with:
            'per_param_snr': tf.Tensor of shape [param_dim]
            'mean_snr': scalar
    """
    result = estimate_gradient_variance(loss_fn, params, num_samples)
    grads = result['gradients']

    mean_grad = tf.reduce_mean(grads, axis=0)
    var_grad = result['per_param_variance']

    snr = tf.square(mean_grad) / (var_grad + 1e-12)

    return {
        'per_param_snr': snr,
        'mean_snr': tf.reduce_mean(snr),
    }


def estimate_condition_number(loss_fn, params, epsilon=1e-3):
    """
    Estimate condition number of the Hessian via finite differences.

    Large condition number means the loss landscape is ill-conditioned,
    which makes HMC inefficient (requires very small step sizes).

    Args:
        loss_fn: Callable(params) -> scalar loss
        params: tf.Variable of model parameters
        epsilon: Perturbation size for Hessian estimation

    Returns:
        dict with:
            'condition_number': scalar sigma_max / sigma_min
            'eigenvalues': tf.Tensor of Hessian eigenvalues
    """
    param_flat = tf.reshape(params, [-1])
    dim = param_flat.shape[0]

    # Compute Hessian via finite differences of gradients
    hessian = np.zeros((dim, dim), dtype=np.float32)

    for d in range(dim):
        e_d = tf.one_hot(d, dim, dtype=tf.float32)

        params_fwd = tf.Variable(tf.reshape(param_flat + epsilon * e_d, params.shape))
        grad_fwd, _ = _compute_gradient(loss_fn, params_fwd)
        grad_fwd = tf.reshape(grad_fwd, [-1])

        params_bwd = tf.Variable(tf.reshape(param_flat - epsilon * e_d, params.shape))
        grad_bwd, _ = _compute_gradient(loss_fn, params_bwd)
        grad_bwd = tf.reshape(grad_bwd, [-1])

        hessian[:, d] = ((grad_fwd - grad_bwd) / (2.0 * epsilon)).numpy()

    # Symmetrize
    hessian = (hessian + hessian.T) / 2.0
    hessian_tf = tf.constant(hessian, dtype=tf.float32)

    eigenvalues = tf.linalg.eigvalsh(hessian_tf)
    abs_eigs = tf.abs(eigenvalues)
    sigma_max = tf.reduce_max(abs_eigs)
    sigma_min = tf.reduce_min(abs_eigs)

    cond = sigma_max / (sigma_min + 1e-12)

    return {
        'condition_number': cond,
        'eigenvalues': eigenvalues,
    }


def estimate_lipschitz(loss_fn, params, num_probes=50, radius=0.1):
    """
    Estimate Lipschitz constant of the gradient map.

    L = max_i ||grad(params + delta_i) - grad(params)|| / ||delta_i||

    Large L means the gradient changes rapidly and HMC/optimization
    requires small step sizes for stability.

    Args:
        loss_fn: Callable(params) -> scalar loss
        params: tf.Variable of model parameters
        num_probes: Number of random perturbations
        radius: Perturbation radius

    Returns:
        dict with:
            'lipschitz_estimate': scalar
            'max_ratio': scalar (same as lipschitz_estimate)
    """
    param_flat = tf.reshape(params, [-1])
    dim = param_flat.shape[0]

    # Gradient at current point
    grad_0, _ = _compute_gradient(loss_fn, params)
    grad_0 = tf.reshape(grad_0, [-1])

    max_ratio = 0.0

    for i in range(num_probes):
        delta = tf.random.normal([dim], dtype=tf.float32)
        delta = delta / (tf.norm(delta) + 1e-12) * radius

        params_perturbed = tf.Variable(tf.reshape(param_flat + delta, params.shape))
        grad_p, _ = _compute_gradient(loss_fn, params_perturbed)
        grad_p = tf.reshape(grad_p, [-1])

        grad_diff_norm = tf.norm(grad_p - grad_0).numpy()
        delta_norm = tf.norm(delta).numpy()
        ratio = grad_diff_norm / (delta_norm + 1e-12)

        max_ratio = max(max_ratio, ratio)

    return {
        'lipschitz_estimate': max_ratio,
        'max_ratio': max_ratio,
    }
