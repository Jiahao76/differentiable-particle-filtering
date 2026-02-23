"""
Benchmark Neural OT Resampling vs Traditional Sinkhorn

Compares:
1. Sinkhorn-100 (baseline, high accuracy)
2. Sinkhorn-30 (faster baseline)
3. mGradNet (neural network)
4. FNO (Fourier Neural Operator)
5. Hybrid (FNO + 5 Sinkhorn)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.sv_model import StochasticVolatilityModel
from src.models.nonlinear_ssm import NonlinearSSM
from src.filters.particle_filter import ParticleFilter
from src.filters.differentiable_particle_filter import DifferentiableParticleFilter
from src.filters.neural_ot_resampling import (
    OTResamplingNetwork,
    FourierOTOperator,
    neural_ot_resample,
    compute_statistics
)


def sinkhorn_ot(
    cost: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float = 0.1,
    num_iterations: int = 100
) -> tf.Tensor:
    """Standard Sinkhorn algorithm."""
    N = tf.shape(weights)[0]
    
    log_a = tf.math.log(weights + 1e-20)
    log_b = tf.math.log(tf.fill([N], 1.0 / tf.cast(N, tf.float32)))
    log_k = -cost / epsilon
    
    log_u = tf.zeros_like(log_a)
    log_v = tf.zeros_like(log_b)
    
    for _ in range(num_iterations):
        log_u = log_a - tf.reduce_logsumexp(log_k + tf.reshape(log_v, [1, -1]), axis=1)
        log_v = log_b - tf.reduce_logsumexp(log_k + tf.reshape(log_u, [-1, 1]), axis=0)
    
    log_p = log_k + tf.reshape(log_u, [-1, 1]) + tf.reshape(log_v, [1, -1])
    return tf.exp(log_p)


def sinkhorn_resample(particles: tf.Tensor, log_weights: tf.Tensor, 
                      epsilon: float, num_iter: int) -> tuple:
    """Resample using Sinkhorn."""
    N = particles.shape[0]
    weights = tf.nn.softmax(log_weights, axis=0)
    
    # Compute cost
    diff = tf.expand_dims(particles, 1) - tf.expand_dims(particles, 0)
    cost = tf.reduce_sum(tf.square(diff), axis=-1)
    
    # Sinkhorn
    P = sinkhorn_ot(cost, weights, epsilon, num_iter)
    
    # Barycentric projection
    resampled = tf.matmul(P, particles, transpose_a=True) * tf.cast(N, tf.float32)
    new_log_weights = tf.zeros(N, dtype=tf.float32)
    
    return resampled, new_log_weights


def hybrid_resample(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    fno: FourierOTOperator,
    epsilon: float,
    num_refine: int = 5
) -> tuple:
    """Hybrid: FNO warm-start + Sinkhorn refinement."""
    N = particles.shape[0]
    weights = tf.nn.softmax(log_weights, axis=0)
    
    # Compute cost
    diff = tf.expand_dims(particles, 1) - tf.expand_dims(particles, 0)
    cost = tf.reduce_sum(tf.square(diff), axis=-1)
    
    # FNO prediction (warm-start)
    P_init = fno(cost, weights, epsilon, training=False)
    
    # Refine with Sinkhorn
    # Convert P_init to log space for Sinkhorn initialization
    log_P_init = tf.math.log(P_init + 1e-20)
    
    # Extract dual variables (approximate)
    log_u = tf.reduce_logsumexp(log_P_init, axis=1)
    log_v = tf.reduce_logsumexp(log_P_init, axis=0)
    
    log_a = tf.math.log(weights + 1e-20)
    log_b = tf.math.log(tf.fill([N], 1.0 / tf.cast(N, tf.float32)))
    log_k = -cost / epsilon
    
    # Sinkhorn refinement
    for _ in range(num_refine):
        log_u = log_a - tf.reduce_logsumexp(log_k + tf.reshape(log_v, [1, -1]), axis=1)
        log_v = log_b - tf.reduce_logsumexp(log_k + tf.reshape(log_u, [-1, 1]), axis=0)
    
    log_p = log_k + tf.reshape(log_u, [-1, 1]) + tf.reshape(log_v, [1, -1])
    P_refined = tf.exp(log_p)
    
    # Barycentric projection
    resampled = tf.matmul(P_refined, particles, transpose_a=True) * tf.cast(N, tf.float32)
    new_log_weights = tf.zeros(N, dtype=tf.float32)
    
    return resampled, new_log_weights


def run_particle_filter(
    model,
    observations: np.ndarray,
    N_particles: int,
    resampling_method: str,
    networks: dict = None,
    epsilon: float = 0.1
) -> dict:
    """
    Run particle filter with specified resampling method.
    
    Returns:
        results: dict with estimates, runtime, ess_history, etc.
    """
    T = len(observations)
    estimates = []
    ess_history = []
    runtimes = []
    
    # Initialize
    particles = tf.random.normal((N_particles, 1), dtype=tf.float32)
    log_weights = tf.zeros(N_particles, dtype=tf.float32)
    
    for t in range(T):
        # Prediction
        particles = model.transition(particles)
        
        # Update
        log_lik = model.log_likelihood(observations[t], particles)
        log_lik = tf.reshape(log_lik, [-1])
        log_weights = log_weights + log_lik
        
        # Normalize
        log_weights = log_weights - tf.reduce_logsumexp(log_weights)
        weights = tf.exp(log_weights).numpy()
        
        # ESS
        ess = 1.0 / np.sum(weights ** 2)
        ess_history.append(ess)
        
        # Estimate
        mean_est = np.sum(weights * particles.numpy()[:, 0])
        estimates.append(mean_est)
        
        # Resampling
        if ess < N_particles / 2:
            start_time = time.time()
            
            if resampling_method == 'sinkhorn-100':
                particles, log_weights = sinkhorn_resample(particles, log_weights, epsilon, 100)
            elif resampling_method == 'sinkhorn-30':
                particles, log_weights = sinkhorn_resample(particles, log_weights, epsilon, 30)
            elif resampling_method == 'mgradnet':
                # Compute statistics
                stats = compute_statistics(particles, tf.exp(log_weights))
                innovation = observations[t].numpy() - model.observation(
                    tf.constant([[mean_est]], dtype=tf.float32)
                ).numpy()
                stats['innovation'] = tf.constant(innovation, dtype=tf.float32)
                
                model_params = model.get_params()
                particles, log_weights = neural_ot_resample(
                    particles, log_weights, networks['mgradnet'],
                    model_params, observations[t], stats['innovation']
                )
            elif resampling_method == 'fno':
                weights_norm = tf.nn.softmax(log_weights, axis=0)
                diff = tf.expand_dims(particles, 1) - tf.expand_dims(particles, 0)
                cost = tf.reduce_sum(tf.square(diff), axis=-1)
                P = networks['fno'](cost, weights_norm, epsilon, training=False)
                particles = tf.matmul(P, particles, transpose_a=True) * float(N_particles)
                log_weights = tf.zeros(N_particles, dtype=tf.float32)
            elif resampling_method == 'hybrid':
                particles, log_weights = hybrid_resample(
                    particles, log_weights, networks['fno'], epsilon, num_refine=5
                )
            else:
                raise ValueError(f"Unknown method: {resampling_method}")
            
            runtime = time.time() - start_time
            runtimes.append(runtime)
    
    return {
        'estimates': np.array(estimates),
        'ess_history': np.array(ess_history),
        'runtimes': runtimes,
        'avg_runtime': np.mean(runtimes) if runtimes else 0.0
    }


def compute_rmse(estimates: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute RMSE."""
    return np.sqrt(np.mean((estimates - ground_truth) ** 2))


def compute_gradient_variance(
    model,
    observations: np.ndarray,
    N_particles: int,
    resampling_method: str,
    networks: dict = None,
    num_trials: int = 5
) -> float:
    """
    Compute gradient variance w.r.t. model parameters.
    """
    gradients = []
    
    for trial in range(num_trials):
        with tf.GradientTape() as tape:
            # Reinitialize particles
            particles = tf.random.normal((N_particles, 1), dtype=tf.float32)
            log_weights = tf.zeros(N_particles, dtype=tf.float32)
            
            # Run a few steps
            for t in range(min(10, len(observations))):
                particles = model.transition(particles)
                log_lik = model.log_likelihood(observations[t], particles)
                log_lik = tf.reshape(log_lik, [-1])
                log_weights = log_weights + log_lik
                
                # Resample if needed
                weights = tf.nn.softmax(log_weights, axis=0)
                ess = 1.0 / tf.reduce_sum(tf.square(weights))
                
                if ess < N_particles / 2:
                    if resampling_method == 'sinkhorn-100':
                        particles, log_weights = sinkhorn_resample(particles, log_weights, 0.1, 100)
                    elif resampling_method == 'fno':
                        weights_norm = tf.nn.softmax(log_weights, axis=0)
                        diff = tf.expand_dims(particles, 1) - tf.expand_dims(particles, 0)
                        cost = tf.reduce_sum(tf.square(diff), axis=-1)
                        P = networks['fno'](cost, weights_norm, 0.1, training=False)
                        particles = tf.matmul(P, particles, transpose_a=True) * float(N_particles)
                        log_weights = tf.zeros(N_particles, dtype=tf.float32)
            
            # Final estimate
            final_weights = tf.nn.softmax(log_weights, axis=0)
            final_estimate = tf.reduce_sum(final_weights * particles[:, 0])
        
        # Get gradients w.r.t. model parameters (if available)
        if hasattr(model, 'alpha'):
            grad = tape.gradient(final_estimate, model.alpha)
            if grad is not None:
                gradients.append(grad.numpy())
    
    if gradients:
        return float(np.var(gradients))
    else:
        return 0.0


def benchmark_sv_model(
    networks: dict,
    T: int = 100,
    N_particles: int = 100,
    num_runs: int = 10,
    epsilon: float = 0.1
):
    """Benchmark on Stochastic Volatility model."""
    print("\n" + "="*70)
    print("Benchmarking on Stochastic Volatility Model")
    print("="*70)
    
    methods = ['sinkhorn-100', 'sinkhorn-30', 'fno', 'hybrid']
    if networks.get('mgradnet'):
        methods.insert(2, 'mgradnet')
    
    results = {method: {'rmse': [], 'avg_ess': [], 'runtime': [], 'grad_var': []}
               for method in methods}
    
    for run in tqdm(range(num_runs), desc="Runs"):
        # Generate data
        model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
        x_true, observations = model.sample_trajectory(T=T, x0=0.0, seed=42 + run)
        
        for method in methods:
            # Run particle filter
            result = run_particle_filter(
                model, observations, N_particles, method, networks, epsilon
            )
            
            # Compute metrics
            rmse = compute_rmse(result['estimates'], x_true)
            avg_ess = np.mean(result['ess_history'])
            runtime = result['avg_runtime'] * 1000  # Convert to ms
            
            results[method]['rmse'].append(rmse)
            results[method]['avg_ess'].append(avg_ess)
            results[method]['runtime'].append(runtime)
        
        # Compute gradient variance (only once)
        if run == 0:
            for method in methods:
                if method in ['sinkhorn-100', 'fno']:
                    grad_var = compute_gradient_variance(
                        model, observations, N_particles, method, networks
                    )
                    results[method]['grad_var'].append(grad_var)
    
    # Aggregate results
    summary = []
    for method in methods:
        summary.append({
            'Method': method,
            'RMSE': f"{np.mean(results[method]['rmse']):.4f} ± {np.std(results[method]['rmse']):.4f}",
            'ESS': f"{np.mean(results[method]['avg_ess']):.1f} ± {np.std(results[method]['avg_ess']):.1f}",
            'Runtime (ms)': f"{np.mean(results[method]['runtime']):.2f} ± {np.std(results[method]['runtime']):.2f}",
            'Speedup': f"{np.mean(results['sinkhorn-100']['runtime']) / np.mean(results[method]['runtime']):.1f}x"
        })
    
    df = pd.DataFrame(summary)
    print("\n" + df.to_string(index=False))
    
    return df, results


def plot_results(results: dict, save_path: str):
    """Plot benchmark results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = list(results.keys())
    colors = sns.color_palette("husl", len(methods))
    
    # RMSE
    axes[0, 0].bar(methods, [np.mean(results[m]['rmse']) for m in methods], color=colors)
    axes[0, 0].errorbar(methods, [np.mean(results[m]['rmse']) for m in methods],
                       yerr=[np.std(results[m]['rmse']) for m in methods],
                       fmt='none', color='black', capsize=5)
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # ESS
    axes[0, 1].bar(methods, [np.mean(results[m]['avg_ess']) for m in methods], color=colors)
    axes[0, 1].errorbar(methods, [np.mean(results[m]['avg_ess']) for m in methods],
                       yerr=[np.std(results[m]['avg_ess']) for m in methods],
                       fmt='none', color='black', capsize=5)
    axes[0, 1].set_ylabel('ESS')
    axes[0, 1].set_title('Effective Sample Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Runtime
    axes[1, 0].bar(methods, [np.mean(results[m]['runtime']) for m in methods], color=colors)
    axes[1, 0].errorbar(methods, [np.mean(results[m]['runtime']) for m in methods],
                       yerr=[np.std(results[m]['runtime']) for m in methods],
                       fmt='none', color='black', capsize=5)
    axes[1, 0].set_ylabel('Runtime (ms)')
    axes[1, 0].set_title('Computational Efficiency')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Speedup
    baseline_runtime = np.mean(results['sinkhorn-100']['runtime'])
    speedups = [baseline_runtime / np.mean(results[m]['runtime']) for m in methods]
    axes[1, 1].bar(methods, speedups, color=colors)
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', label='Baseline')
    axes[1, 1].set_ylabel('Speedup')
    axes[1, 1].set_title('Speedup vs Sinkhorn-100')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark neural OT resampling')
    parser.add_argument('--mgradnet_weights', type=str,
                       help='Path to mGradNet weights')
    parser.add_argument('--fno_weights', type=str, required=True,
                       help='Path to FNO weights')
    parser.add_argument('--model', type=str, choices=['sv', 'nonlinear'], default='sv',
                       help='Which model to benchmark')
    parser.add_argument('--T', type=int, default=100,
                       help='Trajectory length')
    parser.add_argument('--N_particles', type=int, default=100,
                       help='Number of particles')
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of runs for statistics')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Neural OT Resampling Benchmark")
    print("="*70)
    
    # Load networks
    networks = {}
    
    if args.mgradnet_weights:
        print(f"Loading mGradNet from {args.mgradnet_weights}...")
        mgradnet = OTResamplingNetwork(state_dim=1, hidden_dim=256)
        mgradnet.load_weights(args.mgradnet_weights)
        networks['mgradnet'] = mgradnet
        print("✓ mGradNet loaded")
    
    print(f"Loading FNO from {args.fno_weights}...")
    fno = FourierOTOperator(modes=16, width=64)
    fno.load_weights(args.fno_weights)
    networks['fno'] = fno
    print("✓ FNO loaded")
    
    # Run benchmark
    if args.model == 'sv':
        df, results = benchmark_sv_model(
            networks, args.T, args.N_particles, args.num_runs
        )
    else:
        raise NotImplementedError("Nonlinear SSM benchmark not yet implemented")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, 'benchmark_results.csv'), index=False)
    print(f"\n✓ Saved results to {args.output_dir}/benchmark_results.csv")
    
    # Plot
    plot_results(results, os.path.join(args.output_dir, 'benchmark_plots.png'))
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)


if __name__ == '__main__':
    main()
