"""
Data Collection for Neural OT Training

Generates training data by running particle filters on diverse SSMs
and collecting (particles, weights, θ, y_t, P*_t) tuples.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import argparse

from src.models.sv_model import StochasticVolatilityModel
from src.models.nonlinear_ssm import NonlinearSSM
from src.filters.particle_filter import ParticleFilter


def sinkhorn_ot(
    cost: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float = 0.1,
    num_iterations: int = 100
) -> tf.Tensor:
    """
    Ground truth Sinkhorn algorithm for OT.
    
    Args:
        cost: (N, N) cost matrix
        weights: (N,) source distribution (normalized)
        epsilon: entropic regularization
        num_iterations: number of Sinkhorn iterations
    
    Returns:
        transport_plan: (N, N) optimal transport plan
    """
    N = tf.shape(weights)[0]
    
    # Target: uniform distribution
    log_a = tf.math.log(weights + 1e-20)
    log_b = tf.math.log(tf.fill([N], 1.0 / tf.cast(N, tf.float32)))
    
    # Kernel: K = exp(-C/ε)
    log_k = -cost / epsilon
    
    # Sinkhorn iterations
    log_u = tf.zeros_like(log_a)
    log_v = tf.zeros_like(log_b)
    
    for _ in range(num_iterations):
        log_u = log_a - tf.reduce_logsumexp(log_k + tf.reshape(log_v, [1, -1]), axis=1)
        log_v = log_b - tf.reduce_logsumexp(log_k + tf.reshape(log_u, [-1, 1]), axis=0)
    
    # Transport plan: P = diag(u) K diag(v)
    log_p = log_k + tf.reshape(log_u, [-1, 1]) + tf.reshape(log_v, [1, -1])
    transport_plan = tf.exp(log_p)
    
    return transport_plan


def generate_sv_data(
    num_trajectories: int,
    T: int,
    N_particles: int,
    param_ranges: dict,
    epsilon: float = 0.1,
    num_sinkhorn_iter: int = 100,
    seed: int = 42
) -> list:
    """
    Generate training data from Stochastic Volatility model.
    
    Args:
        num_trajectories: number of trajectories to generate
        T: length of each trajectory
        N_particles: number of particles
        param_ranges: dict with ranges for each parameter
        epsilon: Sinkhorn regularization
        num_sinkhorn_iter: Sinkhorn iterations for ground truth
        seed: random seed
    
    Returns:
        data: list of dicts with training examples
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    data = []
    
    for traj_idx in tqdm(range(num_trajectories), desc="SV trajectories"):
        # Sample random parameters
        alpha = np.random.uniform(*param_ranges['alpha'])
        sigma = np.random.uniform(*param_ranges['sigma'])
        beta = np.random.uniform(*param_ranges['beta'])
        
        model = StochasticVolatilityModel(alpha=alpha, sigma=sigma, beta=beta)
        
        # Generate trajectory
        x_true, observations = model.sample_trajectory(T=T, x0=0.0, seed=seed + traj_idx)
        
        # Run particle filter
        pf = ParticleFilter(model, N_particles)
        particles = tf.random.normal((N_particles, 1), dtype=tf.float32)
        
        for t in range(T):
            # Prediction
            particles = model.transition(particles)
            
            # Update weights
            log_weights = model.log_likelihood(observations[t], particles)
            log_weights = tf.reshape(log_weights, [-1])
            weights = tf.nn.softmax(log_weights, axis=0).numpy()
            
            # Check if resampling needed (ESS < N/2)
            ess = 1.0 / np.sum(weights ** 2)
            if ess < N_particles / 2:
                # Compute cost matrix
                diff = tf.expand_dims(particles, 1) - tf.expand_dims(particles, 0)
                cost = tf.reduce_sum(tf.square(diff), axis=-1)
                
                # Ground truth: high-quality Sinkhorn
                P_true = sinkhorn_ot(cost, tf.constant(weights, dtype=tf.float32), 
                                    epsilon, num_sinkhorn_iter)
                
                # Compute statistics
                mean = np.sum(weights[:, None] * particles.numpy(), axis=0)
                centered = particles.numpy() - mean
                cov = np.sum(weights * centered[:, 0] ** 2)
                weight_entropy = -np.sum(weights * np.log(weights + 1e-10))
                innovation = observations[t].numpy() - beta * np.exp(mean / 2.0)
                
                # Store example
                data.append({
                    'particles': particles.numpy(),
                    'weights': weights,
                    'cost': cost.numpy(),
                    'P_true': P_true.numpy(),
                    'model_params': np.array([alpha, sigma, beta], dtype=np.float32),
                    'observation': observations[t].numpy(),
                    'mean': mean,
                    'cov': np.array([[cov]], dtype=np.float32),
                    'ess': ess,
                    'weight_entropy': weight_entropy,
                    'innovation': innovation,
                    'epsilon': epsilon,
                    'model_type': 'sv'
                })
                
                # Resample for next iteration
                indices = np.random.choice(N_particles, N_particles, p=weights)
                particles = tf.gather(particles, indices)
    
    return data


def generate_nonlinear_data(
    num_trajectories: int,
    T: int,
    N_particles: int,
    param_ranges: dict,
    epsilon: float = 0.1,
    num_sinkhorn_iter: int = 100,
    seed: int = 42
) -> list:
    """
    Generate training data from Nonlinear SSM (Andrieu 2010).
    """
    np.random.seed(seed + 1000)
    tf.random.set_seed(seed + 1000)
    
    data = []
    
    for traj_idx in tqdm(range(num_trajectories), desc="Nonlinear SSM trajectories"):
        # Sample random parameters
        sigma_V = np.random.uniform(*param_ranges['sigma_V'])
        sigma_W = np.random.uniform(*param_ranges['sigma_W'])
        
        model = NonlinearSSM(sigma_V=sigma_V, sigma_W=sigma_W)
        
        # Generate trajectory
        x_true, observations = model.sample_trajectory(T=T, x0=0.0, seed=seed + traj_idx + 1000)
        
        # Run particle filter
        pf = ParticleFilter(model, N_particles)
        particles = tf.random.normal((N_particles, 1), dtype=tf.float32)
        
        for t in range(T):
            # Prediction with time index
            model.reset_time()
            for step in range(t + 1):
                if step < t:
                    model.transition(particles)
                else:
                    particles = model.transition(particles)
            
            # Update weights
            log_weights = model.log_likelihood(observations[t], particles)
            log_weights = tf.reshape(log_weights, [-1])
            weights = tf.nn.softmax(log_weights, axis=0).numpy()
            
            # Check if resampling needed
            ess = 1.0 / np.sum(weights ** 2)
            if ess < N_particles / 2:
                # Compute cost matrix
                diff = tf.expand_dims(particles, 1) - tf.expand_dims(particles, 0)
                cost = tf.reduce_sum(tf.square(diff), axis=-1)
                
                # Ground truth Sinkhorn
                P_true = sinkhorn_ot(cost, tf.constant(weights, dtype=tf.float32),
                                    epsilon, num_sinkhorn_iter)
                
                # Compute statistics
                mean = np.sum(weights[:, None] * particles.numpy(), axis=0)
                centered = particles.numpy() - mean
                cov = np.sum(weights * centered[:, 0] ** 2)
                weight_entropy = -np.sum(weights * np.log(weights + 1e-10))
                
                # Innovation (approximate)
                pred_obs = mean ** 2 / 20.0
                innovation = observations[t].numpy() - pred_obs
                
                # Store example
                data.append({
                    'particles': particles.numpy(),
                    'weights': weights,
                    'cost': cost.numpy(),
                    'P_true': P_true.numpy(),
                    'model_params': np.array([sigma_V, sigma_W], dtype=np.float32),
                    'observation': observations[t].numpy(),
                    'mean': mean,
                    'cov': np.array([[cov]], dtype=np.float32),
                    'ess': ess,
                    'weight_entropy': weight_entropy,
                    'innovation': innovation,
                    'epsilon': epsilon,
                    'model_type': 'nonlinear'
                })
                
                # Resample
                indices = np.random.choice(N_particles, N_particles, p=weights)
                particles = tf.gather(particles, indices)
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Collect training data for neural OT')
    parser.add_argument('--num_trajectories', type=int, default=1000,
                       help='Number of trajectories per model')
    parser.add_argument('--T', type=int, default=50,
                       help='Trajectory length')
    parser.add_argument('--N_particles', type=int, default=100,
                       help='Number of particles')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Sinkhorn regularization')
    parser.add_argument('--num_sinkhorn_iter', type=int, default=100,
                       help='Sinkhorn iterations for ground truth')
    parser.add_argument('--output', type=str, default='data/ot_training_data.npz',
                       help='Output file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Neural OT Training Data Collection")
    print("="*70)
    print(f"Trajectories per model: {args.num_trajectories}")
    print(f"Trajectory length: {args.T}")
    print(f"Particles: {args.N_particles}")
    print(f"Sinkhorn iterations: {args.num_sinkhorn_iter}")
    print()
    
    # Parameter ranges
    sv_param_ranges = {
        'alpha': (0.85, 0.99),
        'sigma': (0.5, 2.0),
        'beta': (0.3, 0.7)
    }
    
    nonlinear_param_ranges = {
        'sigma_V': (2.0, 4.0),
        'sigma_W': (0.5, 1.5)
    }
    
    # Generate data
    print("Generating SV model data...")
    sv_data = generate_sv_data(
        args.num_trajectories,
        args.T,
        args.N_particles,
        sv_param_ranges,
        args.epsilon,
        args.num_sinkhorn_iter,
        args.seed
    )
    print(f"✓ Generated {len(sv_data)} SV examples")
    
    print("\nGenerating Nonlinear SSM data...")
    nonlinear_data = generate_nonlinear_data(
        args.num_trajectories // 2,  # Generate fewer (more expensive)
        args.T,
        args.N_particles,
        nonlinear_param_ranges,
        args.epsilon,
        args.num_sinkhorn_iter,
        args.seed
    )
    print(f"✓ Generated {len(nonlinear_data)} Nonlinear SSM examples")
    
    # Combine
    all_data = sv_data + nonlinear_data
    print(f"\nTotal examples: {len(all_data)}")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Convert to numpy arrays
    save_dict = {}
    for key in all_data[0].keys():
        if key == 'model_type':
            save_dict[key] = np.array([d[key] for d in all_data], dtype='U10')
        else:
            # Stack all examples
            save_dict[key] = np.array([d[key] for d in all_data], dtype=object)
    
    np.savez_compressed(args.output, **save_dict)
    print(f"\n✓ Saved to {args.output}")
    print(f"  File size: {os.path.getsize(args.output) / 1e6:.1f} MB")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Particle counts: {args.N_particles}")
    print(f"  Weight entropy range: [{np.min([d['weight_entropy'] for d in all_data]):.3f}, "
          f"{np.max([d['weight_entropy'] for d in all_data]):.3f}]")
    print(f"  ESS range: [{np.min([d['ess'] for d in all_data]):.1f}, "
          f"{np.max([d['ess'] for d in all_data]):.1f}]")
    
    sv_examples = len(sv_data)
    nonlinear_examples = len(nonlinear_data)
    print(f"\nModel distribution:")
    print(f"  SV: {sv_examples} ({100*sv_examples/len(all_data):.1f}%)")
    print(f"  Nonlinear: {nonlinear_examples} ({100*nonlinear_examples/len(all_data):.1f}%)")
    
    print("\n" + "="*70)
    print("Data collection complete!")
    print("="*70)


if __name__ == '__main__':
    main()
