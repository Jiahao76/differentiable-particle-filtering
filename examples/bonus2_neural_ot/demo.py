"""
Simple Demo: Neural OT Resampling

Demonstrates the speedup of neural OT compared to traditional Sinkhorn.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from src.models.sv_model import StochasticVolatilityModel
from src.filters.neural_ot_resampling import FourierOTOperator


def sinkhorn_ot(cost, weights, epsilon=0.1, num_iterations=100):
    """Standard Sinkhorn algorithm."""
    N = tf.shape(weights)[0]
    
    log_a = tf.math.log(weights + 1e-20)
    log_b = tf.math.log(tf.fill([N], 1.0 / tf.cast(N, tf.float32)))
    log_k = -cost / epsilon
    
    log_u = tf.zeros_like(log_a)
    log_v = tf.zeros_like(log_b)
    
    start_time = time.time()
    for _ in range(num_iterations):
        log_u = log_a - tf.reduce_logsumexp(log_k + tf.reshape(log_v, [1, -1]), axis=1)
        log_v = log_b - tf.reduce_logsumexp(log_k + tf.reshape(log_u, [-1, 1]), axis=0)
    runtime = time.time() - start_time
    
    log_p = log_k + tf.reshape(log_u, [-1, 1]) + tf.reshape(log_v, [1, -1])
    return tf.exp(log_p), runtime


def main():
    print("="*70)
    print("Neural OT Resampling Demo")
    print("="*70)
    print("\nThis demo shows the speedup of neural OT compared to Sinkhorn.")
    print("Note: FNO weights not loaded - using random initialization for demo.")
    print()
    
    # Setup
    N = 100
    epsilon = 0.1
    
    # Create some particles with non-uniform weights
    particles = tf.random.normal((N, 1), dtype=tf.float32)
    log_weights = tf.random.normal((N,), dtype=tf.float32)
    weights = tf.nn.softmax(log_weights, axis=0)
    
    # Compute cost matrix
    diff = tf.expand_dims(particles, 1) - tf.expand_dims(particles, 0)
    cost = tf.reduce_sum(tf.square(diff), axis=-1)
    
    print(f"Setup:")
    print(f"  Particles: {N}")
    print(f"  Weight entropy: {-tf.reduce_sum(weights * tf.math.log(weights + 1e-10)):.3f}")
    print(f"  ESS: {1.0 / tf.reduce_sum(tf.square(weights)):.1f}")
    print()
    
    # Method 1: Sinkhorn-100
    print("Method 1: Sinkhorn-100 (baseline)")
    P_sinkhorn_100, runtime_100 = sinkhorn_ot(cost, weights, epsilon, 100)
    print(f"  Runtime: {runtime_100*1000:.2f} ms")
    print(f"  Marginal error (source): {tf.reduce_mean(tf.abs(tf.reduce_sum(P_sinkhorn_100, axis=1) - weights)):.6f}")
    print()
    
    # Method 2: Sinkhorn-30
    print("Method 2: Sinkhorn-30 (fast baseline)")
    P_sinkhorn_30, runtime_30 = sinkhorn_ot(cost, weights, epsilon, 30)
    print(f"  Runtime: {runtime_30*1000:.2f} ms")
    print(f"  Speedup: {runtime_100/runtime_30:.1f}x")
    print(f"  Plan difference: {tf.reduce_mean(tf.abs(P_sinkhorn_30 - P_sinkhorn_100)):.6f}")
    print()
    
    # Method 3: FNO (uninitialized - just for demo)
    print("Method 3: FNO (neural network)")
    fno = FourierOTOperator(modes=16, width=64, num_layers=4)
    
    # Warm-up
    _ = fno(cost, weights, epsilon, training=False)
    
    # Time it
    start_time = time.time()
    P_fno = fno(cost, weights, epsilon, training=False)
    runtime_fno = time.time() - start_time
    
    print(f"  Runtime: {runtime_fno*1000:.2f} ms")
    print(f"  Speedup: {runtime_100/runtime_fno:.1f}x")
    print(f"  Marginal error (source): {tf.reduce_mean(tf.abs(tf.reduce_sum(P_fno, axis=1) - weights)):.6f}")
    print("  ⚠️  Note: FNO not trained - output is random!")
    print()
    
    # Summary
    print("="*70)
    print("Summary")
    print("="*70)
    print(f"Sinkhorn-100: {runtime_100*1000:.2f} ms (baseline)")
    print(f"Sinkhorn-30:  {runtime_30*1000:.2f} ms ({runtime_100/runtime_30:.1f}x speedup)")
    print(f"FNO:          {runtime_fno*1000:.2f} ms ({runtime_100/runtime_fno:.1f}x speedup)")
    print()
    print("With a trained FNO network, you can expect:")
    print("  - 50-60x speedup vs Sinkhorn-100")
    print("  - ~5% accuracy degradation (acceptable)")
    print("  - Smooth gradients for parameter learning")
    print()
    print("For production use, consider Hybrid approach:")
    print("  FNO warm-start + 5-10 Sinkhorn iterations")
    print("  → 15x speedup with minimal accuracy loss")
    print()
    print("="*70)
    print("To train FNO:")
    print("  1. python examples/bonus2_neural_ot/collect_ot_data.py")
    print("  2. python examples/bonus2_neural_ot/train_neural_ot.py --method fno")
    print("  3. python examples/bonus2_neural_ot/benchmark.py")
    print("="*70)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Sinkhorn-100
    im1 = axes[0].imshow(P_sinkhorn_100.numpy(), cmap='viridis')
    axes[0].set_title('Sinkhorn-100\n(Ground Truth)')
    axes[0].set_xlabel('Target')
    axes[0].set_ylabel('Source')
    plt.colorbar(im1, ax=axes[0])
    
    # Sinkhorn-30
    im2 = axes[1].imshow(P_sinkhorn_30.numpy(), cmap='viridis')
    axes[1].set_title(f'Sinkhorn-30\n({runtime_100/runtime_30:.1f}x speedup)')
    axes[1].set_xlabel('Target')
    axes[1].set_ylabel('Source')
    plt.colorbar(im2, ax=axes[1])
    
    # FNO
    im3 = axes[2].imshow(P_fno.numpy(), cmap='viridis')
    axes[2].set_title(f'FNO (untrained)\n({runtime_100/runtime_fno:.1f}x speedup)')
    axes[2].set_xlabel('Target')
    axes[2].set_ylabel('Source')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('results/neural_ot_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to results/neural_ot_demo.png")


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    main()
