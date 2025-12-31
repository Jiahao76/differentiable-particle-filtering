"""
Diagnostic script to visualize and debug particle flow behavior
"""
import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sv_model import StochasticVolatilityModel

def diagnose_flow_step(particles, observation, model, flow_obj):
    """
    Visualize what happens during a single flow step
    """
    beta = 0.5
    
    # Create a range of x values to plot
    x_range = np.linspace(-3, 3, 200)
    
    # Compute log-likelihood landscape
    log_likes = []
    for x_val in x_range:
        x_tensor = tf.constant([[x_val]], dtype=tf.float32)
        ll = model.log_likelihood(observation, x_tensor)
        log_likes.append(ll.numpy()[0])
    
    log_likes = np.array(log_likes)
    likes = np.exp(log_likes - np.max(log_likes))  # Normalize for plotting
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Likelihood landscape
    axes[0, 0].plot(x_range, likes, 'b-', linewidth=2, label='Likelihood p(y|x)')
    axes[0, 0].axvline(observation.numpy()[0], color='r', linestyle='--', 
                       label=f'Observation y={observation.numpy()[0]:.2f}')
    axes[0, 0].hist(particles.numpy(), bins=30, density=True, alpha=0.3, 
                    color='green', label='Particle distribution')
    axes[0, 0].set_xlabel('State x (log-volatility)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Likelihood Landscape vs Particle Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Expected observation function h(x) = β·exp(x/2)
    h_x = beta * np.exp(x_range / 2.0)
    axes[0, 1].plot(x_range, h_x, 'b-', linewidth=2, label='h(x) = β·exp(x/2)')
    axes[0, 1].axhline(observation.numpy()[0], color='r', linestyle='--',
                       label=f'Observed y={observation.numpy()[0]:.2f}')
    axes[0, 1].fill_between(x_range, 0, h_x, alpha=0.2)
    axes[0, 1].set_xlabel('State x')
    axes[0, 1].set_ylabel('Expected observation h(x)')
    axes[0, 1].set_title('Observation Function (Nonlinear)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Gradient of log-likelihood
    gradients = []
    for x_val in x_range:
        x_tensor = tf.Variable([[x_val]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            expected_var = beta**2 * tf.exp(x_tensor)
            log_like = -0.5 * (observation**2 / (expected_var + 1e-8))
        grad = tape.gradient(log_like, x_tensor)
        gradients.append(grad.numpy()[0, 0])
    
    gradients = np.array(gradients)
    axes[1, 0].plot(x_range, gradients, 'b-', linewidth=2)
    axes[1, 0].axhline(0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(observation.numpy()[0], color='r', linestyle='--')
    axes[1, 0].set_xlabel('State x')
    axes[1, 0].set_ylabel('∇_x log p(y|x)')
    axes[1, 0].set_title('Gradient of Log-Likelihood (Flow Direction)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Flow velocity field
    mean_prior = tf.reduce_mean(particles)
    centered = particles - mean_prior
    cov_prior = tf.reduce_mean(centered**2)
    
    velocities = []
    for x_val in x_range:
        x_tensor = tf.Variable([[x_val]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            expected_var = beta**2 * tf.exp(x_tensor)
            log_like = -0.5 * (observation**2 / (expected_var + 1e-8))
        grad = tape.gradient(log_like, x_tensor)
        velocity = cov_prior * grad
        velocities.append(velocity.numpy()[0, 0])
    
    velocities = np.array(velocities)
    axes[1, 1].plot(x_range, velocities, 'b-', linewidth=2)
    axes[1, 1].axhline(0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(observation.numpy()[0], color='r', linestyle='--')
    axes[1, 1].quiver(particles.numpy()[:20, 0], 
                      np.zeros(20),
                      np.ones(20) * 0.01,  # Small horizontal arrows
                      velocities[:20] if len(velocities) >= 20 else np.zeros(20),
                      color='green', alpha=0.5)
    axes[1, 1].set_xlabel('State x')
    axes[1, 1].set_ylabel('Flow velocity v(x)')
    axes[1, 1].set_title('EDH Flow Velocity Field')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Particle Flow Diagnostics (y={observation.numpy()[0]:.3f})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, 'flow_diagnostics.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Diagnostic plot saved to {filepath}")
    plt.close()

def compare_before_after_flow(particles_before, particles_after, observation, model):
    """
    Compare particle distributions before and after flow
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x_range = np.linspace(-4, 4, 200)
    beta = 0.5
    
    # Likelihood
    log_likes = []
    for x_val in x_range:
        x_tensor = tf.constant([[x_val]], dtype=tf.float32)
        ll = model.log_likelihood(observation, x_tensor)
        log_likes.append(ll.numpy()[0])
    likes = np.exp(log_likes - np.max(log_likes))
    
    # Before flow
    axes[0].plot(x_range, likes, 'r-', linewidth=2, label='Target (likelihood)')
    axes[0].hist(particles_before.numpy(), bins=30, density=True, 
                alpha=0.5, color='blue', label='Particles (before)')
    axes[0].set_title('Before Flow')
    axes[0].set_xlabel('State x')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # After flow
    axes[1].plot(x_range, likes, 'r-', linewidth=2, label='Target (likelihood)')
    axes[1].hist(particles_after.numpy(), bins=30, density=True,
                alpha=0.5, color='green', label='Particles (after)')
    axes[1].set_title('After Flow')
    axes[1].set_xlabel('State x')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Overlay comparison
    axes[2].plot(x_range, likes, 'r-', linewidth=2, label='Target')
    axes[2].hist(particles_before.numpy(), bins=30, density=True,
                alpha=0.3, color='blue', label='Before')
    axes[2].hist(particles_after.numpy(), bins=30, density=True,
                alpha=0.3, color='green', label='After')
    axes[2].set_title('Comparison')
    axes[2].set_xlabel('State x')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    filepath = os.path.join(results_dir, 'flow_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to {filepath}")
    plt.close()

def main():
    """
    Run diagnostic on a single timestep
    """
    print("="*60)
    print("DIAGNOSTIC: Particle Flow Behavior")
    print("="*60)
    
    # Setup
    tf.random.set_seed(42)
    model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
    
    # Generate a single observation
    x_true = tf.random.normal((1,))
    for _ in range(5):  # Burn-in
        x_true = model.transition(x_true)
    observation = model.observation(x_true)
    
    print(f"True state: x = {x_true.numpy()[0]:.3f}")
    print(f"Observation: y = {observation.numpy()[0]:.3f}")
    
    # Create particles
    N = 200
    particles_before = tf.random.normal((N, 1), dtype=tf.float32)
    
    # Apply flow (simplified version for diagnosis)
    mean_p = tf.reduce_mean(particles_before)
    centered = particles_before - mean_p
    cov_p = tf.reduce_mean(centered**2)
    
    particles_after = tf.identity(particles_before)
    beta = 0.5
    flow_steps = 20
    epsilon = 0.05
    
    for k in range(flow_steps):
        with tf.GradientTape() as tape:
            tape.watch(particles_after)
            expected_var = beta**2 * tf.exp(particles_after)
            log_like = -0.5 * (observation**2 / (expected_var + 1e-8))
        grad = tape.gradient(log_like, particles_after)
        velocity = cov_p * grad
        velocity = tf.clip_by_value(velocity, -1.0, 1.0)
        particles_after = particles_after + epsilon * velocity
    
    print(f"\nParticle statistics BEFORE flow:")
    print(f"  Mean: {tf.reduce_mean(particles_before).numpy():.3f}")
    print(f"  Std:  {tf.math.reduce_std(particles_before).numpy():.3f}")
    
    print(f"\nParticle statistics AFTER flow:")
    print(f"  Mean: {tf.reduce_mean(particles_after).numpy():.3f}")
    print(f"  Std:  {tf.math.reduce_std(particles_after).numpy():.3f}")
    
    # Create diagnostic plots
    diagnose_flow_step(particles_before, observation, model, None)
    compare_before_after_flow(particles_before, particles_after, observation, model)
    
    print("\n✅ Diagnostic complete! Check results/ directory for plots.")

if __name__ == "__main__":
    main()