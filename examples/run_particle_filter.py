import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sv_model import StochasticVolatilityModel
from src.filters.particle_filter import StandardParticleFilter

def main():
    # 1. Setup Environment
    tf.random.set_seed(42)
    
    # 2. Initialize Model
    print("Initializing Model...")
    sv_model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
    
    # 3. Generate Synthetic Data (Ground Truth)
    print("Generating Synthetic Data...")
    T = 100
    # Use the visualization script logic or manual generation here
    # We will do a quick generation loop similar to before
    x_t = tf.random.normal((1,), stddev=1.0) # Start rough
    true_states = []
    observations = []
    
    for _ in range(T):
        x_t = sv_model.transition(tf.reshape(x_t, [1]))[0]
        y_t = sv_model.observation(tf.reshape(x_t, [1]))[0]
        true_states.append(x_t)
        observations.append(y_t)
        
    observations_tensor = tf.stack(observations)
    observations_tensor = tf.reshape(observations_tensor, (-1, 1)) # [T, 1]
    
    # 4. Initialize Particle Filter
    # Pass the model instance to the filter
    pf = StandardParticleFilter(model=sv_model, num_particles=1000)
    
    # 5. Run Filter
    print("Running Particle Filter...")
    estimates, ess_history = pf.run(observations_tensor)
    
    # 6. Visualization
    print("Plotting...")
    true_states_np = np.array(true_states)
    estimates_np = estimates.numpy()
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Tracking
    plt.subplot(2, 1, 1)
    plt.plot(true_states_np, 'k-', label='True State', linewidth=1.5)
    plt.plot(true_states_np, 'k-', label='True State', linewidth=1.5)
    plt.plot(estimates_np, 'g--', label='PF Estimate (N=1000)', linewidth=1.5)
    plt.title('Part 1.II.c: Particle Filter Tracking Performance')
    plt.ylabel('Log-Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: ESS
    plt.subplot(2, 1, 2)
    plt.plot(ess_history, 'm.-', label='Effective Sample Size (ESS)')
    plt.axhline(y=500, color='r', linestyle=':', label='Resampling Threshold (N/2)')
    plt.title('Particle Degeneracy Analysis')
    plt.ylabel('ESS')
    plt.xlabel('Time Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'particle_filter_tracking.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((estimates_np - true_states_np.reshape(-1, 1))**2))
    print(f"PF RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()