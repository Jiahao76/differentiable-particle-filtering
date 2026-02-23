import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Add the project root to the system path to allow importing from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sv_model import StochasticVolatilityModel

def sample_initial_state(model: StochasticVolatilityModel, num_samples: int = 1) -> tf.Tensor:
    """
    Helper function to sample from the stationary distribution of the AR(1) process.
    
    Stationary Variance = sigma^2 / (1 - alpha^2)
    
    Args:
        model: The Stochastic Volatility Model instance.
        num_samples: Number of samples to generate.
        
    Returns:
        tf.Tensor: Initial state samples.
    """
    # Calculate stationary variance
    stationary_var = model.sigma**2 / (1.0 - model.alpha**2)
    stationary_std = tf.sqrt(stationary_var)
    
    return tf.random.normal((num_samples,), mean=0.0, stddev=stationary_std)

def run_simulation():
    """
    Run a simulation of the Stochastic Volatility Model and plot the results.
    Replicates Figure 1 style from Doucet et al. (2009).
    """
    print("Initializing Stochastic Volatility Model...")
    
    # 1. Initialize the model with standard parameters
    # alpha=0.91, sigma=1.0, beta=0.5
    sv_model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
    
    # 2. Simulation Parameters
    T = 500  # Number of time steps (same as Doucet 09 paper)
    tf.random.set_seed(42) # Ensure reproducibility
    
    # 3. Generate Data
    # Initialize state from stationary distribution
    x_t = sample_initial_state(sv_model, num_samples=1)[0] # Scalar
    
    states = []
    observations = []
    
    print(f"Generating {T} time steps...")
    
    for t in range(T):
        # Store current state
        states.append(x_t)
        
        # Generate observation based on current state: y_t ~ p(y_t | x_t)
        # Note: We pass explicit dimensions to ensure correct shape
        x_curr_tensor = tf.reshape(x_t, [1]) 
        y_t = sv_model.observation(x_curr_tensor)[0]
        observations.append(y_t)
        
        # Transition to next state: x_{t+1} ~ p(x_{t+1} | x_t)
        x_t = sv_model.transition(x_curr_tensor)[0]
        
    # Convert to NumPy for plotting
    states_np = np.array(states)
    observations_np = np.array(observations)
    time_steps = np.arange(T)
    
    # 4. Visualization
    print("Plotting results...")
    
    plt.figure(figsize=(12, 6))
    
    # Plot Volatility (Hidden State)
    plt.plot(time_steps, states_np, 'b-', linewidth=1.0, label='True Log-Volatility ($X_n$)')
    
    # Plot Observations (Returns)
    # Using dots for observations to match the style of particle filter literature
    plt.plot(time_steps, observations_np, 'r.', markersize=4, alpha=0.6, label='Observations ($Y_n$)')
    
    plt.title('Stochastic Volatility Model Simulation\n(Replicating Doucet et al., 2009, Example 4)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'sv_model_simulation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    run_simulation()