import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from typing import Tuple, Dict


# ============================================================================
# Part 1: Standard Particle Filter 
# ============================================================================

def standard_particle_filter(
    observations: np.ndarray,
    n_particles: int = 1000,
    process_noise: float = 0.5,
    observation_noise: float = 0.3,
    transition_coef: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard Particle Filter for 1D state space model.
    
    State Space Model:
        x_t = transition_coef * x_{t-1} + N(0, process_noise^2)
        y_t = x_t + N(0, observation_noise^2)
    
    Args:
        observations: Observation sequence (T,)
        n_particles: Number of particles
        process_noise: Standard deviation of process noise
        observation_noise: Standard deviation of observation noise
        transition_coef: State transition coefficient
        
    Returns:
        estimates: State estimates (T,)
        particles_history: Particle positions over time (T, n_particles)
    """
    T = len(observations)
    N = n_particles
    
    # Initialize particles and weights
    particles = np.random.randn(N)
    weights = np.ones(N) / N
    
    estimates = []
    particles_history = []
    
    # Main filter loop
    for t in range(T):
        # Step 1: Prediction - propagate particles through dynamics
        particles = transition_coef * particles + np.random.randn(N) * process_noise
        
        # Step 2: Weight Update - compute likelihood for each particle
        # p(y_t | x_t) = N(y_t; x_t, observation_noise^2)
        likelihood = np.exp(-0.5 * ((observations[t] - particles) / observation_noise)**2)
        likelihood /= (np.sqrt(2 * np.pi) * observation_noise)  # Normalization constant
        
        # Update weights
        weights *= likelihood
        weights += 1e-300  # Avoid numerical underflow
        weights /= np.sum(weights)  # Normalize
        
        # Step 3: State Estimation - weighted average
        estimate = np.sum(particles * weights)
        estimates.append(estimate)
        particles_history.append(particles.copy())
        
        # Step 4: Resampling - avoid particle degeneracy
        # Compute effective sample size
        ess = 1.0 / np.sum(weights**2)
        
        # Resample if ESS is too low
        if ess < N / 2:
            indices = np.random.choice(N, N, p=weights)
            particles = particles[indices]
            weights = np.ones(N) / N
    
    return np.array(estimates), np.array(particles_history)


def generate_synthetic_data(
    T: int,
    process_noise: float = 0.5,
    observation_noise: float = 0.3,
    transition_coef: float = 0.9,
    initial_state: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data from state space model.
    
    Args:
        T: Number of time steps
        process_noise: Standard deviation of process noise
        observation_noise: Standard deviation of observation noise
        transition_coef: State transition coefficient
        initial_state: Initial state value
        
    Returns:
        true_states: True hidden states (T,)
        observations: Noisy observations (T,)
    """
    true_states = np.zeros(T)
    true_states[0] = initial_state
    
    # Generate state trajectory
    for t in range(1, T):
        true_states[t] = transition_coef * true_states[t-1] + np.random.randn() * process_noise
    
    # Generate observations
    observations = true_states + np.random.randn(T) * observation_noise
    
    return true_states, observations


def visualize_particle_filter_results(
    true_states: np.ndarray,
    observations: np.ndarray,
    estimates: np.ndarray,
    particles_history: np.ndarray = None,
    save_path: str = None
):
    """
    Visualize particle filter results.
    
    Args:
        true_states: True hidden states (T,)
        observations: Observations (T,)
        estimates: Filter estimates (T,)
        particles_history: Particle positions (T, N), optional
        save_path: Path to save figure, optional
    """
    T = len(true_states)
    time_steps = np.arange(T)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: State estimation
    ax1 = axes[0]
    ax1.plot(time_steps, true_states, 'b-', linewidth=2, label='True State')
    ax1.plot(time_steps, observations, 'r.', alpha=0.5, markersize=4, label='Observations')
    ax1.plot(time_steps, estimates, 'g-', linewidth=2, label='PF Estimate')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Particle Filter Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Estimation error
    ax2 = axes[1]
    errors = np.abs(estimates - true_states)
    ax2.plot(time_steps, errors, 'r-', linewidth=1.5)
    ax2.axhline(y=np.mean(errors), color='k', linestyle='--', 
                label=f'Mean Error: {np.mean(errors):.3f}')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Estimation Error Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# Part 2: Differentiable Particle Filter (TensorFlow Implementation)
# ============================================================================

def soft_resample(weights: tf.Tensor, temperature: float = 0.1, eps: float = 1e-8) -> tf.Tensor:
    """
    Differentiable resampling using Gumbel-Softmax trick.
    
    The Gumbel-Softmax provides a continuous, differentiable approximation
    to categorical sampling by:
    1. Adding Gumbel noise to log-probabilities
    2. Applying softmax with temperature parameter
    
    Args:
        weights: Particle weights (N,)
        temperature: Softmax temperature (lower = more discrete)
        eps: Small constant for numerical stability
        
    Returns:
        soft_weights: Differentiable resampled weights (N,)
    """
    # Sample Gumbel noise: G ~ Gumbel(0, 1)
    # Using: G = -log(-log(U)) where U ~ Uniform(0, 1)
    uniform_samples = tf.random.uniform(tf.shape(weights), dtype=weights.dtype)
    gumbel_noise = -tf.math.log(-tf.math.log(uniform_samples + eps) + eps)
    
    # Add Gumbel noise to log-weights
    log_weights = tf.math.log(weights + eps)
    logits = log_weights + gumbel_noise
    
    # Apply softmax with temperature
    soft_weights = tf.nn.softmax(logits / temperature)
    
    return soft_weights


def differentiable_particle_filter(
    observations: tf.Tensor,
    n_particles: int = 100,
    process_noise: float = 0.5,
    observation_noise: float = 0.3,
    transition_coef: float = 0.9,
    temperature: float = 0.1
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Differentiable Particle Filter using soft resampling.
    
    This version allows gradients to flow through the entire filtering process,
    enabling end-to-end learning of model parameters.
    
    Args:
        observations: Observation sequence (T,)
        n_particles: Number of particles
        process_noise: Standard deviation of process noise
        observation_noise: Standard deviation of observation noise
        transition_coef: State transition coefficient
        temperature: Temperature for soft resampling
        
    Returns:
        estimates: State estimates (T,)
        all_weights: Weights history for analysis (T, N)
    """
    T = tf.shape(observations)[0]
    N = n_particles
    
    # Initialize particles and weights
    particles = tf.random.normal([N], dtype=tf.float32)
    weights = tf.ones([N], dtype=tf.float32) / N
    
    estimates = []
    all_weights = []
    
    for t in range(T):
        # Prediction step
        noise = tf.random.normal([N], dtype=tf.float32) * process_noise
        particles = transition_coef * particles + noise
        
        # Weight update
        y_t = observations[t]
        squared_error = ((y_t - particles) / observation_noise) ** 2
        likelihood = tf.exp(-0.5 * squared_error)
        
        weights = weights * likelihood
        weights = weights / tf.reduce_sum(weights)
        
        # Soft resampling (differentiable!)
        weights = soft_resample(weights, temperature=temperature)
        
        # State estimation
        estimate = tf.reduce_sum(particles * weights)
        estimates.append(estimate)
        all_weights.append(weights)
    
    return tf.stack(estimates), tf.stack(all_weights)


# ============================================================================
# Part 3: Learnable State Space Model 
# ============================================================================

class LearnableSSM(keras.Model):
    """
    Learnable State Space Model with Differentiable Particle Filter.
    
    This model learns the transition coefficient by minimizing prediction error
    through gradient descent. The differentiable particle filter allows gradients
    to flow from the loss function back to the model parameters.
    
    Attributes:
        transition_coef: Learnable state transition coefficient (trainable)
        process_noise: Process noise standard deviation (fixed)
        observation_noise: Observation noise standard deviation (fixed)
    """
    
    def __init__(
        self,
        initial_transition_coef: float = 0.5,
        process_noise: float = 0.5,
        observation_noise: float = 0.3,
        n_particles: int = 500,
        temperature: float = 0.1
    ):
        """
        Initialize Learnable SSM.
        
        Args:
            initial_transition_coef: Initial value for transition coefficient
            process_noise: Process noise standard deviation
            observation_noise: Observation noise standard deviation
            n_particles: Number of particles for filtering
            temperature: Temperature for soft resampling
        """
        super(LearnableSSM, self).__init__()
        
        # Learnable parameter (initialized incorrectly to test learning)
        self.transition_coef = tf.Variable(
            initial_transition_coef, 
            dtype=tf.float32,
            trainable=True,
            name='transition_coefficient'
        )
        
        # Fixed parameters
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.n_particles = n_particles
        self.temperature = temperature
    
    def call(self, observations: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Forward pass: run differentiable particle filter.
        
        Args:
            observations: Observation sequence (T,)
            training: Whether in training mode
            
        Returns:
            estimates: State estimates (T,)
        """
        T = tf.shape(observations)[0]
        N = self.n_particles
        
        # Initialize
        particles = tf.random.normal([N], dtype=tf.float32)
        weights = tf.ones([N], dtype=tf.float32) / N
        
        estimates = []
        
        for t in range(T):
            # Prediction with learnable coefficient
            noise = tf.random.normal([N], dtype=tf.float32) * self.process_noise
            particles = self.transition_coef * particles + noise
            
            # Weight update
            y_t = observations[t]
            squared_error = ((y_t - particles) / self.observation_noise) ** 2
            likelihood = tf.exp(-0.5 * squared_error)
            
            weights = weights * likelihood
            weights = weights / tf.reduce_sum(weights+ 1e-8)
            
            # Soft resampling
            if training:
                weights = soft_resample(weights, temperature=self.temperature)
            
            # Estimate
            estimate = tf.reduce_sum(particles * weights)
            estimates.append(estimate)
        
        return tf.stack(estimates)


def train_learnable_ssm(
    observations: np.ndarray,
    true_states: np.ndarray,
    n_epochs: int = 100,
    learning_rate: float = 0.005,
    initial_coef: float = 0.5,
    verbose: bool = True
) -> Tuple[LearnableSSM, Dict]:
    
    # Convert to TensorFlow tensors
    obs_tensor = tf.constant(observations, dtype=tf.float32)
    
    # Initialize model
    model = LearnableSSM(initial_transition_coef=initial_coef)
    
    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    history = {'loss': [], 'transition_coef': [], 'rmse': []}
    
    if verbose:
        print(f"Initial coef: {model.transition_coef.numpy():.4f} | Target: 0.9000")
        print("-" * 60)
    
    for epoch in range(n_epochs):
        with tf.GradientTape() as tape:
            predictions = model(obs_tensor, training=True)
            loss = tf.reduce_mean((obs_tensor - predictions) ** 2)
        
        train_vars = [model.transition_coef]
        gradients = tape.gradient(loss, train_vars)
        
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        
        optimizer.apply_gradients(zip(gradients, train_vars))
        
        current_loss = loss.numpy()
        current_coef = model.transition_coef.numpy()

        history['loss'].append(current_loss)
        history['transition_coef'].append(current_coef)
        
        predictions_np = predictions.numpy()
        rmse = np.sqrt(np.mean((predictions_np - true_states) ** 2))
        history['rmse'].append(rmse)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {current_loss:.4f} | "
                  f"Coef: {current_coef:.4f} | RMSE: {rmse:.4f}")
    
    return model, history


def visualize_learning_progress(history: Dict, save_path: str = None):
    """
    Visualize the learning progress of the learnable SSM.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure, optional
    """
    epochs = np.arange(1, len(history['loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Training loss
    ax1 = axes[0]
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Transition coefficient convergence
    ax2 = axes[1]
    ax2.plot(epochs, history['transition_coef'], 'g-', linewidth=2, label='Learned')
    ax2.axhline(y=0.9, color='r', linestyle='--', linewidth=2, label='True Value')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Transition Coefficient', fontsize=12)
    ax2.set_title('Parameter Convergence', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RMSE
    ax3 = axes[2]
    ax3.plot(epochs, history['rmse'], 'r-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('RMSE', fontsize=12)
    ax3.set_title('State Estimation Error', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Particle Filter Examples - Improved Implementation")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # ========================================================================
    # Example 1: Standard Particle Filter
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: Standard Particle Filter (NumPy)")
    print("=" * 70)
    
    # Generate synthetic data
    T = 100
    true_states, observations = generate_synthetic_data(T)
    
    print(f"\nGenerated {T} time steps of synthetic data")
    print(f"True transition coefficient: 0.9")
    print(f"Process noise std: 0.5")
    print(f"Observation noise std: 0.3")
    
    # Run standard particle filter
    print("\nRunning standard particle filter with 1000 particles...")
    estimates, particles_history = standard_particle_filter(observations, n_particles=1000)
    
    # Compute performance metrics
    rmse = np.sqrt(np.mean((estimates - true_states) ** 2))
    mae = np.mean(np.abs(estimates - true_states))
    
    print(f"\nPerformance Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_particle_filter_results(true_states, observations, estimates)
    
    # ========================================================================
    # Example 2: Differentiable Particle Filter
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: Differentiable Particle Filter (TensorFlow)")
    print("=" * 70)
    
    # Generate new data
    true_states_tf, observations_tf = generate_synthetic_data(T=50)
    obs_tensor = tf.constant(observations_tf, dtype=tf.float32)
    
    print(f"\nGenerated {len(observations_tf)} time steps of synthetic data")
    print("Running differentiable particle filter with 100 particles...")
    
    # Run differentiable particle filter
    estimates_tf, weights_history = differentiable_particle_filter(obs_tensor, n_particles=100)
    
    # Compute metrics
    estimates_np = estimates_tf.numpy()
    rmse_tf = np.sqrt(np.mean((estimates_np - true_states_tf) ** 2))
    
    print(f"\nPerformance Metrics:")
    print(f"  RMSE: {rmse_tf:.4f}")
    print(f"  Note: Gradients can now flow through the entire filter!")
    
    # ========================================================================
    # Example 3: Learning System Parameters
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Learning Transition Coefficient with DPF")
    print("=" * 70)
    
    # Generate training data (true coefficient = 0.9)
    T_train = 100
    true_states_train, observations_train = generate_synthetic_data(T_train)
    
    print(f"\nGenerated {T_train} time steps for training")
    print("True transition coefficient: 0.9")
    print("Initial guess: 0.5 (intentionally wrong)\n")
    
    # Train model
    model, history = train_learnable_ssm(
        observations_train,
        true_states_train,
        n_epochs=100,
        learning_rate=0.01,
        initial_coef=0.5,
        verbose=True
    )
    
    # Visualize learning progress
    print("\nGenerating learning progress visualization...")
    visualize_learning_progress(history)
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)