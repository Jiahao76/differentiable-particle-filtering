import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.sv_model import StochasticVolatilityModelTF

class StandardParticleFilterTF:
    """
    Standard Particle Filter (SIR) for Stochastic Volatility Model.
    
    Model:
        x_t = alpha * x_{t-1} + sigma * v_t
        y_t = beta * exp(x_t/2) * w_t
    
    Features:
        - Manual implementation of Sequential Importance Resampling (SIR)
        - Effective Sample Size (ESS) monitoring to detect degeneracy
    """
    def __init__(self, num_particles, alpha, sigma, beta):
        self.N = num_particles
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.sigma = tf.constant(sigma, dtype=tf.float32)
        self.beta = tf.constant(beta, dtype=tf.float32)
        
    def run_filter(self, observations):
        """
        Run the particle filter over the observation sequence.
        """
        T = len(observations)
        
        # 1. Initialization
        # Sample from stationary distribution N(0, sigma^2/(1-alpha^2))
        stat_std = self.sigma / tf.sqrt(1.0 - self.alpha**2)
        particles = tf.random.normal((self.N,), mean=0.0, stddev=stat_std)
        
        # Initial uniform weights
        log_weights = tf.zeros((self.N,), dtype=tf.float32) - tf.math.log(float(self.N))
        
        # Storage for results
        estimates = []
        ess_history = []
        
        print(f"Starting PF with {self.N} particles...")
        
        for t in range(T):
            y_curr = observations[t]
            
            # --- A. Prediction (Transition) ---
            # x_t ~ p(x_t | x_{t-1})
            noise = tf.random.normal((self.N,))
            particles = self.alpha * particles + self.sigma * noise
            
            # --- B. Update (Likelihood) ---
            # y_t | x_t ~ N(0, beta^2 * exp(x_t))
            # std_dev = beta * exp(x_t / 2)
            obs_std = self.beta * tf.exp(particles / 2.0)
            
            # Calculate Log-Likelihood of y_curr given each particle
            # log N(y; 0, std) = -log(std) - 0.5 * (y/std)^2 + const
            # We ignore constants for weights as they cancel out
            log_likelihood = -tf.math.log(obs_std + 1e-8) - 0.5 * (y_curr / obs_std)**2
            
            # Update log weights
            log_weights = log_weights + log_likelihood
            
            # Normalize weights (using Log-Sum-Exp for numerical stability)
            log_weights_norm = log_weights - tf.reduce_logsumexp(log_weights)
            weights = tf.exp(log_weights_norm)
            
            # --- C. State Estimation ---
            # E[x] = sum(w_i * x_i)
            estimate = tf.reduce_sum(weights * particles)
            estimates.append(estimate)
            
            # --- D. Degeneracy Check & Resampling ---
            # Calculate Effective Sample Size (ESS)
            # ESS = 1 / sum(w^2)
            ess = 1.0 / tf.reduce_sum(tf.square(weights))
            ess_history.append(ess)
            
            # Resample threshold (usually N/2)
            # Standard PF (SIR) often resamples at every step or when ESS < N/2
            # Here we implement resampling to fix degeneracy
            if ess < self.N / 2.0:
                # Multinomial Resampling using TensorFlow
                # tf.random.categorical requires logits (unnormalized log probs)
                # It returns indices [num_samples, num_to_sample]
                indices = tf.random.categorical(tf.reshape(log_weights, (1, -1)), self.N)
                indices = tf.reshape(indices, (-1,))
                
                # Copy particles
                particles = tf.gather(particles, indices)
                
                # Reset weights to uniform 1/N
                log_weights = tf.zeros((self.N,), dtype=tf.float32) - tf.math.log(float(self.N))
        
        return tf.stack(estimates), tf.stack(ess_history)

# ============================================================================
# Execution & Visualization
# ============================================================================

def run_pf_experiment():
    # 1. Generate Data using the Model from part (a)
    sv_model = StochasticVolatilityModelTF(alpha=0.91, sigma=1.0, beta=0.5)
    T = 100
    true_states, observations = sv_model.generate_data(T)
    
    # 2. Run Particle Filter
    # Try with N=1000 particles
    pf = StandardParticleFilterTF(num_particles=1000, alpha=0.91, sigma=1.0, beta=0.5)
    pf_estimates, ess_history = pf.run_filter(observations)
    
    # 3. Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Tracking Performance
    ax1 = axes[0]
    ax1.plot(true_states, 'k-', label='True State', linewidth=1.5)
    ax1.plot(pf_estimates, 'g--', label='PF Estimate (N=1000)', linewidth=1.5)
    ax1.set_title('Part 1(II)(c): Particle Filter Tracking Performance')
    ax1.set_ylabel('Log-Volatility')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Particle Degeneracy (ESS)
    ax2 = axes[1]
    ax2.plot(ess_history, 'm.-', label='Effective Sample Size (ESS)')
    ax2.axhline(y=500, color='r', linestyle=':', label='Resampling Threshold (N/2)')
    ax2.set_title('Particle Degeneracy Analysis: ESS over Time')
    ax2.set_ylabel('ESS')
    ax2.set_xlabel('Time Step')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Interpret text
    plt.tight_layout()
    plt.show()
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((pf_estimates.numpy() - true_states.numpy())**2))
    print(f"PF RMSE: {rmse:.4f}")
    print("-" * 50)
    print("Discussion on Degeneracy:")
    print("In the ESS plot, you will see the ESS drop as weights become concentrated")
    print("on a few particles (Degeneracy). When ESS hits the red line (500),")
    print("Resampling is triggered, restoring ESS to N (1000).")

if __name__ == "__main__":
    run_pf_experiment()