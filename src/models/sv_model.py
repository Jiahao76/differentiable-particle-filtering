import tensorflow as tf
import tensorflow_probability as tfp
from src.models.base_model import StateSpaceModel
class StochasticVolatilityModel(StateSpaceModel):
    """
    Stochastic Volatility Model implementation (Doucet et al., 2009, Example 4).
    
    This class defines the dynamics and observation model for a stochastic
    volatility process, commonly used in financial econometrics.
    
    State Space:
        X_t = alpha * X_{t-1} + sigma * V_t,    V_t ~ N(0, 1)
        
    Observation Space:
        Y_t = beta * exp(X_t / 2) * W_t,        W_t ~ N(0, 1)
        
    Attributes:
        alpha (tf.Tensor): Auto-regressive coefficient (persistence).
        sigma (tf.Tensor): Standard deviation of the state process.
        beta (tf.Tensor): Scaling factor for the observation variance.
    """
    
    def __init__(self, alpha: float = 0.91, sigma: float = 1.0, beta: float = 0.5):
        """
        Initialize the Stochastic Volatility Model parameters.

        Args:
            alpha (float): AR(1) coefficient. Default is 0.91 (Doucet 09).
            sigma (float): State noise std dev. Default is 1.0.
            beta (float): Observation scaling. Default is 0.5.
        """
        # Initialize base class (state_dim=1, obs_dim=1)
        super().__init__(state_dim=1, obs_dim=1)
        
        # Cast parameters to float32 for TensorFlow compatibility
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.sigma = tf.constant(sigma, dtype=tf.float32)
        self.beta = tf.constant(beta, dtype=tf.float32)

    def transition(self, x_prev: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        """
        Propagate the state from time t-1 to t.
        
        Equation: X_t = alpha * X_{t-1} + sigma * V_t

        Args:
            x_prev (tf.Tensor): State at t-1. Shape: [num_particles, state_dim]
            noise (tf.Tensor, optional): Standard normal noise. 
                                         If None, generated internally.

        Returns:
            tf.Tensor: Propagated state X_t.
        """
        # Generate noise if not provided (useful for simple simulations)
        if noise is None:
            noise = tf.random.normal(tf.shape(x_prev))
            
        return self.alpha * x_prev + self.sigma * noise

    def observation(self, x_curr: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        """
        Generate an observation from the current state (Generative Mode).
        
        Equation: Y_t = beta * exp(X_t / 2) * W_t

        Args:
            x_curr (tf.Tensor): Current state X_t.
            noise (tf.Tensor, optional): Standard normal noise.

        Returns:
            tf.Tensor: Observation Y_t.
        """
        if noise is None:
            noise = tf.random.normal(tf.shape(x_curr))
            
        return self.beta * tf.exp(x_curr / 2.0) * noise
    
    def log_likelihood(self, y_true: tf.Tensor, x_particles: tf.Tensor) -> tf.Tensor:
        """
        Compute the log-likelihood of the observation given the state particles.
        
        This method is used by the Particle Filter to update weights.
        
        Model:
            y_t | x_t ~ N(0, beta^2 * exp(x_t))
            std_dev_t = beta * exp(x_t / 2)

        Args:
            y_true (tf.Tensor): The actual observation at time t. Shape: [obs_dim]
            x_particles (tf.Tensor): The particles representing the state. 
                                     Shape: [num_particles, state_dim]

        Returns:
            tf.Tensor: Log-likelihood for each particle. Shape: [num_particles]
        """
        # Calculate the standard deviation dependent on the state
        # std_dev = beta * exp(x / 2)
        obs_std = self.beta * tf.exp(x_particles / 2.0)
        
        # Add a small epsilon to prevent division by zero or log(0)
        epsilon = 1e-8
        safe_std = obs_std + epsilon
        
        # Calculate log p(y | x) for a Normal distribution centered at 0
        # Formula: -log(sigma) - 0.5 * (y / sigma)^2
        # Note: Constant terms like -0.5*log(2*pi) are often omitted in PF 
        # as weights are normalized later, but we include them for correctness.
        log_prob = -tf.math.log(safe_std) - 0.5 * ((y_true) / safe_std)**2
        
        # If observation is multidimensional, sum log-probs across dimensions
        if self.obs_dim > 1:
            log_prob = tf.reduce_sum(log_prob, axis=-1)
            
        return log_prob
    
    def transition_log_pdf(self, x_curr: tf.Tensor, x_prev: tf.Tensor) -> tf.Tensor:
        """
        Compute log p(x_curr | x_prev) for the state transition model.
        
        For SV model: X_t = alpha * X_{t-1} + sigma * V_t, V_t ~ N(0, 1)
        So: X_t | X_{t-1} ~ N(alpha * X_{t-1}, sigma^2)
        
        Args:
            x_curr (tf.Tensor): Current state X_t. Shape: [num_particles, state_dim]
            x_prev (tf.Tensor): Previous state X_{t-1}. Shape: [num_particles, state_dim]
        
        Returns:
            tf.Tensor: Log probability p(x_curr | x_prev). Shape: [num_particles]
        """
        # Mean of transition: mu_t = alpha * X_{t-1}
        mean = self.alpha * x_prev
        
        # Variance: sigma^2
        var = self.sigma ** 2
        std = self.sigma
        
        # Log-likelihood of Normal distribution
        # log p(x | mean, std) = -log(std) - 0.5 * ((x - mean) / std)^2
        log_prob = -tf.math.log(std) - 0.5 * ((x_curr - mean) / std) ** 2
        
        # Sum across dimensions if multidimensional
        if self.state_dim > 1:
            log_prob = tf.reduce_sum(log_prob, axis=-1)
        else:
            log_prob = tf.reshape(log_prob, [-1])
            
        return log_prob