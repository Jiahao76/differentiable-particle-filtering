"""
Nonlinear State Space Model from Andrieu et al. (2010)
Section 3.1, Equations 14-15

Reference:
[Andrieu(10)] Andrieu, C., Doucet, A., & Holenstein, R.
"Particle Markov chain Monte Carlo methods."
Journal of the Royal Statistical Society: Series B, 72(3), 269-342, 2010.
"""
import tensorflow as tf
import numpy as np
from src.models.base_model import StateSpaceModel


class NonlinearSSM(StateSpaceModel):
    """
    Highly nonlinear state space model from Andrieu et al. (2010) Section 3.1.
    
    State Equation (Eq. 14):
        X_n = X_{n-1}/2 + 25*X_{n-1}/(1+X_{n-1}^2) + 8*cos(1.2*n) + V_n
        where V_n ~ N(0, sigma_V^2)
    
    Observation Equation (Eq. 15):
        Y_n = X_n^2/20 + W_n
        where W_n ~ N(0, sigma_W^2)
    
    This model is challenging because:
    - State evolution is highly nonlinear with time-varying forcing term
    - Observation is quadratic in state (non-Gaussian even with Gaussian state)
    - Posterior is typically multimodal
    """
    
    def __init__(self, sigma_V: float = np.sqrt(10.0), sigma_W: float = 1.0):
        """
        Initialize the Nonlinear SSM.
        
        Args:
            sigma_V: Standard deviation of state noise (default: sqrt(10) as in Andrieu(10))
            sigma_W: Standard deviation of observation noise (default: 1.0)
        """
        super().__init__(state_dim=1, obs_dim=1)
        
        self.sigma_V = tf.constant(sigma_V, dtype=tf.float32)
        self.sigma_W = tf.constant(sigma_W, dtype=tf.float32)
        self.sigma_V_sq = self.sigma_V ** 2
        self.sigma_W_sq = self.sigma_W ** 2
        
        # For time-dependent forcing term
        self.time_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    
    def get_params(self) -> tf.Tensor:
        """
        Get model parameters as a tensor.
        
        Returns:
            tf.Tensor: Parameter vector [sigma_V, sigma_W]
        """
        return tf.stack([self.sigma_V, self.sigma_W])
    
    def reset_time(self):
        """Reset time step counter."""
        self.time_step.assign(0)
    
    def transition(self, x_prev: tf.Tensor, noise: tf.Tensor = None, time_step: int = None) -> tf.Tensor:
        """
        State transition: X_n = f(X_{n-1}, n) + V_n
        
        Args:
            x_prev: Previous state [num_particles, 1]
            noise: Optional noise ~ N(0, 1) for sampling
            time_step: Time step n (if None, uses internal counter)
        
        Returns:
            Current state X_n
        """
        if time_step is None:
            time_step = self.time_step.numpy()
            self.time_step.assign_add(1)
        
        # Deterministic part: f(x, t)
        # f(x) = x/2 + 25*x/(1+x^2) + 8*cos(1.2*n)
        x_sq = x_prev ** 2
        f_x = x_prev / 2.0 + 25.0 * x_prev / (1.0 + x_sq) + 8.0 * tf.cos(1.2 * float(time_step))
        
        # Add noise
        if noise is None:
            noise = tf.random.normal(tf.shape(x_prev), dtype=tf.float32)
        
        return f_x + self.sigma_V * noise
    
    def observation(self, x_curr: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        """
        Observation: Y_n = h(X_n) + W_n.

        Args:
            x_curr (tf.Tensor): Current state of shape (N, 1).
            noise (tf.Tensor, optional): Standard normal noise of shape (N, 1).

        Returns:
            tf.Tensor: Observation Y_n of shape (N, 1).
        """
        h_x = self.observation_mean(x_curr)
        if noise is None:
            noise = tf.random.normal(tf.shape(x_curr), dtype=tf.float32)
        return h_x + self.sigma_W * noise

    def observation_mean(self, x_curr: tf.Tensor) -> tf.Tensor:
        """
        Deterministic observation function h(x) = x^2 / 20.

        Args:
            x_curr (tf.Tensor): Current state of shape (..., 1).

        Returns:
            tf.Tensor: Predicted observation of shape (..., 1).
        """
        return (x_curr ** 2) / 20.0
    
    def log_likelihood(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Compute log p(y | x).
        
        Args:
            y: Observation [obs_dim] or [1]
            x: States [num_particles, state_dim]
        
        Returns:
            Log likelihood [num_particles]
        """
        # h(x) = x^2 / 20
        h_x = (x ** 2) / 20.0
        
        # Gaussian likelihood: N(y | h(x), sigma_W^2)
        residual = y - h_x
        log_lik = -0.5 * (residual ** 2) / self.sigma_W_sq - 0.5 * tf.math.log(2.0 * np.pi * self.sigma_W_sq)
        
        return tf.reduce_sum(log_lik, axis=-1)
    
    def transition_log_pdf(self, x_curr: tf.Tensor, x_prev: tf.Tensor, time_step: int = None) -> tf.Tensor:
        """
        Compute log p(x_curr | x_prev).
        
        Args:
            x_curr: Current state [num_particles, state_dim]
            x_prev: Previous state [num_particles, state_dim]
            time_step: Time step n (required for time-varying dynamics)
        
        Returns:
            Log probability [num_particles]
        """
        if time_step is None:
            raise ValueError("time_step is required for NonlinearSSM.transition_log_pdf")
        
        # Compute deterministic mean
        x_sq = x_prev ** 2
        mean = x_prev / 2.0 + 25.0 * x_prev / (1.0 + x_sq) + 8.0 * tf.cos(1.2 * float(time_step))
        
        # Gaussian distribution: N(x_curr | mean, sigma_V^2)
        residual = x_curr - mean
        log_prob = -0.5 * (residual ** 2) / self.sigma_V_sq - 0.5 * tf.math.log(2.0 * np.pi * self.sigma_V_sq)
        
        return tf.reduce_sum(log_prob, axis=-1)
    
    def sample_trajectory(self, T: int, x0: float = 0.0, seed: int = None) -> tuple:
        """
        Generate a complete trajectory.
        
        Args:
            T: Number of time steps
            x0: Initial state
            seed: Random seed
        
        Returns:
            (states, observations) both of shape [T, 1]
        """
        if seed is not None:
            tf.random.set_seed(seed)
        
        self.reset_time()
        
        states = []
        observations = []
        
        x = tf.constant([[x0]], dtype=tf.float32)
        
        for t in range(T):
            x = self.transition(x, time_step=t)
            y = self.observation(x)
            
            states.append(x[0, 0])
            observations.append(y[0, 0])
        
        return tf.stack(states), tf.stack(observations)
