import tensorflow as tf
import numpy as np
from src.models.base_model import StateSpaceModel


class StochasticVolatilityModel(StateSpaceModel):
    """
    Stochastic Volatility Model (Doucet et al., 2009, Example 4).

    State Space:
        X_t = alpha * X_{t-1} + sigma * V_t,    V_t ~ N(0, 1)

    Observation Space:
        Y_t = beta * exp(X_t / 2) * W_t,        W_t ~ N(0, 1)

    Equivalently the observation model is:
        Y_t | X_t ~ N(0, beta^2 * exp(X_t))

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
        super().__init__(state_dim=1, obs_dim=1)

        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.sigma = tf.constant(sigma, dtype=tf.float32)
        self.beta = tf.constant(beta, dtype=tf.float32)

    def get_params(self) -> tf.Tensor:
        """
        Get model parameters as a tensor.

        Returns:
            tf.Tensor: Parameter vector [alpha, sigma, beta].
        """
        return tf.stack([self.alpha, self.sigma, self.beta])

    def transition(self, x_prev: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        """
        Propagate the state: X_t = alpha * X_{t-1} + sigma * V_t.

        Args:
            x_prev (tf.Tensor): State at t-1 of shape (N, 1).
            noise (tf.Tensor, optional): Standard normal noise of shape (N, 1).

        Returns:
            tf.Tensor: Propagated state X_t of shape (N, 1).
        """
        if noise is None:
            noise = tf.random.normal(tf.shape(x_prev))
        return self.alpha * x_prev + self.sigma * noise

    def observation(self, x_curr: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        """
        Generate observation: Y_t = beta * exp(X_t / 2) * W_t.

        Args:
            x_curr (tf.Tensor): Current state X_t of shape (N, 1).
            noise (tf.Tensor, optional): Standard normal noise of shape (N, 1).

        Returns:
            tf.Tensor: Observation Y_t of shape (N, 1).
        """
        if noise is None:
            noise = tf.random.normal(tf.shape(x_curr))
        return self.observation_mean(x_curr) * noise

    def observation_mean(self, x_curr: tf.Tensor) -> tf.Tensor:
        """
        Deterministic observation scale: h(x) = beta * exp(x / 2).

        For the SV model the observation is Y = h(x) * W with W ~ N(0,1),
        so h(x) is the state-dependent standard deviation.

        Args:
            x_curr (tf.Tensor): Current state of shape (..., 1).

        Returns:
            tf.Tensor: Observation scale of shape (..., 1).
        """
        return self.beta * tf.exp(x_curr / 2.0)

    def log_likelihood(self, y_true: tf.Tensor, x_particles: tf.Tensor) -> tf.Tensor:
        """
        Compute log p(y | x) for the SV observation model.

        Model: Y_t | X_t ~ N(0, beta^2 * exp(X_t))

        Args:
            y_true (tf.Tensor): Observation at time t of shape (obs_dim,).
            x_particles (tf.Tensor): Particles of shape (N, state_dim).

        Returns:
            tf.Tensor: Log-likelihood for each particle of shape (N,).
        """
        obs_std = self.observation_mean(x_particles)
        safe_std = obs_std + 1e-8

        # Full log-normal density including the -0.5*log(2*pi) constant
        log_prob = (
            -tf.math.log(safe_std)
            - 0.5 * tf.square(y_true / safe_std)
            - 0.5 * tf.math.log(2.0 * np.pi)
        )

        if self.obs_dim > 1:
            log_prob = tf.reduce_sum(log_prob, axis=-1)

        return tf.reshape(log_prob, [-1])

    def transition_log_pdf(self, x_curr: tf.Tensor, x_prev: tf.Tensor) -> tf.Tensor:
        """
        Compute log p(x_curr | x_prev) for the transition.

        X_t | X_{t-1} ~ N(alpha * X_{t-1}, sigma^2)

        Args:
            x_curr (tf.Tensor): Current state of shape (N, 1).
            x_prev (tf.Tensor): Previous state of shape (N, 1).

        Returns:
            tf.Tensor: Log probability of shape (N,).
        """
        mean = self.alpha * x_prev
        log_prob = (
            -tf.math.log(self.sigma)
            - 0.5 * tf.square((x_curr - mean) / self.sigma)
            - 0.5 * tf.math.log(2.0 * np.pi)
        )

        if self.state_dim > 1:
            log_prob = tf.reduce_sum(log_prob, axis=-1)
        else:
            log_prob = tf.reshape(log_prob, [-1])

        return log_prob
