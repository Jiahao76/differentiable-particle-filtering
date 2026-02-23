import tensorflow as tf
import numpy as np
from src.models.base_model import StateSpaceModel


class LinearGaussianSSM(StateSpaceModel):
    """
    Linear Gaussian State Space Model (LGSSM).

    Dynamics:
        x_t = F * x_{t-1} + q_t,  q_t ~ N(0, Q)
    Observation:
        y_t = H * x_t + r_t,      r_t ~ N(0, R)

    Reference:
        Doucet & Johansen (2009), Example 2.
    """

    def __init__(self, F, H, Q, R):
        """
        Initialize the Linear Gaussian State Space Model.

        Args:
            F (tf.Tensor): State transition matrix of shape (dim_x, dim_x).
            H (tf.Tensor): Observation matrix of shape (dim_y, dim_x).
            Q (tf.Tensor): Process noise covariance of shape (dim_x, dim_x).
            R (tf.Tensor): Observation noise covariance of shape (dim_y, dim_y).
        """
        state_dim = F.shape[0]
        obs_dim = H.shape[0]
        super().__init__(state_dim, obs_dim)

        self.F = tf.cast(F, dtype=tf.float32)
        self.H = tf.cast(H, dtype=tf.float32)
        self.Q = tf.cast(Q, dtype=tf.float32)
        self.R = tf.cast(R, dtype=tf.float32)

        # Pre-compute Cholesky factors for sampling
        self._chol_Q = tf.linalg.cholesky(self.Q)
        self._chol_R = tf.linalg.cholesky(self.R)

    def transition(self, x_prev, noise=None):
        """
        Propagate state: x_t = F * x_{t-1} + q_t, q_t ~ N(0, Q).

        Args:
            x_prev (tf.Tensor): Previous state of shape (N, dim_x).
            noise (tf.Tensor, optional): Standard normal noise of shape (N, dim_x).

        Returns:
            tf.Tensor: Next state of shape (N, dim_x).
        """
        mean = tf.matmul(x_prev, self.F, transpose_b=True)
        if noise is None:
            noise = tf.random.normal(tf.shape(mean), dtype=tf.float32)
        return mean + tf.matmul(noise, self._chol_Q, transpose_b=True)

    def observation(self, x_curr, noise=None):
        """
        Generate observation: y_t = H * x_t + r_t, r_t ~ N(0, R).

        Args:
            x_curr (tf.Tensor): Current state of shape (N, dim_x).
            noise (tf.Tensor, optional): Standard normal noise of shape (N, dim_y).

        Returns:
            tf.Tensor: Observation of shape (N, dim_y).
        """
        mean = self.observation_mean(x_curr)
        if noise is None:
            noise = tf.random.normal(tf.shape(mean), dtype=tf.float32)
        return mean + tf.matmul(noise, self._chol_R, transpose_b=True)

    def observation_mean(self, x_curr):
        """
        Deterministic observation function h(x) = H * x.

        Args:
            x_curr (tf.Tensor): Current state of shape (N, dim_x).

        Returns:
            tf.Tensor: Predicted observation of shape (N, dim_y).
        """
        return tf.matmul(x_curr, self.H, transpose_b=True)

    def log_likelihood(self, y_true, x_particles):
        """
        Compute log p(y | x) for Gaussian observation model.

        Args:
            y_true (tf.Tensor): Observation of shape (dim_y,) or (1, dim_y).
            x_particles (tf.Tensor): Particles of shape (N, dim_x).

        Returns:
            tf.Tensor: Log-likelihood for each particle of shape (N,).
        """
        h_x = self.observation_mean(x_particles)
        residual = y_true - h_x
        R_inv = tf.linalg.inv(self.R)
        mahal = tf.reduce_sum(
            residual * tf.linalg.matvec(R_inv, residual), axis=-1
        )
        log_det = tf.linalg.logdet(self.R)
        d = tf.cast(self.obs_dim, tf.float32)
        return -0.5 * mahal - 0.5 * log_det - 0.5 * d * tf.math.log(2.0 * np.pi)

    def transition_log_pdf(self, x_curr, x_prev):
        """
        Compute log p(x_curr | x_prev) for the transition model.

        Args:
            x_curr (tf.Tensor): Current state of shape (N, dim_x).
            x_prev (tf.Tensor): Previous state of shape (N, dim_x).

        Returns:
            tf.Tensor: Log probability of shape (N,).
        """
        mean = tf.matmul(x_prev, self.F, transpose_b=True)
        residual = x_curr - mean
        Q_inv = tf.linalg.inv(self.Q)
        mahal = tf.reduce_sum(
            residual * tf.linalg.matvec(Q_inv, residual), axis=-1
        )
        log_det = tf.linalg.logdet(self.Q)
        d = tf.cast(self.state_dim, tf.float32)
        return -0.5 * mahal - 0.5 * log_det - 0.5 * d * tf.math.log(2.0 * np.pi)
