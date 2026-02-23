"""Standard Kalman Filter for Linear Gaussian State-Space Models.

Reference:
    Bar-Shalom, Li, and Kirubarajan (2001), "Estimation with Applications
    to Tracking and Navigation", Chapter 5.
"""
import tensorflow as tf
from src.models.lgssm import LinearGaussianSSM


class KalmanFilter:
    """Standard Kalman Filter with Joseph stabilized covariance update.

    Assumes a :class:`LinearGaussianSSM` model with matrices *F*, *H*, *Q*, *R*.
    """

    def __init__(self, model: LinearGaussianSSM):
        """Initialize the Kalman Filter.

        Args:
            model: A :class:`LinearGaussianSSM` instance.
        """
        self.model = model
        self.F = model.F
        self.H = model.H
        self.Q = model.Q
        self.R = model.R

    def predict(self, x, P):
        """Predict step (time update).

        Args:
            x: State estimate of shape ``(dim_x, 1)``.
            P: Covariance of shape ``(dim_x, dim_x)``.

        Returns:
            Tuple ``(x_pred, P_pred)``.
        """
        x_pred = tf.matmul(self.F, x)

        fp = tf.matmul(self.F, P)
        P_pred = tf.matmul(fp, self.F, transpose_b=True) + self.Q

        return x_pred, P_pred

    def update(self, x_pred, P_pred, z_meas):
        """Update step (measurement update) using Joseph form.

        Args:
            x_pred: Predicted state of shape ``(dim_x, 1)``.
            P_pred: Predicted covariance of shape ``(dim_x, dim_x)``.
            z_meas: Measurement of shape ``(dim_y, 1)``.

        Returns:
            Tuple ``(x_new, P_new)``.
        """
        z_pred_val = tf.matmul(self.H, x_pred)
        y_res = z_meas - z_pred_val

        hp = tf.matmul(self.H, P_pred)
        S = tf.matmul(hp, self.H, transpose_b=True) + self.R

        pht = tf.matmul(P_pred, self.H, transpose_b=True)
        K = tf.matmul(pht, tf.linalg.inv(S))

        x_new = x_pred + tf.matmul(K, y_res)

        dim_x = tf.shape(P_pred)[0]
        I = tf.eye(dim_x, dtype=tf.float32)
        I_KH = I - tf.matmul(K, self.H)

        p_term = tf.matmul(tf.matmul(I_KH, P_pred), I_KH, transpose_b=True)
        r_term = tf.matmul(tf.matmul(K, self.R), K, transpose_b=True)
        P_new = p_term + r_term

        return x_new, P_new

    def run(self, observations, x_init, P_init):
        """Run the Kalman Filter over a sequence of observations.

        Args:
            observations: Tensor of shape ``(T, dim_y)``.
            x_init: Initial state of shape ``(dim_x,)``.
            P_init: Initial covariance of shape ``(dim_x, dim_x)``.

        Returns:
            Tuple ``(estimates, covariances)`` where *estimates* has shape
            ``(T, dim_x, 1)`` and *covariances* has shape ``(T, dim_x, dim_x)``.
        """
        T = tf.shape(observations)[0]
        x = tf.reshape(x_init, (-1, 1))
        P = P_init

        estimates = []
        covariances = []

        for t in range(T):
            z = tf.reshape(observations[t], (-1, 1))

            x_pred, P_pred = self.predict(x, P)
            x, P = self.update(x_pred, P_pred, z)

            estimates.append(x)
            covariances.append(P)

        return tf.stack(estimates), tf.stack(covariances)
