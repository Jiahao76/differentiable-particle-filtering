"""Unscented Kalman Filter (UKF) with TensorFlow graph compilation.

Reference:
    Julier, S. J., & Uhlmann, J. K. (2004). "Unscented Filtering and
    Nonlinear Estimation." Proceedings of the IEEE, 92(3), 401-422.
"""
import tensorflow as tf
from src.models.base_model import StateSpaceModel


class UnscentedKalmanFilter:
    """Unscented Kalman Filter using sigma-point propagation.

    Uses ``observation_mean()`` from the model for the deterministic
    observation function *h(x)*.
    """

    def __init__(self, model: StateSpaceModel, alpha=1e-3, beta=2.0, kappa=0.0):
        """Initialize the UKF.

        Args:
            model: State-space model with ``transition`` and ``observation_mean``.
            alpha: Scaling parameter controlling sigma-point spread.
            beta: Prior knowledge parameter (2 is optimal for Gaussian).
            kappa: Secondary scaling parameter (typically 0).
        """
        self.model = model
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim

        n = float(self.state_dim)
        self.lam = alpha**2 * (n + kappa) - n
        w_m_0 = self.lam / (n + self.lam)
        w_c_0 = self.lam / (n + self.lam) + (1 - alpha**2 + beta)
        w_i = 0.5 / (n + self.lam)

        self.Wm = tf.concat([[w_m_0], tf.fill([2 * int(n)], w_i)], axis=0)
        self.Wc = tf.concat([[w_c_0], tf.fill([2 * int(n)], w_i)], axis=0)

    def generate_sigma_points(self, x, P):
        """Generate sigma points from state mean and covariance.

        Args:
            x: State mean of shape ``(dim_x,)`` or ``(dim_x, 1)``.
            P: Covariance of shape ``(dim_x, dim_x)``.

        Returns:
            Sigma points of shape ``(2*dim_x + 1, dim_x)``.
        """
        n = float(self.state_dim)
        x = tf.reshape(x, (-1,))
        epsilon = 1e-6
        L = tf.linalg.cholesky(P + tf.eye(int(n)) * epsilon)
        scale = tf.sqrt(n + self.lam)
        scaled_L = scale * L
        scaled_L = tf.transpose(scaled_L)

        sp = [x]
        for i in range(int(n)):
            sp.append(x + scaled_L[i])
        for i in range(int(n)):
            sp.append(x - scaled_L[i])
        return tf.stack(sp)

    @tf.function(reduce_retracing=True)
    def predict(self, x, P, Q):
        """UKF predict step.

        Args:
            x: State estimate of shape ``(dim_x, 1)``.
            P: Covariance of shape ``(dim_x, dim_x)``.
            Q: Process noise covariance.

        Returns:
            Tuple ``(x_pred, P_pred, sigmas_pred)``.
        """
        sigmas = self.generate_sigma_points(x, P)
        sigmas_pred = self.model.transition(sigmas, noise=tf.zeros_like(sigmas))

        Wm_col = tf.reshape(self.Wm, (-1, 1))
        x_pred = tf.reduce_sum(Wm_col * sigmas_pred, axis=0)
        x_pred = tf.reshape(x_pred, (-1, 1))

        x_pred_flat = tf.reshape(x_pred, (-1,))
        residuals = sigmas_pred - x_pred_flat

        P_pred = tf.zeros_like(P)
        for i in range(sigmas_pred.shape[0]):
            diff = tf.reshape(residuals[i], (-1, 1))
            P_pred += self.Wc[i] * tf.matmul(diff, diff, transpose_b=True)
        P_pred += Q

        return x_pred, P_pred, sigmas_pred

    @tf.function(reduce_retracing=True)
    def update(self, x_pred, P_pred, z_meas, R, sigmas_pred=None):
        """UKF measurement update step.

        Args:
            x_pred: Predicted state of shape ``(dim_x, 1)``.
            P_pred: Predicted covariance of shape ``(dim_x, dim_x)``.
            z_meas: Measurement of shape ``(dim_y,)``.
            R: Measurement noise covariance.
            sigmas_pred: Pre-propagated sigma points (optional).

        Returns:
            Tuple ``(x_new, P_new)``.
        """
        if sigmas_pred is None:
            sigmas_pred = self.generate_sigma_points(x_pred, P_pred)

        sigmas_obs = self.model.observation_mean(sigmas_pred)

        Wm_col = tf.reshape(self.Wm, (-1, 1))
        z_pred = tf.reduce_sum(Wm_col * sigmas_obs, axis=0)
        z_pred = tf.reshape(z_pred, (-1, 1))

        x_pred_flat = tf.reshape(x_pred, (-1,))
        z_pred_flat = tf.reshape(z_pred, (-1,))

        res_x = sigmas_pred - x_pred_flat
        res_z = sigmas_obs - z_pred_flat

        S = tf.zeros((self.obs_dim, self.obs_dim))
        Pxz = tf.zeros((self.state_dim, self.obs_dim))

        for i in range(sigmas_pred.shape[0]):
            dz = tf.reshape(res_z[i], (-1, 1))
            dx = tf.reshape(res_x[i], (-1, 1))
            weight = self.Wc[i]
            S += weight * tf.matmul(dz, dz, transpose_b=True)
            Pxz += weight * tf.matmul(dx, dz, transpose_b=True)
        S += R

        epsilon = 1e-6
        S_inv = tf.linalg.inv(S + tf.eye(self.obs_dim) * epsilon)
        K = tf.matmul(Pxz, S_inv)

        z_meas = tf.reshape(z_meas, (-1, 1))
        y_residual = z_meas - z_pred
        x_new = x_pred + tf.matmul(K, y_residual)
        P_new = P_pred - tf.matmul(tf.matmul(K, S), K, transpose_b=True)
        return x_new, P_new

    def run(self, observations, x_init, P_init, Q, R):
        """Run the UKF over a sequence of observations.

        Args:
            observations: Tensor of shape ``(T, dim_y)``.
            x_init: Initial state of shape ``(dim_x,)``.
            P_init: Initial covariance of shape ``(dim_x, dim_x)``.
            Q: Process noise covariance.
            R: Measurement noise covariance.

        Returns:
            State estimates of shape ``(T, dim_x, 1)``.
        """
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        x_init = tf.convert_to_tensor(x_init, dtype=tf.float32)
        P_init = tf.convert_to_tensor(P_init, dtype=tf.float32)
        Q = tf.convert_to_tensor(Q, dtype=tf.float32)
        R = tf.convert_to_tensor(R, dtype=tf.float32)
        return self._run_compiled(observations, x_init, P_init, Q, R)

    @tf.function(reduce_retracing=True)
    def _run_compiled(self, observations, x_init, P_init, Q, R):
        """Compiled inner loop for UKF."""
        T = tf.shape(observations)[0]
        x = tf.reshape(x_init, (-1, 1))
        P = P_init
        estimates = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            z = observations[t]
            x_pred, P_pred, sigmas_pred = self.predict(x, P, Q)
            x, P = self.update(x_pred, P_pred, z, R, sigmas_pred)
            estimates = estimates.write(t, x)

        return estimates.stack()
