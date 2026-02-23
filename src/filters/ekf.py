import tensorflow as tf
from src.models.base_model import StateSpaceModel


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter using TensorFlow AutoDiff for Jacobians.

    Uses observation_mean() from the model for the deterministic observation
    function h(x) and computes the Jacobian H = dh/dx via GradientTape.

    Reference:
        Bar-Shalom, Li, and Kirubarajan (2001), "Estimation with Applications
        to Tracking and Navigation", Chapter 5.
    """

    def __init__(self, model: StateSpaceModel):
        """
        Initialize the Extended Kalman Filter.

        Args:
            model (StateSpaceModel): State-space model instance.
        """
        self.model = model

    def get_jacobian(self, func, state):
        """
        Compute Jacobian of func w.r.t. state using AutoDiff.

        Args:
            func (callable): Function mapping state to output.
            state (tf.Tensor): Input state tensor.

        Returns:
            tf.Tensor: Jacobian matrix.
        """
        with tf.GradientTape() as tape:
            tape.watch(state)
            output = func(state)
            if len(output.shape) == 0:
                output = tf.reshape(output, [1])
        return tape.jacobian(output, state)

    @tf.function(reduce_retracing=True)
    def predict(self, x, P, Q):
        """
        EKF predict step.

        Args:
            x (tf.Tensor): State estimate of shape (dim_x, 1).
            P (tf.Tensor): Covariance of shape (dim_x, dim_x).
            Q (tf.Tensor): Process noise covariance.

        Returns:
            Tuple of (x_pred, P_pred).
        """
        x_flat = tf.reshape(x, (-1,))

        x_pred = self.model.transition(x_flat, noise=tf.zeros_like(x_flat))
        x_pred = tf.reshape(x_pred, (-1, 1))

        def trans_func(s):
            return self.model.transition(s, noise=tf.zeros_like(s))

        F = self.get_jacobian(trans_func, x_flat)
        F = tf.squeeze(F)
        if len(F.shape) < 2:
            F = tf.reshape(F, (1, 1))

        P_pred = tf.matmul(tf.matmul(F, P), F, transpose_b=True) + Q
        return x_pred, P_pred

    @tf.function(reduce_retracing=True)
    def update(self, x_pred, P_pred, z_meas, R):
        """
        EKF measurement update step using Joseph stabilized form.

        Args:
            x_pred (tf.Tensor): Predicted state of shape (dim_x, 1).
            P_pred (tf.Tensor): Predicted covariance of shape (dim_x, dim_x).
            z_meas (tf.Tensor): Measurement of shape (dim_y,).
            R (tf.Tensor): Measurement noise covariance.

        Returns:
            Tuple of (x_new, P_new).
        """
        x_flat = tf.reshape(x_pred, (-1,))

        # Use observation_mean for the deterministic h(x)
        z_pred = self.model.observation_mean(tf.expand_dims(x_flat, 0))
        z_pred = tf.reshape(z_pred, (-1, 1))
        z_meas = tf.reshape(z_meas, (-1, 1))

        y_res = z_meas - z_pred

        # Jacobian of h(x) using observation_mean
        def obs_func(s):
            return self.model.observation_mean(tf.expand_dims(s, 0))[0]

        H = self.get_jacobian(obs_func, x_flat)
        H = tf.squeeze(H)
        if len(H.shape) < 2:
            H = tf.reshape(H, (1, 1))

        # Innovation covariance S = H P H^T + R
        S = tf.matmul(tf.matmul(H, P_pred), H, transpose_b=True) + R

        # Kalman gain K = P H^T S^{-1}
        epsilon = 1e-6
        S_inv = tf.linalg.inv(S + tf.eye(tf.shape(S)[0]) * epsilon)
        pht = tf.matmul(P_pred, H, transpose_b=True)
        K = tf.matmul(pht, S_inv)

        # State update
        x_new = x_pred + tf.matmul(K, y_res)

        # Joseph stabilized covariance update: P = (I-KH)P(I-KH)^T + KRK^T
        dim_x = tf.shape(P_pred)[0]
        I = tf.eye(dim_x, dtype=tf.float32)
        I_KH = I - tf.matmul(K, H)
        P_new = (
            tf.matmul(tf.matmul(I_KH, P_pred), I_KH, transpose_b=True)
            + tf.matmul(tf.matmul(K, R), K, transpose_b=True)
        )

        return x_new, P_new

    def run(self, observations, x_init, P_init, Q_matrix, R_matrix):
        """
        Run the EKF over a sequence of observations.

        Args:
            observations (tf.Tensor): Observations of shape (T, dim_y).
            x_init (tf.Tensor): Initial state of shape (dim_x,).
            P_init (tf.Tensor): Initial covariance of shape (dim_x, dim_x).
            Q_matrix (tf.Tensor): Process noise covariance.
            R_matrix (tf.Tensor): Measurement noise covariance.

        Returns:
            tf.Tensor: State estimates of shape (T, dim_x, 1).
        """
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        x_init = tf.convert_to_tensor(x_init, dtype=tf.float32)
        P_init = tf.convert_to_tensor(P_init, dtype=tf.float32)
        Q_matrix = tf.convert_to_tensor(Q_matrix, dtype=tf.float32)
        R_matrix = tf.convert_to_tensor(R_matrix, dtype=tf.float32)

        return self._run_compiled(
            observations, x_init, P_init, Q_matrix, R_matrix
        )

    @tf.function(reduce_retracing=True)
    def _run_compiled(self, observations, x_init, P_init, Q_matrix, R_matrix):
        """Compiled inner loop for EKF."""
        T = tf.shape(observations)[0]
        x = tf.reshape(x_init, (-1, 1))
        P = P_init

        estimates = tf.TensorArray(dtype=tf.float32, size=T)

        for t in tf.range(T):
            z = observations[t]
            x_pred, P_pred = self.predict(x, P, Q_matrix)
            x, P = self.update(x_pred, P_pred, z, R_matrix)
            estimates = estimates.write(t, x)

        return estimates.stack()
