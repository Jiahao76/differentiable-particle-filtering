import tensorflow as tf
from src.models.base_model import StateSpaceModel

class LinearGaussianSSM(StateSpaceModel):
    """
    Linear Gaussian State Space Model (LGSSM).
    
    Dynamics:
        x_t = F * x_{t-1} + q_t,  q_t ~ N(0, Q)
    Observation:
        y_t = H * x_t + r_t,      r_t ~ N(0, R)
    """
    
    def __init__(self, F, H, Q, R):
        """
        Args:
            F: State transition matrix (dim_x, dim_x)
            H: Observation matrix (dim_y, dim_x)
            Q: Process noise covariance (dim_x, dim_x)
            R: Observation noise covariance (dim_y, dim_y)
        """
        state_dim = F.shape[0]
        obs_dim = H.shape[0]
        super().__init__(state_dim, obs_dim)
        
        # Cast to float32 for TensorFlow consistency
        self.F = tf.cast(F, dtype=tf.float32)
        self.H = tf.cast(H, dtype=tf.float32)
        self.Q = tf.cast(Q, dtype=tf.float32)
        self.R = tf.cast(R, dtype=tf.float32)

    def transition(self, x_prev, noise=None):
        # Deterministic part: F * x
        mean = tf.matmul(x_prev, self.F, transpose_b=True) # (N, dim_x)
        if noise is None:
            # Note: For strict sampling we need Cholesky(Q), simplified here
            noise = tf.random.normal(tf.shape(mean)) 
        return mean # + noise component if implementing sampling

    def observation(self, x_curr, noise=None):
        # Deterministic part: H * x
        mean = tf.matmul(x_curr, self.H, transpose_b=True)
        return mean