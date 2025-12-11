import tensorflow as tf
from src.models.lgssm import LinearGaussianSSM

class KalmanFilter:
    """
    Standard Kalman Filter for Linear Gaussian Systems.
    Implements Joseph stabilized covariance update.
    """
    
    def __init__(self, model: LinearGaussianSSM):
        self.model = model
        self.F = model.F
        self.H = model.H
        self.Q = model.Q
        self.R = model.R
        
    def predict(self, x, P):
        """
        Time Update:
        x_pred = F * x
        P_pred = F * P * F^T + Q
        """
        x_pred = tf.matmul(self.F, x)
        
        fp = tf.matmul(self.F, P)
        P_pred = tf.matmul(fp, self.F, transpose_b=True) + self.Q
        
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z_meas):
        """
        Measurement Update (Joseph Form):
        K = P * H^T * S^-1
        x_new = x + K(z - Hx)
        P_new = (I - KH)P(I - KH)^T + KRK^T
        """
        # 1. Innovation
        z_pred_val = tf.matmul(self.H, x_pred)
        y_res = z_meas - z_pred_val
        
        # 2. Innovation Covariance S = HPH' + R
        hp = tf.matmul(self.H, P_pred)
        S = tf.matmul(hp, self.H, transpose_b=True) + self.R
        
        # 3. Kalman Gain K
        # Solve S * K^T = H * P  => K = (S^-1 * H * P)^T
        # More stable: K^T = CholeskySolve(S, H*P)
        pht = tf.matmul(P_pred, self.H, transpose_b=True) # P * H^T
        K = tf.matmul(pht, tf.linalg.inv(S)) # Simplified inversion
        
        # 4. State Update
        x_new = x_pred + tf.matmul(K, y_res)
        
        # 5. Covariance Update (Joseph Form)
        dim_x = tf.shape(P_pred)[0]
        I = tf.eye(dim_x, dtype=tf.float32)
        I_KH = I - tf.matmul(K, self.H)
        
        p_term = tf.matmul(tf.matmul(I_KH, P_pred), I_KH, transpose_b=True)
        r_term = tf.matmul(tf.matmul(K, self.R), K, transpose_b=True)
        P_new = p_term + r_term
        
        return x_new, P_new

    def run(self, observations, x_init, P_init):
        """Batch run over observations."""
        T = tf.shape(observations)[0]
        x = tf.reshape(x_init, (-1, 1))
        P = P_init
        
        estimates = []
        covariances = []
        
        for t in range(T):
            z = tf.reshape(observations[t], (-1, 1))
            
            # Predict
            x_pred, P_pred = self.predict(x, P)
            
            # Update
            x, P = self.update(x_pred, P_pred, z)
            
            estimates.append(x)
            covariances.append(P)
            
        return tf.stack(estimates), tf.stack(covariances)