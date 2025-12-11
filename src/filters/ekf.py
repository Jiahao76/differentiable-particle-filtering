import tensorflow as tf
from src.models.base_model import StateSpaceModel

class ExtendedKalmanFilter:
    """
    Optimized Extended Kalman Filter using TensorFlow Graph Compilation.
    
    PERFORMANCE FIX:
    Added @tf.function decorators. This compiles the AutoDiff gradients (Jacobians)
    into a static graph, removing the Python interpreter overhead. 
    This approximates the speed of analytical Jacobians while keeping the class generic.
    """
    
    def __init__(self, model: StateSpaceModel):
        self.model = model
        
    def get_jacobian(self, func, state):
        """Compute Jacobian using AutoDiff (Compiled)."""
        with tf.GradientTape() as tape:
            tape.watch(state)
            output = func(state)
            if len(output.shape) == 0:
                output = tf.reshape(output, [1])
        return tape.jacobian(output, state)

    @tf.function(reduce_retracing=True)
    def predict(self, x, P, Q):
        x_flat = tf.reshape(x, (-1,)) 
        
        # Prediction step
        x_pred = self.model.transition(x_flat, noise=tf.zeros_like(x_flat))
        x_pred = tf.reshape(x_pred, (-1, 1))
        
        # Calculate Jacobian F (Compiled)
        def trans_func(s): return self.model.transition(s, noise=tf.zeros_like(s))
        F = self.get_jacobian(trans_func, x_flat)
        
        F = tf.squeeze(F) 
        if len(F.shape) < 2: F = tf.reshape(F, (1, 1))
        
        P_pred = tf.matmul(tf.matmul(F, P), F, transpose_b=True) + Q
        return x_pred, P_pred

    @tf.function(reduce_retracing=True)
    def update(self, x_pred, P_pred, z_meas, R):
        x_flat = tf.reshape(x_pred, (-1,))
        
        # 1. Measurement Prediction
        z_pred = self.model.observation(x_flat, noise=tf.zeros_like(x_flat))
        z_pred = tf.reshape(z_pred, (-1, 1))
        z_meas = tf.reshape(z_meas, (-1, 1))
        
        y_res = z_meas - z_pred
        
        # 2. Jacobian H (Compiled)
        # Using noise=1.0 proxy for structural sensitivity as per SV model logic
        def obs_func(s): return self.model.observation(s, noise=tf.ones_like(s))
        H = self.get_jacobian(obs_func, x_flat)
        H = tf.squeeze(H)
        if len(H.shape) < 2: H = tf.reshape(H, (1, 1))
        
        # 3. Innovation Covariance S
        S = tf.matmul(tf.matmul(H, P_pred), H, transpose_b=True) + R
        
        # 4. Kalman Gain K
        epsilon = 1e-6
        S_inv = tf.linalg.inv(S + tf.eye(tf.shape(S)[0]) * epsilon)
        
        pht = tf.matmul(P_pred, H, transpose_b=True)
        K = tf.matmul(pht, S_inv)
        
        # 5. State Update
        x_new = x_pred + tf.matmul(K, y_res)
        
        # 6. Covariance Update
        dim_x = tf.shape(P_pred)[0]
        I = tf.eye(dim_x, dtype=tf.float32)
        tmp = I - tf.matmul(K, H)
        P_new = tf.matmul(tmp, P_pred)
        
        return x_new, P_new
        
    def run(self, observations, x_init, P_init, Q_matrix, R_matrix):
        """
        Runs the EKF loop.
        Note: The loop body is compiled via the methods above.
        For maximum speed, we wrap the loop itself in tf.function.
        """
        # Convert inputs to tensors once
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        x_init = tf.convert_to_tensor(x_init, dtype=tf.float32)
        P_init = tf.convert_to_tensor(P_init, dtype=tf.float32)
        Q_matrix = tf.convert_to_tensor(Q_matrix, dtype=tf.float32)
        R_matrix = tf.convert_to_tensor(R_matrix, dtype=tf.float32)

        return self._run_compiled(observations, x_init, P_init, Q_matrix, R_matrix)

    @tf.function(reduce_retracing=True)
    def _run_compiled(self, observations, x_init, P_init, Q_matrix, R_matrix):
        T = tf.shape(observations)[0]
        x = tf.reshape(x_init, (-1, 1))
        P = P_init
        
        # TensorArray is required for loop output in tf.function
        estimates = tf.TensorArray(dtype=tf.float32, size=T)
        
        for t in tf.range(T):
            z = observations[t]
            x_pred, P_pred = self.predict(x, P, Q_matrix)
            x, P = self.update(x_pred, P_pred, z, R_matrix)
            estimates = estimates.write(t, x)
            
        return estimates.stack()