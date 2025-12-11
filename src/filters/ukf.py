import tensorflow as tf
from src.models.base_model import StateSpaceModel

class UnscentedKalmanFilter:
    """
    Optimized Unscented Kalman Filter using TensorFlow Graph Compilation.
    """
    
    def __init__(self, model: StateSpaceModel, alpha=1e-3, beta=2.0, kappa=0.0):
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
        if sigmas_pred is None:
            sigmas_pred = self.generate_sigma_points(x_pred, P_pred)
            
        sigmas_obs = self.model.observation(sigmas_pred, noise=tf.ones_like(sigmas_pred))
        
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
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        x_init = tf.convert_to_tensor(x_init, dtype=tf.float32)
        P_init = tf.convert_to_tensor(P_init, dtype=tf.float32)
        Q = tf.convert_to_tensor(Q, dtype=tf.float32)
        R = tf.convert_to_tensor(R, dtype=tf.float32)
        return self._run_compiled(observations, x_init, P_init, Q, R)

    @tf.function(reduce_retracing=True)
    def _run_compiled(self, observations, x_init, P_init, Q, R):
        T = tf.shape(observations)[0]
        x = tf.reshape(x_init, (-1, 1))
        P = P_init
        estimates = tf.TensorArray(dtype=tf.float32, size=T)
        
        for t in tf.range(T):
            z = observations[t]
            # Note: Need to manage sigmas_pred passing carefully in graph mode
            # For simplicity in graph, we regenerate sigmas in update or pass explicitly
            # Here calling predict then update matches logic
            x_pred, P_pred, sigmas_pred = self.predict(x, P, Q)
            x, P = self.update(x_pred, P_pred, z, R, sigmas_pred)
            estimates = estimates.write(t, x)
            
        return estimates.stack()