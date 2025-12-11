import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from models.sv_model import StochasticVolatilityModelTF # Import from previous step

class EKF_SV:
    """
    Extended Kalman Filter (EKF) for Stochastic Volatility Model.
    
    Strategy:
        We filter the squared observations z_t = y_t^2 to handle multiplicative noise.
        Measurement Model: z_t = beta^2 * exp(x_t) * w_t^2
        Approximation: z_t approx h(x_t) + noise
        h(x_t) = beta^2 * exp(x_t)
    """
    def __init__(self, alpha, sigma, beta, P_init=1.0):
        self.alpha = tf.cast(alpha, tf.float32)
        self.sigma = tf.cast(sigma, tf.float32)
        self.beta = tf.cast(beta, tf.float32)
        
        # Initial Covariance
        self.P = tf.constant([[P_init]], dtype=tf.float32)
        self.x = tf.constant([[0.0]], dtype=tf.float32) # Initial state estimate

    def predict(self):
        # 1. State Prediction: x_{k|k-1} = alpha * x_{k-1|k-1}
        self.x = self.alpha * self.x
        
        # 2. Covariance Prediction: P_{k|k-1} = F P F^T + Q
        # F (Jacobian of state transition) is just alpha for linear transition
        F = self.alpha
        Q = self.sigma**2
        self.P = F * self.P * F + Q
        return self.x, self.P

    def update(self, y_meas):
        # Transform observation: z = y^2
        z_meas = tf.square(y_meas)
        z_meas = tf.reshape(z_meas, [1, 1])
        
        # 1. Measurement Prediction: h(x) = beta^2 * exp(x)
        # We use GradientTape to compute Jacobian H automatically (Modern EKF)
        with tf.GradientTape() as tape:
            tape.watch(self.x)
            h_x = tf.square(self.beta) * tf.exp(self.x)
            
        # 2. Jacobian H = dh/dx
        H = tape.gradient(h_x, self.x)
        
        # 3. Dynamic Observation Noise R
        # Since z = h(x) * w^2, the noise variance depends on the state!
        # Var(z) = h(x)^2 * Var(w^2) = h(x)^2 * 2  (since w~N(0,1), w^2~Chi2(1))
        # We approximate using current estimate
        R = tf.square(h_x) * 2.0 + 1e-4 # Add epsilon for stability
        
        # 4. Innovation
        y_tilde = z_meas - h_x
        
        # 5. Innovation Covariance S = H P H^T + R
        S = H * self.P * H + R
        
        # 6. Kalman Gain K = P H^T S^-1
        K = (self.P * H) / S
        
        # 7. Update State & Covariance
        self.x = self.x + K * y_tilde
        self.P = (1.0 - K * H) * self.P
        
        return self.x, self.P

class UKF_SV:
    """
    Unscented Kalman Filter (UKF) for Stochastic Volatility Model.
    Uses Sigma Points to handle the nonlinearity in h(x) = beta^2 * exp(x).
    """
    def __init__(self, alpha, sigma, beta, P_init=1.0):
        self.alpha = tf.cast(alpha, tf.float32)
        self.sigma = tf.cast(sigma, tf.float32)
        self.beta = tf.cast(beta, tf.float32)
        
        self.P = tf.constant([[P_init]], dtype=tf.float32)
        self.x = tf.constant([[0.0]], dtype=tf.float32)
        
        # UKF Parameters
        self.kappa = 3.0 - 1.0 # dim=1
        self.w0 = self.kappa / (1.0 + self.kappa)
        self.wi = 0.5 / (1.0 + self.kappa)

    def generate_sigma_points(self, x, P):
        sigma = tf.sqrt(P)
        # Points: mean, mean + sqrt, mean - sqrt
        sp0 = x
        sp1 = x + tf.sqrt(1.0 + self.kappa) * sigma
        sp2 = x - tf.sqrt(1.0 + self.kappa) * sigma
        return [sp0, sp1, sp2]

    def predict(self):
        # 1. Generate Sigma Points
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # 2. Propagate through State Transition (Linear here, but general methodology)
        # f(x) = alpha * x
        propagated_sp = [self.alpha * sp for sp in sigma_points]
        
        # 3. Compute Predicted Mean
        x_pred = self.w0 * propagated_sp[0] + self.wi * (propagated_sp[1] + propagated_sp[2])
        
        # 4. Compute Predicted Covariance
        # P = sum w * (sp - mean)(sp - mean)^T + Q
        Q = self.sigma**2
        diff0 = propagated_sp[0] - x_pred
        diff1 = propagated_sp[1] - x_pred
        diff2 = propagated_sp[2] - x_pred
        
        P_pred = (self.w0 * diff0**2 + 
                  self.wi * diff1**2 + 
                  self.wi * diff2**2) + Q
                  
        self.x = x_pred
        self.P = P_pred
        return self.x, self.P

    def update(self, y_meas):
        z_meas = tf.square(y_meas)
        z_meas = tf.reshape(z_meas, [1, 1])
        
        # 1. Generate Sigma Points from predicted state
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # 2. Propagate through Measurement Function
        # h(x) = beta^2 * exp(x)
        obs_sp = [tf.square(self.beta) * tf.exp(sp) for sp in sigma_points]
        
        # 3. Predicted Observation Mean
        z_pred = self.w0 * obs_sp[0] + self.wi * (obs_sp[1] + obs_sp[2])
        
        # 4. Innovation Covariance S
        # R is state dependent: R approx z_pred^2 * 2
        R = tf.square(z_pred) * 2.0 + 1e-4
        
        diff_z0 = obs_sp[0] - z_pred
        diff_z1 = obs_sp[1] - z_pred
        diff_z2 = obs_sp[2] - z_pred
        
        S = (self.w0 * diff_z0**2 + 
             self.wi * diff_z1**2 + 
             self.wi * diff_z2**2) + R
             
        # 5. Cross Covariance P_xz
        diff_x0 = sigma_points[0] - self.x
        diff_x1 = sigma_points[1] - self.x
        diff_x2 = sigma_points[2] - self.x
        
        P_xz = (self.w0 * diff_x0 * diff_z0 +
                self.wi * diff_x1 * diff_z1 +
                self.wi * diff_x2 * diff_z2)
                
        # 6. Update
        K = P_xz / S
        self.x = self.x + K * (z_meas - z_pred)
        self.P = self.P - K * S * K # K * S * K^T (scalar K^T = K)
        
        return self.x, self.P

# ============================================================================
# Comparison Experiment
# ============================================================================

def run_comparison():
    print("Generating Data...")
    sv_model = StochasticVolatilityModelTF(alpha=0.91, sigma=1.0, beta=0.5)
    T = 100
    true_states, observations = sv_model.generate_data(T)
    
    # Initialize Filters
    ekf = EKF_SV(0.91, 1.0, 0.5)
    ukf = UKF_SV(0.91, 1.0, 0.5)
    
    ekf_est = []
    ukf_est = []
    
    print("Running Filters...")
    for y in observations:
        # EKF Step
        ekf.predict()
        x_e, _ = ekf.update(y)
        ekf_est.append(x_e[0,0].numpy())
        
        # UKF Step
        ukf.predict()
        x_u, _ = ukf.update(y)
        ukf_est.append(x_u[0,0].numpy())
        
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(true_states, 'k-', label='True State', linewidth=1.5)
    plt.plot(ekf_est, 'b--', label='EKF Estimate')
    plt.plot(ukf_est, 'r-.', label='UKF Estimate')
    plt.title('Comparison: EKF vs UKF on Stochastic Volatility Model')
    plt.xlabel('Time Step')
    plt.ylabel('Log-Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate RMSE
    rmse_ekf = np.sqrt(np.mean((np.array(ekf_est) - true_states.numpy())**2))
    rmse_ukf = np.sqrt(np.mean((np.array(ukf_est) - true_states.numpy())**2))
    
    print(f"RMSE EKF: {rmse_ekf:.4f}")
    print(f"RMSE UKF: {rmse_ukf:.4f}")
    print("-" * 50)
    print("Analysis of Linearization:")
    print("The observation function h(x) = beta^2 * exp(x) is highly convex.")
    print("EKF linearizes h(x) at the mean, which systematically underestimates")
    print("the mean of the transformed distribution (Jensen's Inequality).")
    print("UKF captures the posterior mean better by propagating sigma points")
    print("through the nonlinearity.")

if __name__ == "__main__":
    run_comparison()