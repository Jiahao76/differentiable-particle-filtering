import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilterTF

def run_experiment():
    """
    Sets up a 2D tracking problem (Multidimensional SSM) to test the Kalman Filter.
    State vector x = [position, velocity]^T.
    """
    print("Initializing Multidimensional Kalman Filter (TensorFlow)...")
    
    # 1. Define Model Parameters (Constant Velocity Model)
    dt = 0.1 # Time step
    
    # State Transition Matrix (F): Position = Position + Velocity * dt
    F = np.array([[1.0, dt],
                  [0.0, 1.0]], dtype=np.float32)
    
    # Observation Matrix (H): We only observe Position, not Velocity
    H = np.array([[1.0, 0.0]], dtype=np.float32)
    
    # Process Noise (Q): Uncertainty in velocity changes
    q_var = 0.1
    Q = np.array([[0.25*dt**4, 0.5*dt**3],
                  [0.5*dt**3,    dt**2]], dtype=np.float32) * q_var
    
    # Observation Noise (R): Noisy position sensor
    R = np.array([[0.5]], dtype=np.float32)
    
    # Initial State
    x_init = np.array([[0.0], [1.0]], dtype=np.float32) # Start at 0, velocity 1
    P_init = np.eye(2, dtype=np.float32) * 5.0
    
    # 2. Generate Synthetic Data (Ground Truth & Observations)
    steps = 100
    true_states = []
    measurements = []
    
    current_x = x_init.copy()
    
    np.random.seed(42) # For reproducibility
    
    for _ in range(steps):
        # Propagate state
        noise_proc = np.random.multivariate_normal([0,0], Q).reshape(-1,1)
        current_x = F @ current_x + noise_proc
        true_states.append(current_x)
        
        # Generate measurement
        noise_meas = np.random.normal(0, np.sqrt(R[0,0]))
        z = H @ current_x + noise_meas
        measurements.append(z)
        
    # 3. Run TensorFlow Kalman Filter
    kf = KalmanFilterTF(F, H, Q, R, x_init, P_init)
    
    est_states = []
    est_covariances = []
    
    print(f"Processing {steps} time steps...")
    
    for z in measurements:
        # Step 1: Predict
        kf.predict()
        
        # Step 2: Update
        x_est, P_est = kf.update(z)
        
        est_states.append(x_est.numpy())
        est_covariances.append(P_est.numpy())
        
    print("Filtering complete.")

    # 4. Visualization
    true_pos = [s[0,0] for s in true_states]
    est_pos = [s[0,0] for s in est_states]
    meas_pos = [m[0,0] for m in measurements]
    
    # Extract confidence intervals (2 * standard deviation)
    std_devs = [np.sqrt(P[0,0]) for P in est_covariances]
    upper_bound = [e + 2*s for e, s in zip(est_pos, std_devs)]
    lower_bound = [e - 2*s for e, s in zip(est_pos, std_devs)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(true_pos, 'k-', label='True State (Position)')
    plt.plot(meas_pos, 'r.', alpha=0.5, label='Noisy Measurements')
    plt.plot(est_pos, 'b-', linewidth=2, label='KF Estimate')
    plt.fill_between(range(steps), lower_bound, upper_bound, color='blue', alpha=0.2, label='95% Confidence')
    
    plt.title(f'Part 1.1.a: Multidimensional Linear-Gaussian SSM (TensorFlow)\nPosition Tracking with Joseph Stabilized Update')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_experiment()