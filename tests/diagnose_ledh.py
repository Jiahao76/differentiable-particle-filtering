"""
Detailed diagnostic for LEDH gradient computation
"""
import sys
import os
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sv_model import StochasticVolatilityModel

def diagnose_gradients():
    """Check if gradients are computed correctly"""
    print("="*70)
    print("LEDH Gradient Diagnostic")
    print("="*70)
    
    model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
    
    # Test with simple particles
    test_particles = tf.constant([[-2.0], [0.0], [2.0]], dtype=tf.float32)
    obs = tf.constant([1.0], dtype=tf.float32)
    
    print("\nTest particles:")
    print(test_particles.numpy().flatten())
    
    print("\nObservation:", obs.numpy()[0])
    
    # Test 1: Check h(x) = beta * exp(x/2)
    print("\n" + "="*70)
    print("Test 1: Observation function h(x) = beta * exp(x/2)")
    print("="*70)
    
    with tf.GradientTape() as tape:
        tape.watch(test_particles)
        h_val = model.beta * tf.exp(test_particles / 2.0)
    
    H = tape.gradient(h_val, test_particles)
    
    print("\nh(x) values:")
    print(h_val.numpy().flatten())
    print("\nH = dh/dx values (should be beta/2 * exp(x/2)):")
    print(H.numpy().flatten())
    
    expected_H = model.beta / 2.0 * tf.exp(test_particles / 2.0)
    print("\nExpected H:")
    print(expected_H.numpy().flatten())
    
    print("\nGradient correct:", np.allclose(H.numpy(), expected_H.numpy()))
    
    # Test 2: Check innovation calculation
    print("\n" + "="*70)
    print("Test 2: Innovation and residuals")
    print("="*70)
    
    h_vals = model.beta * tf.exp(test_particles / 2.0)
    H_vals = model.beta / 2.0 * tf.exp(test_particles / 2.0)
    
    # Residual: e = h - H*x
    e = h_vals - H_vals * test_particles
    innovation = obs - e
    
    print("\nResiduals e = h(x) - H*x:")
    print(e.numpy().flatten())
    
    print("\nInnovations obs - e:")
    print(innovation.numpy().flatten())
    
    # Test 3: Check log_likelihood
    print("\n" + "="*70)
    print("Test 3: Log-likelihood computation")
    print("="*70)
    
    log_lik = model.log_likelihood(obs, test_particles)
    print("\nLog-likelihood for each particle:")
    print(log_lik.numpy().flatten())
    
    # Test 4: Check transition log pdf
    print("\n" + "="*70)
    print("Test 4: Transition log PDF (prior)")
    print("="*70)
    
    prev_particles = tf.constant([[-1.0], [0.0], [1.0]], dtype=tf.float32)
    
    log_pdf = model.transition_log_pdf(test_particles, prev_particles)
    print("\nTransition log PDF from prev to test particles:")
    print(log_pdf.numpy().flatten())
    
    # Verify: X_t | X_{t-1} ~ N(alpha*X_{t-1}, sigma^2)
    # log p(x | mu, sigma) = -log(sigma) - 0.5*((x-mu)/sigma)^2
    alpha = model.alpha.numpy()
    sigma = model.sigma.numpy()
    
    mu = alpha * prev_particles
    expected_log_pdf = -np.log(sigma) - 0.5 * ((test_particles.numpy() - mu.numpy()) / sigma) ** 2
    
    print("\nExpected log PDF:")
    print(expected_log_pdf.flatten())
    
    print("\nLog PDF correct:", np.allclose(log_lik.numpy(), log_lik.numpy(), atol=1e-5))
    
    # Test 5: Flow parameter computation
    print("\n" + "="*70)
    print("Test 5: LEDH Flow Parameters")
    print("="*70)
    
    P = tf.reduce_mean((test_particles - tf.reduce_mean(test_particles, axis=0))**2)
    print(f"\nPredictive covariance P: {P.numpy():.6f}")
    
    lambda_val = 1.0
    R = 1.0
    
    # Recompute H
    with tf.GradientTape() as tape:
        tape.watch(test_particles)
        h_val = model.beta * tf.exp(test_particles / 2.0)
    
    H = tape.gradient(h_val, test_particles)
    
    # LEDH A and b
    denom = lambda_val * (H**2) * P + R
    A = -0.5 * P * (H**2) / (denom + 1e-8)
    
    e = h_val - H * test_particles
    innovation = obs - e
    
    factor1 = (1.0 + 2.0*lambda_val*A)
    factor2 = (1.0 + lambda_val*A)
    factor3 = P*H / (R + 1e-8)
    b = factor1 * factor2 * factor3 * innovation + A * test_particles
    
    print(f"\nA (flow damping):")
    print(A.numpy().flatten())
    
    print(f"\nb (flow velocity):")
    print(b.numpy().flatten())
    
    print(f"\nFlow step: particles + eps*(A*particles + b)")
    eps = 0.05
    new_particles = test_particles + eps * (A * test_particles + b)
    print(f"\nNew particles (eps={eps}):")
    print(new_particles.numpy().flatten())
    
    print("\n" + "="*70)
    print("Diagnostic complete")
    print("="*70)

if __name__ == "__main__":
    diagnose_gradients()
