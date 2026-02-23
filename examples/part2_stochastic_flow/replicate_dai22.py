"""
Complete replication of Dai22 Section 4 bearing-only tracking experiment.
Fixes critical issues in the original implementation.
"""

import numpy as np
import tensorflow as tf
import sys
import os

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.bearing_only_tracking import BearingOnlyTrackingModel, BearingOnlyScenario
from filters.particle_flow_filters import TFStochasticParticleFlowFilter
from filters.homotopy_optimizer import HomotopyOptimizer


def run_dai22_experiment():
    """
    Replicate Dai22 Section 4 results with correct parameters.
    
    Key fixes:
    1. Use atan2 instead of atan for measurement function
    2. Correct prior mean: [3.0, 5.0] not [3.0, 0.0]
    3. Proper Hessian computation at ensemble mean
    4. Adaptive regularization for numerical stability
    """
    
    print("="*70)
    print("Dai22 Section 4: Bearing-Only Tracking Experiment")
    print("="*70)
    
    # Initialize scenario
    scenario = BearingOnlyScenario()
    model = scenario.model
    
    # Get measurement (from paper)
    z_sample = scenario.z_sample
    print(f"\nMeasurement z = {z_sample.numpy()}")
    print(f"True target location = {model.target_truth.numpy()}")
    print(f"Prior mean = {model.prior_mean.numpy()}")
    print(f"Prior covariance = \n{model.prior_cov.numpy()}")
    
    # Experiment parameters from Dai22
    n_particles = 50
    n_mc_runs = 20
    mu = 0.2  # Homotopy optimization weight
    
    # Diffusion matrix Q
    Q = np.array([[4.0, 0.0], [0.0, 0.4]], dtype=np.float32)
    print(f"\nDiffusion matrix Q = \n{Q}")
    
    # ================================================================
    # PART 1: Linear homotopy (baseline)
    # ================================================================
    
    print("\n" + "="*70)
    print("PART 1: Linear Homotopy β(λ) = λ")
    print("="*70)
    
    def linear_homotopy(lam):
        """Baseline linear homotopy β(λ) = λ"""
        lam_t = tf.cast(lam, tf.float32)
        alpha = 1.0 - lam_t
        beta = lam_t
        alpha_dot = tf.constant(-1.0, dtype=tf.float32)
        beta_dot = tf.constant(1.0, dtype=tf.float32)
        return alpha, beta, alpha_dot, beta_dot
    
    # Run Monte Carlo simulations
    mse_linear = []
    trace_linear = []
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print(f"\nRunning {n_mc_runs} Monte Carlo runs with {n_particles} particles...")
    
    for mc_run in range(n_mc_runs):
        # Sample from prior
        particles = scenario.sample_prior(n_particles, seed=42+mc_run)
        
        # Initialize filter
        filter_linear = TFStochasticParticleFlowFilter(
            n_dim=2,
            q_matrix=Q,
            mu=mu
        )
        
        # Flow particles
        particles_final = filter_linear.flow_particles(
            particles=particles,
            model=model,
            z=z_sample,
            beta_func=linear_homotopy,
            n_steps=200,
            verbose=False
        )
        
        # Compute metrics
        mse, trace_cov, post_mean = scenario.compute_metrics(particles_final)
        mse_linear.append(mse)
        trace_linear.append(trace_cov)
        
        if mc_run < 3 or mc_run == n_mc_runs - 1:
            print(f"  Run {mc_run+1}: MSE = {mse:.4f}, tr(P) = {trace_cov:.4f}, mean = {post_mean}")
    
    mse_linear_avg = np.mean(mse_linear)
    trace_linear_avg = np.mean(trace_linear)
    
    print(f"\nLinear Homotopy Results:")
    print(f"  Average MSE = {mse_linear_avg:.4f}")
    print(f"  Average tr(P) = {trace_linear_avg:.4f}")
    
    # ================================================================
    # PART 2: Optimal homotopy
    # ================================================================
    
    print("\n" + "="*70)
    print("PART 2: Optimal Homotopy β*(λ) from TPBVP")
    print("="*70)
    
    # Compute Hessians at ensemble mean
    # This is critical: use ensemble mean, not individual particles
    particles_init = scenario.sample_prior(n_particles, seed=42)
    ensemble_mean = tf.reduce_mean(particles_init, axis=0)
    
    print(f"\nEnsemble mean for Hessian computation: {ensemble_mean.numpy()}")
    
    # Compute Hessians
    # Use numerical Hessian with careful finite differencing
    # Analytical (Gauss-Newton) approximation can be very inaccurate for nonlinear problems
    M0 = model.hessian_log_prior()
    Mh = model.hessian_log_likelihood_numerical(ensemble_mean, z_sample, eps=1e-3)
    
    print(f"\nHessian of log-prior (M0):")
    print(M0.numpy())
    print(f"\nHessian of log-likelihood at ensemble mean (Mh):")
    print(Mh.numpy())
    
    # Check condition numbers
    cond_M0 = np.linalg.cond(M0.numpy())
    cond_Mh = np.linalg.cond(Mh.numpy())
    print(f"\nCondition numbers:")
    print(f"  κ(M0) = {cond_M0:.2e}")
    print(f"  κ(Mh) = {cond_Mh:.2e}")
    
    # Solve optimal homotopy
    # Note: optimizer.solve_optimal_homotopy() expects POSITIVE semi-definite matrices
    # (energy/negative log-density Hessians).
    # But model.hessian_log_prior() and hessian_log_likelihood_analytical() return
    # NEGATIVE Hessians (they are negative because log densities are concave).
    # So we must negate them before passing to optimizer.
    optimizer = HomotopyOptimizer(mu=mu, norm_type='nuclear')
    
    beta_func_optimal = optimizer.solve_optimal_homotopy(
        M0=-M0,   # Negate to convert to positive semi-definite (energy Hessian)
        Mh=-Mh,   # Negate to convert to positive semi-definite (energy Hessian)
        method='auto'  # Try solve_bvp first, fall back to shooting
    )
    
    # Store optimal solution if available for later visualization
    lambda_grid_opt = None
    beta_vals_opt = None
    if hasattr(optimizer, 'lambda_grid') and optimizer.lambda_grid is not None:
        lambda_grid_opt = optimizer.lambda_grid
        beta_vals_opt = optimizer.beta_vals
    
    if beta_func_optimal is None:
        print("\nWarning: Optimal homotopy solver failed. Using linear homotopy.")
        beta_func_optimal = linear_homotopy
    
    # Run Monte Carlo simulations with optimal homotopy
    mse_optimal = []
    trace_optimal = []
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print(f"\nRunning {n_mc_runs} Monte Carlo runs with optimal homotopy...")
    
    for mc_run in range(n_mc_runs):
        # Sample from prior (same seed as baseline)
        particles = scenario.sample_prior(n_particles, seed=42+mc_run)
        
        # Initialize filter
        filter_optimal = TFStochasticParticleFlowFilter(
            n_dim=2,
            q_matrix=Q,
            mu=mu
        )
        
        # Flow particles
        particles_final = filter_optimal.flow_particles(
            particles=particles,
            model=model,
            z=z_sample,
            beta_func=beta_func_optimal,
            n_steps=200,
            verbose=False
        )
        
        # Compute metrics
        mse, trace_cov, post_mean = scenario.compute_metrics(particles_final)
        mse_optimal.append(mse)
        trace_optimal.append(trace_cov)
        
        if mc_run < 3 or mc_run == n_mc_runs - 1:
            print(f"  Run {mc_run+1}: MSE = {mse:.4f}, tr(P) = {trace_cov:.4f}, mean = {post_mean}")
    
    mse_optimal_avg = np.mean(mse_optimal)
    trace_optimal_avg = np.mean(trace_optimal)
    
    print(f"\nOptimal Homotopy Results:")
    print(f"  Average MSE = {mse_optimal_avg:.4f}")
    print(f"  Average tr(P) = {trace_optimal_avg:.4f}")
    
    # ================================================================
    # PART 3: Comparison and Analysis
    # ================================================================
    
    print("\n" + "="*70)
    print("COMPARISON: Linear vs Optimal Homotopy")
    print("="*70)
    
    # Create comparison table
    print(f"\n{'Metric':<20} {'Linear':<15} {'Optimal':<15} {'Improvement':<15}")
    print("-"*65)
    
    mse_improvement = (mse_linear_avg - mse_optimal_avg) / mse_linear_avg * 100
    trace_improvement = (trace_linear_avg - trace_optimal_avg) / trace_linear_avg * 100
    
    print(f"{'Average MSE':<20} {mse_linear_avg:<15.4f} {mse_optimal_avg:<15.4f} {mse_improvement:>14.2f}%")
    print(f"{'Average tr(P)':<20} {trace_linear_avg:<15.4f} {trace_optimal_avg:<15.4f} {trace_improvement:>14.2f}%")
    
    # Statistical significance test
    from scipy import stats
    t_stat_mse, p_value_mse = stats.ttest_rel(mse_linear, mse_optimal)
    t_stat_trace, p_value_trace = stats.ttest_rel(trace_linear, trace_optimal)
    
    print(f"\nStatistical Significance (paired t-test):")
    print(f"  MSE: t = {t_stat_mse:.4f}, p = {p_value_mse:.4f}")
    print(f"  tr(P): t = {t_stat_trace:.4f}, p = {p_value_trace:.4f}")
    
    # Detailed MC run comparison
    print(f"\n{'Run':<6} {'MSE (Linear)':<15} {'MSE (Optimal)':<15} {'tr(P) Linear':<15} {'tr(P) Optimal':<15}")
    print("-"*75)
    for i in range(min(10, n_mc_runs)):
        print(f"{i+1:<6} {mse_linear[i]:<15.4f} {mse_optimal[i]:<15.4f} {trace_linear[i]:<15.4f} {trace_optimal[i]:<15.4f}")
    
    if n_mc_runs > 10:
        print("  ...")
        i = n_mc_runs - 1
        print(f"{i+1:<6} {mse_linear[i]:<15.4f} {mse_optimal[i]:<15.4f} {trace_linear[i]:<15.4f} {trace_optimal[i]:<15.4f}")
    
    print("\n" + "="*70)
    print("Experiment completed successfully!")
    print("="*70)
    
    # ================================================================
    # PART 4: Visualization of optimal homotopy curve
    # ================================================================
    
    if lambda_grid_opt is not None and beta_vals_opt is not None:
        print("\n" + "="*70)
        print("PART 4: Optimal Homotopy Curve Analysis")
        print("="*70)
        
        print(f"\nOptimal β*(λ) trajectory:")
        print(f"  β*(0) = {beta_vals_opt[0]:.6f} (expected: 0.0)")
        print(f"  β*(1) = {beta_vals_opt[-1]:.6f} (expected: 1.0)")
        
        # Compute some intermediate values
        idx_quarter = len(lambda_grid_opt) // 4
        idx_half = len(lambda_grid_opt) // 2
        idx_three_quarter = 3 * len(lambda_grid_opt) // 4
        
        print(f"\n  β*(0.25) = {beta_vals_opt[idx_quarter]:.6f}")
        print(f"  β*(0.50) = {beta_vals_opt[idx_half]:.6f}")
        print(f"  β*(0.75) = {beta_vals_opt[idx_three_quarter]:.6f}")
        
        # Check if curve is non-linear (compare to linear β(λ) = λ)
        linear_curve = lambda_grid_opt
        deviation = np.mean(np.abs(beta_vals_opt - linear_curve))
        print(f"\n  Mean deviation from linear homotopy: {deviation:.6f}")
        
        # Determine curvature type
        if beta_vals_opt[idx_half] < 0.5:
            print(f"  Curve type: CONCAVE (below linear)")
        elif beta_vals_opt[idx_half] > 0.5:
            print(f"  Curve type: CONVEX (above linear)")
        else:
            print(f"  Curve type: APPROXIMATELY LINEAR")
        
        # Optional: Save trajectory for plotting
        trajectory_file = os.path.join(os.path.dirname(__file__), '../results/optimal_homotopy.npy')
        os.makedirs(os.path.dirname(trajectory_file), exist_ok=True)
        np.save(trajectory_file, np.array([lambda_grid_opt, beta_vals_opt]))
        print(f"\n  Trajectory saved to: {trajectory_file}")
    
    return {
        'linear': {
            'mse': mse_linear,
            'trace': trace_linear,
            'mse_avg': mse_linear_avg,
            'trace_avg': trace_linear_avg
        },
        'optimal': {
            'mse': mse_optimal,
            'trace': trace_optimal,
            'mse_avg': mse_optimal_avg,
            'trace_avg': trace_optimal_avg
        },
        'homotopy': {
            'lambda_grid': lambda_grid_opt,
            'beta_vals': beta_vals_opt
        }
    }


if __name__ == "__main__":
    results = run_dai22_experiment()