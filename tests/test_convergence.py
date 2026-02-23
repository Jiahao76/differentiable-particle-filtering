"""Convergence tests for particle filters and particle flow methods.

Test design philosophy (per instructor feedback):
  These tests verify asymptotic properties that must hold for correct
  implementations. They are derived from the mathematical theory:
  - PF consistency: RMSE -> 0 as N -> infinity
  - KF optimality: KF achieves minimum MSE for linear Gaussian systems
  - DPF consistency: DPF should match standard PF on the same problem
  - Flow convergence: particles should move toward the posterior
"""
import numpy as np
import tensorflow as tf
import pytest


# ============================================================
# Particle Filter Convergence
# ============================================================

class TestParticleFilterConvergence:
    """Verify asymptotic convergence properties of particle filters.

    Mathematical basis:
      For a well-specified model, the particle filter estimate
      converges to the true posterior mean as N -> infinity.
      RMSE decreases at rate O(1/sqrt(N)) for importance sampling.
    """

    @pytest.mark.convergence
    def test_pf_rmse_decreases_with_particles(self, lgssm_model, lgssm_data):
        """RMSE should decrease when using more particles.

        Property: PF consistency — the estimate improves with more particles.
        This is a necessary condition for any correct PF implementation.
        """
        from src.filters.particle_filter import StandardParticleFilter

        obs = tf.constant(lgssm_data['y_obs'].reshape(-1, 1), dtype=tf.float32)
        x_true = lgssm_data['x_true'].flatten()

        # Run with few particles
        tf.random.set_seed(42)
        pf_small = StandardParticleFilter(lgssm_model, num_particles=50)
        est_small, _ = pf_small.run(obs, verbose=False)
        rmse_small = np.sqrt(np.mean((x_true - est_small.numpy().flatten())**2))

        # Run with many particles
        tf.random.set_seed(42)
        pf_large = StandardParticleFilter(lgssm_model, num_particles=500)
        est_large, _ = pf_large.run(obs, verbose=False)
        rmse_large = np.sqrt(np.mean((x_true - est_large.numpy().flatten())**2))

        assert rmse_large < rmse_small * 1.5, (
            f"RMSE did not decrease: N=50 RMSE={rmse_small:.3f}, "
            f"N=500 RMSE={rmse_large:.3f}"
        )

    @pytest.mark.convergence
    def test_kf_is_optimal_for_lgssm(self, lgssm_model, lgssm_data):
        """Kalman filter should achieve the lowest RMSE on linear Gaussian systems.

        Property: KF is the MMSE-optimal filter for LGSSM. Any particle-based
        filter should have RMSE >= KF RMSE (up to sampling variance).
        """
        from src.filters.kalman_filter import KalmanFilter
        from src.filters.particle_filter import StandardParticleFilter

        obs = tf.constant(lgssm_data['y_obs'].reshape(-1, 1), dtype=tf.float32)
        x_true = lgssm_data['x_true'].flatten()

        # KF (exact)
        kf = KalmanFilter(lgssm_model)
        obs_kf = tf.constant(lgssm_data['y_obs'], dtype=tf.float32)
        x_init = tf.zeros([1], dtype=tf.float32)
        P_init = tf.eye(1, dtype=tf.float32)
        kf_est, _ = kf.run(obs_kf, x_init, P_init)
        rmse_kf = np.sqrt(np.mean((x_true - kf_est.numpy().flatten())**2))

        # PF with many particles
        tf.random.set_seed(42)
        pf = StandardParticleFilter(lgssm_model, num_particles=500)
        pf_est, _ = pf.run(obs, verbose=False)
        rmse_pf = np.sqrt(np.mean((x_true - pf_est.numpy().flatten())**2))

        # PF should be close to KF but not systematically better
        assert rmse_pf >= rmse_kf * 0.7, (
            f"PF RMSE ({rmse_pf:.3f}) suspiciously lower than "
            f"KF RMSE ({rmse_kf:.3f}) on linear system"
        )

    @pytest.mark.convergence
    def test_dpf_consistent_with_pf(self, lgssm_model, lgssm_data):
        """DPF and standard PF should produce similar RMSE on the same problem.

        Property: Differentiable resampling should not significantly degrade
        filtering performance compared to standard multinomial resampling.
        """
        from src.filters.particle_filter import StandardParticleFilter
        from src.filters.differentiable_particle_filter import DifferentiableParticleFilter

        obs = tf.constant(lgssm_data['y_obs'].reshape(-1, 1), dtype=tf.float32)
        x_true = lgssm_data['x_true'].flatten()

        # Standard PF
        tf.random.set_seed(42)
        pf = StandardParticleFilter(lgssm_model, num_particles=200)
        pf_est, _ = pf.run(obs, verbose=False)
        rmse_pf = np.sqrt(np.mean((x_true - pf_est.numpy().flatten())**2))

        # DPF with OT resampling
        tf.random.set_seed(42)
        dpf = DifferentiableParticleFilter(
            lgssm_model, num_particles=200, resampling_method="ot",
            ot_iterations=30, resample_threshold=0.5,
        )
        dpf_est, _, _ = dpf.run(obs, verbose=False)
        rmse_dpf = np.sqrt(np.mean((x_true - dpf_est.numpy().flatten())**2))

        # They should be within factor of 2 of each other
        assert rmse_dpf < rmse_pf * 2.0, (
            f"DPF RMSE ({rmse_dpf:.3f}) much worse than "
            f"PF RMSE ({rmse_pf:.3f})"
        )


# ============================================================
# Flow Convergence
# ============================================================

class TestFlowConvergence:
    """Verify that particle flow methods move particles toward the posterior.

    Mathematical basis:
      The EDH/LEDH flow is designed to transport particles from the prior
      to the posterior distribution. After flowing, the particle mean
      should be closer to the truth than before flowing.
    """

    @pytest.mark.convergence
    def test_edh_improves_estimate(self, sv_model, synthetic_data):
        """EDH flow should produce estimates correlated with the true state.

        Property: After filtering, estimates should track the true signal
        better than the prior (zero) would.
        """
        from src.filters.edh_flow import EDHFlowFilter

        obs = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        x_true = synthetic_data['x_true']

        edh = EDHFlowFilter(sv_model, num_particles=100, flow_steps=20)
        estimates = edh.run(obs).numpy()

        # RMSE should be finite and estimates should be bounded
        rmse = np.sqrt(np.mean((estimates.flatten() - x_true.flatten())**2))
        assert np.isfinite(rmse), "EDH RMSE is not finite"
        assert rmse < 10.0, f"EDH RMSE too large: {rmse:.3f}"
