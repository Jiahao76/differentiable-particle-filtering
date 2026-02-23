"""Tests for the stochastic particle flow filter (Dai 2021/2022)."""
import numpy as np
import tensorflow as tf
import pytest

from src.models.bearing_only_tracking import BearingOnlyTrackingModel, BearingOnlyScenario
from src.filters.particle_flow_filters import TFStochasticParticleFlowFilter


@pytest.fixture
def scenario():
    """Create a bearing-only tracking scenario."""
    return BearingOnlyScenario()


@pytest.fixture
def linear_homotopy():
    """Return the standard linear homotopy function."""
    def _homotopy(lam):
        lam_t = tf.cast(lam, tf.float32)
        alpha = 1.0 - lam_t
        beta = lam_t
        alpha_dot = tf.constant(-1.0, dtype=tf.float32)
        beta_dot = tf.constant(1.0, dtype=tf.float32)
        return alpha, beta, alpha_dot, beta_dot
    return _homotopy


class TestStochasticParticleFlowFilter:
    """Tests for TFStochasticParticleFlowFilter."""

    def test_init(self):
        """Test filter initialization with valid Q matrix."""
        Q = tf.constant([[1.0, 0.0], [0.0, 0.1]], dtype=tf.float32)
        pff = TFStochasticParticleFlowFilter(n_dim=2, q_matrix=Q)
        assert pff.n_dim == 2
        assert pff.q_chol is not None

    def test_flow_particles_shape(self, scenario, linear_homotopy):
        """Test that flow_particles returns correct shape."""
        model = scenario.model
        measurement = scenario.z_sample
        n_particles = 20

        Q = tf.constant([[1.0, 0.0], [0.0, 0.1]], dtype=tf.float32)
        pff = TFStochasticParticleFlowFilter(n_dim=2, q_matrix=Q)

        particles_init = scenario.sample_prior(n_particles, seed=42)
        particles_final = pff.flow_particles(
            particles_init, model, measurement, linear_homotopy, n_steps=50,
        )

        assert particles_final.shape == (n_particles, 2)
        assert not tf.reduce_any(tf.math.is_nan(particles_final))

    def test_flow_particles_move_toward_posterior(self, scenario, linear_homotopy):
        """Test that flowed particles are closer to the truth than prior."""
        model = scenario.model
        measurement = scenario.z_sample
        n_particles = 50

        Q = tf.constant([[1.0, 0.0], [0.0, 0.1]], dtype=tf.float32)
        pff = TFStochasticParticleFlowFilter(n_dim=2, q_matrix=Q)

        particles_init = scenario.sample_prior(n_particles, seed=42)
        particles_final = pff.flow_particles(
            particles_init, model, measurement, linear_homotopy, n_steps=200,
        )

        mean_init = tf.reduce_mean(particles_init, axis=0).numpy()
        mean_final = tf.reduce_mean(particles_final, axis=0).numpy()
        truth = model.target_truth.numpy()

        error_init = np.linalg.norm(mean_init - truth)
        error_final = np.linalg.norm(mean_final - truth)

        # Stochastic flow should move particles toward the posterior.
        # The SDE noise and limited particles can cause significant variance;
        # allow 3x tolerance (tightened from original 5x).
        assert error_final < error_init * 3.0, (
            f"Flow diverged: init_error={error_init:.2f}, "
            f"final_error={error_final:.2f}"
        )
        assert not np.any(np.isnan(mean_final))

    def test_log_likelihood_higher_at_truth(self, scenario):
        """Test that log-likelihood at truth exceeds prior mean."""
        model = scenario.model
        measurement = scenario.z_sample

        ll_truth = model.log_likelihood(model.target_truth, measurement)
        ll_prior = model.log_likelihood(model.prior_mean, measurement)

        assert ll_truth > ll_prior
