"""Comprehensive tests for all models, filters, and inference algorithms."""
import numpy as np
import tensorflow as tf
import pytest


# ============================================================
# Model Tests
# ============================================================

class TestLinearGaussianSSM:
    """Tests for the LinearGaussianSSM model."""

    def test_transition_shape(self, lgssm_model):
        """Transition preserves batch shape."""
        x = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        x_next = lgssm_model.transition(x)
        assert x_next.shape == (2, 1)

    def test_transition_deterministic_with_zero_noise(self, lgssm_model):
        """Passing zero noise gives deterministic mean."""
        x = tf.constant([[1.0]], dtype=tf.float32)
        noise = tf.zeros_like(x)
        x_next = lgssm_model.transition(x, noise=noise)
        expected = tf.matmul(x, lgssm_model.F, transpose_b=True)
        np.testing.assert_allclose(x_next.numpy(), expected.numpy(), atol=1e-6)

    def test_observation_mean_shape(self, lgssm_model):
        """observation_mean returns correct shape."""
        x = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        h_x = lgssm_model.observation_mean(x)
        assert h_x.shape == (2, 1)

    def test_observation_adds_noise(self, lgssm_model):
        """observation differs from observation_mean due to noise."""
        x = tf.constant([[1.0]], dtype=tf.float32)
        h_x = lgssm_model.observation_mean(x)
        y_samples = [lgssm_model.observation(x) for _ in range(50)]
        y_mean = tf.reduce_mean(tf.stack(y_samples), axis=0)
        np.testing.assert_allclose(y_mean.numpy(), h_x.numpy(), atol=0.5)

    def test_log_likelihood_shape(self, lgssm_model):
        """log_likelihood returns one value per particle."""
        x = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)
        y = tf.constant([0.5], dtype=tf.float32)
        ll = lgssm_model.log_likelihood(y, x)
        assert ll.shape == (3,)

    def test_log_likelihood_peaked_at_truth(self, lgssm_model):
        """Closer particles get higher log-likelihood."""
        x_close = tf.constant([[0.5]], dtype=tf.float32)
        x_far = tf.constant([[10.0]], dtype=tf.float32)
        y = tf.constant([0.5], dtype=tf.float32)
        ll_close = lgssm_model.log_likelihood(y, x_close)
        ll_far = lgssm_model.log_likelihood(y, x_far)
        assert ll_close > ll_far

    def test_transition_log_pdf(self, lgssm_model):
        """Transition log-pdf is higher near the predicted mean."""
        x_prev = tf.constant([[1.0]], dtype=tf.float32)
        x_near = tf.constant([[0.9]], dtype=tf.float32)
        x_far = tf.constant([[5.0]], dtype=tf.float32)
        lp_near = lgssm_model.transition_log_pdf(x_near, x_prev)
        lp_far = lgssm_model.transition_log_pdf(x_far, x_prev)
        assert lp_near > lp_far


class TestNonlinearSSM:
    """Tests for the NonlinearSSM (Andrieu 2010)."""

    def test_transition_shape(self, nonlinear_model):
        """Transition preserves shape."""
        x = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        nonlinear_model.reset_time()
        x_next = nonlinear_model.transition(x, time_step=0)
        assert x_next.shape == (2, 1)

    def test_observation_mean_quadratic(self, nonlinear_model):
        """h(x) = x^2 / 20."""
        x = tf.constant([[4.0]], dtype=tf.float32)
        h_x = nonlinear_model.observation_mean(x)
        np.testing.assert_allclose(h_x.numpy(), [[0.8]], atol=1e-6)

    def test_log_likelihood_shape(self, nonlinear_model):
        """log_likelihood returns one value per particle."""
        x = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)
        y = tf.constant([0.5], dtype=tf.float32)
        ll = nonlinear_model.log_likelihood(y, x)
        assert ll.shape == (3,)

    def test_sample_trajectory(self, nonlinear_model):
        """sample_trajectory returns correct length."""
        states, obs = nonlinear_model.sample_trajectory(T=20, seed=42)
        assert states.shape == (20,)
        assert obs.shape == (20,)


class TestSVModelObservationMean:
    """Tests for the SV model observation_mean method."""

    def test_observation_mean_at_zero(self, sv_model):
        """At x=0, h(x) = beta * exp(0) = beta."""
        x = tf.constant([[0.0]], dtype=tf.float32)
        h_x = sv_model.observation_mean(x)
        np.testing.assert_allclose(h_x.numpy(), [[0.5]], atol=1e-6)

    def test_observation_mean_positive(self, sv_model):
        """h(x) should be positive for all x."""
        x = tf.constant([[-5.0], [0.0], [5.0]], dtype=tf.float32)
        h_x = sv_model.observation_mean(x)
        assert tf.reduce_all(h_x > 0)


# ============================================================
# Filter Tests
# ============================================================

class TestKalmanFilter:
    """Tests for the Kalman Filter."""

    def test_run_lgssm(self, lgssm_model, lgssm_data):
        """KF runs on LGSSM and produces finite estimates."""
        from src.filters.kalman_filter import KalmanFilter

        kf = KalmanFilter(lgssm_model)
        obs = tf.constant(lgssm_data['y_obs'], dtype=tf.float32)
        x_init = tf.zeros([1], dtype=tf.float32)
        P_init = tf.eye(1, dtype=tf.float32)

        estimates, covariances = kf.run(obs, x_init, P_init)
        assert estimates.shape[0] == lgssm_data['T']
        assert not tf.reduce_any(tf.math.is_nan(estimates))

    def test_kf_tracks_signal(self, lgssm_model, lgssm_data):
        """KF estimate correlates with true state."""
        from src.filters.kalman_filter import KalmanFilter

        kf = KalmanFilter(lgssm_model)
        obs = tf.constant(lgssm_data['y_obs'], dtype=tf.float32)
        x_init = tf.zeros([1], dtype=tf.float32)
        P_init = tf.eye(1, dtype=tf.float32)

        estimates, _ = kf.run(obs, x_init, P_init)
        est = estimates.numpy().flatten()
        true = lgssm_data['x_true'].flatten()
        corr = np.corrcoef(est, true)[0, 1]
        assert corr > 0.5, f"Correlation too low: {corr:.3f}"


class TestEKF:
    """Tests for the Extended Kalman Filter."""

    def test_ekf_runs_sv(self, sv_model, synthetic_data):
        """EKF runs on SV model without errors."""
        from src.filters.ekf import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter(sv_model)
        obs = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        x_init = tf.zeros([1], dtype=tf.float32)
        P_init = tf.eye(1, dtype=tf.float32)
        Q = tf.eye(1, dtype=tf.float32)
        R = tf.eye(1, dtype=tf.float32)

        estimates = ekf.run(obs, x_init, P_init, Q, R)
        assert estimates.shape[0] == synthetic_data['T']
        assert not tf.reduce_any(tf.math.is_nan(estimates))


class TestUKF:
    """Tests for the Unscented Kalman Filter."""

    def test_ukf_runs_lgssm(self, lgssm_model, lgssm_data):
        """UKF produces finite estimates on LGSSM."""
        from src.filters.ukf import UnscentedKalmanFilter

        ukf = UnscentedKalmanFilter(lgssm_model)
        obs = tf.constant(lgssm_data['y_obs'], dtype=tf.float32)
        x_init = tf.zeros([1], dtype=tf.float32)
        P_init = tf.eye(1, dtype=tf.float32)
        Q = lgssm_model.Q
        R = lgssm_model.R

        estimates = ukf.run(obs, x_init, P_init, Q, R)
        assert estimates.shape[0] == lgssm_data['T']
        assert not tf.reduce_any(tf.math.is_nan(estimates))


class TestParticleFilter:
    """Tests for the Standard Particle Filter (SIR)."""

    def test_pf_runs_sv(self, sv_model, synthetic_data):
        """Particle filter runs on SV model."""
        from src.filters.particle_filter import StandardParticleFilter

        pf = StandardParticleFilter(sv_model, num_particles=200)
        obs = tf.constant(
            synthetic_data['y_obs'].reshape(-1, 1), dtype=tf.float32
        )
        estimates, ess_history = pf.run(obs, verbose=False)
        assert estimates.shape[0] == synthetic_data['T']
        assert not tf.reduce_any(tf.math.is_nan(estimates))


class TestFlowFilters:
    """Tests for EDH/LEDH flow filters."""

    def test_edh_runs(self, sv_model, synthetic_data):
        """EDH flow filter runs on SV model."""
        from src.filters.edh_flow import EDHFlowFilter

        obs = tf.constant(
            synthetic_data['y_obs'].reshape(-1, 1), dtype=tf.float32
        )
        f = EDHFlowFilter(sv_model, num_particles=50, flow_steps=10)
        estimates = f.run(obs)
        assert estimates.shape[0] == synthetic_data['T']
        assert not tf.reduce_any(tf.math.is_nan(estimates))

    def test_ledh_runs(self, sv_model, synthetic_data):
        """LEDH flow filter runs on SV model."""
        from src.filters.ledh_flow import LEDHFlowFilter

        obs = tf.constant(
            synthetic_data['y_obs'].reshape(-1, 1), dtype=tf.float32
        )
        f = LEDHFlowFilter(sv_model, num_particles=50, flow_steps=10)
        estimates = f.run(obs)
        assert estimates.shape[0] == synthetic_data['T']
        assert not tf.reduce_any(tf.math.is_nan(estimates))

    def test_pfpf_edh_runs(self, sv_model, synthetic_data):
        """PFPF-EDH filter runs and returns tuple."""
        from src.filters.pfpf_edh import PFPF_EDH

        obs = tf.constant(
            synthetic_data['y_obs'].reshape(-1, 1), dtype=tf.float32
        )
        f = PFPF_EDH(sv_model, num_particles=50, flow_steps=10)
        result = f.run(obs)
        assert isinstance(result, tuple)
        estimates, avg_ess = result
        assert estimates.shape[0] == synthetic_data['T']
        assert not tf.reduce_any(tf.math.is_nan(estimates))

    def test_pfpf_ledh_runs(self, sv_model, synthetic_data):
        """PFPF-LEDH filter runs and returns tuple."""
        from src.filters.pfpf_ledh import PFPF_LEDH

        obs = tf.constant(
            synthetic_data['y_obs'].reshape(-1, 1), dtype=tf.float32
        )
        f = PFPF_LEDH(sv_model, num_particles=50, flow_steps=10)
        result = f.run(obs)
        assert isinstance(result, tuple)
        estimates, avg_ess = result
        assert estimates.shape[0] == synthetic_data['T']


class TestEnhancedPFPF:
    """Tests for the enhanced PFPF filters with Dai22 homotopy."""

    def test_pfpf_ledh_enhanced_runs(self, sv_model, synthetic_data):
        """Enhanced PFPF-LEDH runs with default linear homotopy."""
        from src.filters.pfpf_enhanced import PFPF_LEDH_Enhanced

        obs = tf.constant(
            synthetic_data['y_obs'].reshape(-1, 1), dtype=tf.float32
        )
        f = PFPF_LEDH_Enhanced(sv_model, num_particles=50, flow_steps=10)
        estimates, metadata = f.run(obs)
        assert estimates.shape[0] == synthetic_data['T']
        assert 'ess_history' in metadata
        assert len(metadata['ess_history']) == synthetic_data['T']

    def test_pfpf_edh_enhanced_runs(self, sv_model, synthetic_data):
        """Enhanced PFPF-EDH runs with default linear homotopy."""
        from src.filters.pfpf_enhanced import PFPF_EDH_Enhanced

        obs = tf.constant(
            synthetic_data['y_obs'].reshape(-1, 1), dtype=tf.float32
        )
        f = PFPF_EDH_Enhanced(sv_model, num_particles=50, flow_steps=10)
        estimates, metadata = f.run(obs)
        assert estimates.shape[0] == synthetic_data['T']


class TestDifferentiablePFPF:
    """Tests for the differentiable PFPF with OT resampling."""

    def test_dpfpf_runs_sv(self, sv_model, synthetic_data):
        """DifferentiablePFPF runs on SV model."""
        from src.filters.differentiable_pfpf import DifferentiablePFPF

        obs = tf.constant(
            synthetic_data['y_obs'][:10].reshape(-1, 1), dtype=tf.float32
        )
        f = DifferentiablePFPF(
            sv_model, num_particles=30, flow_steps=5, ot_iterations=10,
        )
        estimates, log_ml, ess_history = f.run(obs)
        assert estimates.shape[0] == 10
        assert not tf.reduce_any(tf.math.is_nan(estimates))

    def test_sinkhorn_transport(self, sv_model):
        """Sinkhorn transport produces doubly-stochastic matrix."""
        from src.filters.differentiable_pfpf import DifferentiablePFPF

        f = DifferentiablePFPF(sv_model, num_particles=10, ot_iterations=50)
        weights = tf.ones(10, dtype=tf.float32) / 10.0
        cost = tf.random.uniform((10, 10), dtype=tf.float32)
        cost = (cost + tf.transpose(cost)) / 2.0

        T = f.sinkhorn_transport(weights, cost)

        # Rows should sum to weights, columns to 1/N
        row_sums = tf.reduce_sum(T, axis=1)
        col_sums = tf.reduce_sum(T, axis=0)
        np.testing.assert_allclose(row_sums.numpy(), weights.numpy(), atol=0.05)
        np.testing.assert_allclose(col_sums.numpy(), 1.0 / 10.0 * np.ones(10), atol=0.05)


# ============================================================
# Inference Tests
# ============================================================

class TestHMC:
    """Tests for Hamiltonian Monte Carlo."""

    def test_hmc_samples_gaussian(self):
        """HMC samples from a 1-D Gaussian."""
        from src.inference.hmc import HMC

        target_mean, target_var = 3.0, 2.0

        def log_post(q):
            return -0.5 * (q[0] - target_mean) ** 2 / target_var

        def grad_log_post(q):
            return np.array([-(q[0] - target_mean) / target_var])

        hmc = HMC(log_post, grad_log_post, step_size=0.3, num_leapfrog_steps=10)
        result = hmc.sample(np.array([0.0]), num_samples=200, burn_in=100, verbose=False)

        samples = result['samples'].flatten()
        assert abs(np.mean(samples) - target_mean) < 1.0
        assert result['acceptance_rate'] > 0.3


class TestPMMH:
    """Tests for Particle Marginal Metropolis-Hastings."""

    def test_pmmh_accepts_sometimes(self):
        """PMMH has non-zero acceptance rate on trivial problem."""
        from src.inference.pmmh import PMMH, random_walk_proposal
        from functools import partial

        def pf_fn(theta, data):
            # Trivial: log-likelihood = -0.5 * ||theta||^2
            return -0.5 * np.sum(theta ** 2)

        def log_prior(theta):
            return -0.5 * np.sum(theta ** 2) / 10.0

        pmmh = PMMH(
            particle_filter=pf_fn,
            log_prior_fn=log_prior,
            proposal_fn=partial(random_walk_proposal, step_size=0.5),
        )
        result = pmmh.sample(
            np.array([0.0]), data=None, num_samples=50, burn_in=10, verbose=False,
        )
        assert result['acceptance_rate'] > 0.0


class TestComputeESS:
    """Tests for the shared compute_ess utility."""

    def test_ess_iid(self):
        """ESS of i.i.d. samples should be close to N."""
        from src.inference.utils import compute_ess

        samples = np.random.randn(500, 1)
        ess = compute_ess(samples)
        assert ess > 200  # Should be near 500

    def test_ess_correlated(self):
        """ESS of correlated chain should be lower than N."""
        from src.inference.utils import compute_ess

        rng = np.random.RandomState(42)
        chain = [0.0]
        for _ in range(499):
            chain.append(0.99 * chain[-1] + rng.randn() * 0.1)
        samples = np.array(chain).reshape(-1, 1)
        ess = compute_ess(samples)
        assert ess < 250
