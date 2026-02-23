"""Unit tests for SV model and filter methods.

Test design philosophy (per instructor feedback):
  - Tests are derived from mathematical necessary conditions of the models.
  - Unit tests verify individual model properties (densities, moments, gradients).
  - Integration tests verify complete pipeline behavior.
  - Each test documents which mathematical property it validates.
"""
import numpy as np
import tensorflow as tf
import pytest
from src.models.sv_model import StochasticVolatilityModel


class TestSVModel:
    """Test cases for StochasticVolatility model."""
    
    def test_model_initialization(self, sv_model):
        """Test that model initializes with correct parameters."""
        assert sv_model.alpha == 0.91
        assert sv_model.sigma == 1.0
        assert sv_model.beta == 0.5
    
    def test_transition_shape(self, sv_model):
        """Test transition produces correct output shape."""
        x_t = tf.constant([0.0, 1.0, -1.0], dtype=tf.float32)
        x_next = sv_model.transition(x_t)
        assert x_next.shape == x_t.shape
    
    def test_transition_values(self, sv_model):
        """Test transition computation (mean = alpha * x_t)."""
        x_t = tf.constant([1.0], dtype=tf.float32)
        # Test multiple transitions - mean should be close to alpha * x_t
        samples = [sv_model.transition(x_t) for _ in range(100)]
        mean = tf.reduce_mean(tf.stack(samples))
        expected_mean = 0.91 * 1.0
        assert abs(mean.numpy() - expected_mean) < 0.2  # Allow larger variance
    
    def test_observation_shape(self, sv_model):
        """Test observation produces correct output shape."""
        x = tf.constant([0.0, 1.0, -1.0], dtype=tf.float32)
        y = sv_model.observation(x)
        assert y.shape == x.shape
    
    def test_observation_mean_zero(self, sv_model):
        """Test observation is zero-mean at state=0."""
        x = tf.constant([0.0], dtype=tf.float32)
        samples = [sv_model.observation(x) for _ in range(100)]
        mean = tf.reduce_mean(tf.stack(samples))
        assert abs(mean.numpy()) < 0.15  # Should be close to 0
    
    def test_log_likelihood_shape(self, sv_model):
        """Test log_likelihood computes for batch of particles."""
        particles = tf.constant([[0.0], [1.0], [-1.0]], dtype=tf.float32)
        observation = tf.constant([0.5], dtype=tf.float32)
        log_lik = sv_model.log_likelihood(observation, particles)
        # Shape should be [num_particles] or [num_particles, 1]
        assert log_lik.shape[0] == 3
    
    def test_log_likelihood_negative(self, sv_model):
        """Test log_likelihood is mostly negative (log of probability)."""
        particles = tf.constant([[0.0], [1.0], [2.0], [-1.0]], dtype=tf.float32)
        observation = tf.constant([5.0], dtype=tf.float32)  # Far from most particles
        log_lik = sv_model.log_likelihood(observation, particles)
        # Mean log-likelihood should be negative for distant observation
        assert tf.reduce_mean(log_lik) < 0.0
    
    def test_log_likelihood_far_observation_lower(self, sv_model):
        """Test that far observations have lower likelihood."""
        particles = tf.constant([[0.0], [0.0]], dtype=tf.float32)
        obs_close = tf.constant([0.01], dtype=tf.float32)
        obs_far = tf.constant([10.0], dtype=tf.float32)
        
        lik_close = sv_model.log_likelihood(obs_close, particles)[0]
        lik_far = sv_model.log_likelihood(obs_far, particles)[0]
        
        assert lik_close > lik_far
    
    def test_transition_log_pdf_shape(self, sv_model):
        """Test transition_log_pdf has correct shape."""
        x_prev = tf.constant([0.0, 1.0], dtype=tf.float32)
        x_curr = tf.constant([0.1, 0.9], dtype=tf.float32)
        log_pdf = sv_model.transition_log_pdf(x_prev, x_curr)
        assert log_pdf.shape == x_curr.shape
    
    def test_transition_log_pdf_negative(self, sv_model):
        """Test transition_log_pdf is negative."""
        x_prev = tf.constant([0.0], dtype=tf.float32)
        x_curr = tf.constant([0.1], dtype=tf.float32)
        log_pdf = sv_model.transition_log_pdf(x_prev, x_curr)
        assert log_pdf < 0.0
    
    def test_transition_log_pdf_mean_shift(self, sv_model):
        """Test transition_log_pdf higher for values near mean."""
        x_prev = tf.constant([1.0], dtype=tf.float32)
        
        # Expected mean is phi * x_prev = 0.91
        x_mean = tf.constant([0.91], dtype=tf.float32)
        x_far = tf.constant([5.0], dtype=tf.float32)
        
        log_pdf_mean = sv_model.transition_log_pdf(x_prev, x_mean)
        log_pdf_far = sv_model.transition_log_pdf(x_prev, x_far)
        
        assert log_pdf_mean > log_pdf_far
    
    def test_gradients_computable(self, sv_model):
        """Test that gradients can be computed through model functions."""
        particles = tf.Variable([[0.0], [1.0]], dtype=tf.float32)
        observation = tf.constant([0.5], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            log_lik = sv_model.log_likelihood(observation, particles)
            loss = tf.reduce_sum(log_lik)
        
        grads = tape.gradient(loss, particles)
        assert grads is not None
        assert grads.shape == particles.shape
        assert not tf.reduce_all(tf.equal(grads, 0.0))  # Gradients non-zero


class TestFilterShapes:
    """Test output shapes of filter methods."""
    
    def test_edh_flow_complete_run(self, sv_model, synthetic_data):
        """Test EDH flow filter runs on synthetic data."""
        from src.filters.edh_flow import EDHFlowFilter
        
        observations = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        N = synthetic_data['N']
        
        filter_obj = EDHFlowFilter(sv_model, num_particles=N, flow_steps=10)
        estimates = filter_obj.run(observations)
        
        # Check output length
        assert len(estimates) == len(observations)
        # Check no NaN or Inf
        assert not tf.reduce_any(tf.math.is_nan(estimates))
        assert not tf.reduce_any(tf.math.is_inf(estimates))
    
    def test_ledh_flow_complete_run(self, sv_model, synthetic_data):
        """Test LEDH flow filter runs on synthetic data."""
        from src.filters.ledh_flow import LEDHFlowFilter
        
        observations = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        N = synthetic_data['N']
        
        filter_obj = LEDHFlowFilter(sv_model, num_particles=N, flow_steps=10)
        estimates = filter_obj.run(observations)
        
        # Check output length
        assert len(estimates) == len(observations)
        # Check no NaN or Inf
        assert not tf.reduce_any(tf.math.is_nan(estimates))
        assert not tf.reduce_any(tf.math.is_inf(estimates))


class TestNumericalStability:
    """Test numerical stability of computations."""
    
    def test_pfpf_edh_returns_tuple(self, sv_model, synthetic_data):
        """Test that PF-PF (EDH) returns estimates and ESS."""
        from src.filters.pfpf_edh import PFPF_EDH
        
        observations = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        N = synthetic_data['N']
        
        filter_obj = PFPF_EDH(sv_model, num_particles=N, flow_steps=10)
        result = filter_obj.run(observations)
        
        # Should return tuple (estimates, ESS)
        assert isinstance(result, tuple)
        assert len(result) == 2
        estimates, ess = result
        
        # Check estimates length
        assert len(estimates) == len(observations)
        assert not tf.reduce_any(tf.math.is_nan(estimates))


# ============================================================
# Mathematical Property Tests
# ============================================================

class TestLGSSMAnalyticProperties:
    """Verify analytic properties of the Linear Gaussian SSM.

    Mathematical basis:
      - Transition: x_t = F*x_{t-1} + q_t, q_t ~ N(0, Q)
      - Observation: y_t = H*x_t + r_t, r_t ~ N(0, R)
      - Log-likelihood is Gaussian: log N(y; Hx, R)
      - Stationary variance: Var[x_inf] = Q / (1 - F^2) for 1D stable system
    """

    @pytest.mark.unit
    def test_log_likelihood_is_gaussian(self, lgssm_model):
        """Verify log_likelihood matches analytic Gaussian log-density.

        Property: log p(y|x) = -0.5*(y-Hx)^T R^{-1} (y-Hx) - 0.5*log|R| - d/2*log(2pi)
        """
        x = tf.constant([[2.0]], dtype=tf.float32)
        y = tf.constant([1.5], dtype=tf.float32)

        ll = lgssm_model.log_likelihood(y, x)

        # Analytic computation for 1D: H=1, R=0.5
        Hx = 1.0 * 2.0  # H @ x
        residual = 1.5 - Hx
        R = 0.5
        expected = -0.5 * residual**2 / R - 0.5 * np.log(R) - 0.5 * np.log(2 * np.pi)
        np.testing.assert_allclose(ll.numpy()[0], expected, atol=1e-5)

    @pytest.mark.unit
    def test_transition_stationary_variance(self, lgssm_model):
        """Verify the transition converges to stationary variance Q/(1-F^2).

        Property: For stable AR(1) with |F| < 1, the stationary variance is Q/(1-F^2).
        """
        F_val = 0.9
        Q_val = 1.0
        expected_var = Q_val / (1.0 - F_val**2)

        x = tf.constant([[0.0]], dtype=tf.float32)
        samples = []
        for _ in range(5000):
            x = lgssm_model.transition(x)
            samples.append(x.numpy()[0, 0])

        # Use last 3000 samples (after burn-in)
        empirical_var = np.var(samples[2000:])
        np.testing.assert_allclose(empirical_var, expected_var, rtol=0.25)


class TestSVModelAnalyticProperties:
    """Verify analytic properties of the Stochastic Volatility model.

    Mathematical basis:
      - Transition: X_t = alpha*X_{t-1} + sigma*V_t, V_t ~ N(0,1)
      - Observation: Y_t | X_t ~ N(0, beta^2 * exp(X_t))
      - h(x) = beta * exp(x/2) is the observation standard deviation
      - Stationary distribution: X ~ N(0, sigma^2 / (1 - alpha^2))
    """

    @pytest.mark.unit
    def test_observation_variance_matches_state(self, sv_model):
        """Verify Var[Y|X=x] = beta^2 * exp(x).

        Property: At fixed state x, observations are N(0, beta^2*exp(x)).
        """
        x = tf.constant([[1.5]], dtype=tf.float32)
        samples = [sv_model.observation(x).numpy()[0, 0] for _ in range(2000)]
        empirical_var = np.var(samples)
        expected_var = 0.5**2 * np.exp(1.5)  # beta^2 * exp(x)
        np.testing.assert_allclose(empirical_var, expected_var, rtol=0.2)

    @pytest.mark.unit
    def test_log_likelihood_value_at_known_point(self, sv_model):
        """Verify log p(y=0|x=0) = -log(beta) - 0.5*log(2*pi).

        Property: At x=0, Y~N(0, beta^2), so log p(0|0) = -log(beta) - 0.5*log(2pi).
        """
        x = tf.constant([[0.0]], dtype=tf.float32)
        y = tf.constant([0.0], dtype=tf.float32)
        ll = sv_model.log_likelihood(y, x)
        expected = -np.log(0.5) - 0.5 * np.log(2 * np.pi)
        np.testing.assert_allclose(ll.numpy()[0], expected, atol=1e-5)

    @pytest.mark.unit
    def test_stationary_distribution(self, sv_model):
        """Verify the AR(1) transition has stationary variance sigma^2/(1-alpha^2).

        Property: For stable AR(1), the stationary distribution is N(0, sigma^2/(1-alpha^2)).
        """
        alpha_val = 0.91
        sigma_val = 1.0
        expected_var = sigma_val**2 / (1.0 - alpha_val**2)

        x = tf.constant([[0.0]], dtype=tf.float32)
        samples = []
        for _ in range(10000):
            x = sv_model.transition(x)
            samples.append(x.numpy()[0, 0])

        # Use last 7000 samples (after burn-in)
        empirical_mean = np.mean(samples[3000:])
        empirical_var = np.var(samples[3000:])

        assert abs(empirical_mean) < 0.5, f"Stationary mean should be ~0, got {empirical_mean:.3f}"
        np.testing.assert_allclose(empirical_var, expected_var, rtol=0.25)
