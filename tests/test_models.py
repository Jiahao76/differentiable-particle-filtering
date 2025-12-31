"""Unit tests for SV model and filter methods."""
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
            log_lik = sv_model.log_likelihood(particles, observation)
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
        
        # Check ESS is in valid range [1, N]
        assert 1.0 <= ess <= N
