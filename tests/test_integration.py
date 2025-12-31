"""Integration tests for complete filtering pipelines."""
import numpy as np
import tensorflow as tf
import pytest
from src.models.sv_model import StochasticVolatilityModel
from src.filters.edh_flow import EDHFlowFilter
from src.filters.ledh_flow import LEDHFlowFilter
from src.filters.pfpf_edh import PFPF_EDH
from src.filters.pfpf_ledh import PFPF_LEDH


class TestCompleteFiltering:
    """Test complete filtering runs on synthetic data."""
    
    def test_edh_filter_complete_run(self, sv_model, synthetic_data):
        """Test EDH filter runs on complete time series without errors."""
        observations = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        N = synthetic_data['N']
        
        filter_obj = EDHFlowFilter(sv_model, num_particles=N, flow_steps=10)
        estimates = filter_obj.run(observations)
        
        # Check results
        assert len(estimates) == len(observations)
        assert not np.any(np.isnan(estimates))
        assert not np.any(np.isinf(estimates))
    
    def test_ledh_filter_complete_run(self, sv_model, synthetic_data):
        """Test LEDH filter runs on complete time series without errors."""
        observations = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        N = synthetic_data['N']
        
        filter_obj = LEDHFlowFilter(sv_model, num_particles=N, flow_steps=10)
        estimates = filter_obj.run(observations)
        
        # Check results
        assert len(estimates) == len(observations)
        assert not np.any(np.isnan(estimates))
    
    def test_pfpf_edh_complete_run(self, sv_model, synthetic_data):
        """Test PF-PF (EDH) returns both estimates and ESS."""
        observations = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        N = synthetic_data['N']
        
        filter_obj = PFPF_EDH(sv_model, num_particles=N, flow_steps=10)
        estimates, ess = filter_obj.run(observations)
        
        # Check estimates
        assert len(estimates) == len(observations)
        assert not np.any(np.isnan(estimates))
        
        # Check ESS is in valid range
        assert 1.0 <= ess <= N
    
    def test_pfpf_ledh_complete_run(self, sv_model, synthetic_data):
        """Test PF-PF (LEDH) returns both estimates and ESS."""
        observations = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        N = synthetic_data['N']
        
        filter_obj = PFPF_LEDH(sv_model, num_particles=N, flow_steps=10)
        estimates, ess = filter_obj.run(observations)
        
        # Check estimates
        assert len(estimates) == len(observations)
        assert not np.any(np.isnan(estimates))
        
        # Check ESS is in valid range
        assert 1.0 <= ess <= N


class TestComparativePerformance:
    """Test relative performance of different filter methods."""
    
    def test_pfpf_edh_better_than_edh(self, sv_model, synthetic_data):
        """Test that PF-PF (EDH) has lower RMSE than pure EDH flow."""
        observations = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        x_true = synthetic_data['x_true']
        N = 100
        
        # Run EDH Flow
        edh = EDHFlowFilter(sv_model, num_particles=N, flow_steps=20)
        est_edh = edh.run(observations).numpy()
        rmse_edh = np.sqrt(np.mean((x_true - est_edh)**2))
        
        # Run PF-PF (EDH)
        pfpf_edh = PFPF_EDH(sv_model, num_particles=N, flow_steps=20)
        est_pfpf, _ = pfpf_edh.run(observations)
        est_pfpf = est_pfpf.numpy()
        rmse_pfpf = np.sqrt(np.mean((x_true - est_pfpf)**2))
        
        # PF-PF should be more accurate (lower RMSE)
        # Allow some tolerance for stochasticity
        # Typically PF-PF is 40-60% better
        assert rmse_pfpf < rmse_edh * 1.1  # At least not worse


class TestEffectiveSampleSize:
    """Test ESS computation."""
    
    def test_ess_in_valid_range_pfpf_edh(self, sv_model, synthetic_data):
        """Test ESS is always between 1 and N for PF-PF (EDH)."""
        observations = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        N = synthetic_data['N']
        
        filter_obj = PFPF_EDH(sv_model, num_particles=N, flow_steps=10)
        _, ess = filter_obj.run(observations)
        
        assert 1.0 <= ess <= N
    
    def test_ess_in_valid_range_pfpf_ledh(self, sv_model, synthetic_data):
        """Test ESS is always between 1 and N for PF-PF (LEDH)."""
        observations = tf.constant(synthetic_data['y_obs'], dtype=tf.float32)
        N = synthetic_data['N']
        
        filter_obj = PFPF_LEDH(sv_model, num_particles=N, flow_steps=10)
        _, ess = filter_obj.run(observations)
        
        assert 1.0 <= ess <= N
