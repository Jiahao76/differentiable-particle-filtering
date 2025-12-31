"""Pytest configuration and shared fixtures."""
import numpy as np
import tensorflow as tf
import pytest
from src.models.sv_model import StochasticVolatilityModel


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    tf.random.set_seed(42)


@pytest.fixture
def sv_model():
    """Create a StochasticVolatility model instance."""
    return StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)


@pytest.fixture
def synthetic_data():
    """Generate synthetic data from SV model for testing."""
    model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
    T = 50  # Use smaller time horizon for tests
    N = 100  # Number of particles
    
    # Generate true trajectory
    x_true = [0.0]
    y_obs = [model.beta * np.exp(x_true[0]/2) * np.random.normal()]
    
    for t in range(1, T):
        x_t = 0.91 * x_true[-1] + np.random.normal()
        x_true.append(x_t)
        y_t = model.beta * np.exp(x_t/2) * np.random.normal()
        y_obs.append(y_t)
    
    return {
        'x_true': np.array(x_true),
        'y_obs': np.array(y_obs),
        'T': T,
        'N': N,
    }


@pytest.fixture
def particles_and_weights(synthetic_data):
    """Create initial particles and uniform weights for testing."""
    N = synthetic_data['N']
    particles = tf.Variable(
        tf.random.normal([N], mean=0.0, stddev=1.0),
        trainable=False,
        dtype=tf.float32
    )
    weights = tf.Variable(
        tf.ones([N]) / N,
        trainable=False,
        dtype=tf.float32
    )
    return particles, weights
