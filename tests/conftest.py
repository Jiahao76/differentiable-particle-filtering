"""Pytest configuration and shared fixtures."""
import numpy as np
import tensorflow as tf
import pytest
from src.models.sv_model import StochasticVolatilityModel
from src.models.lgssm import LinearGaussianSSM
from src.models.nonlinear_ssm import NonlinearSSM


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    tf.random.set_seed(42)


@pytest.fixture
def sv_model():
    """Create a StochasticVolatility model instance."""
    return StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)


@pytest.fixture
def lgssm_model():
    """Create a 1-D Linear Gaussian SSM instance."""
    F = tf.constant([[0.9]], dtype=tf.float32)
    H = tf.constant([[1.0]], dtype=tf.float32)
    Q = tf.constant([[1.0]], dtype=tf.float32)
    R = tf.constant([[0.5]], dtype=tf.float32)
    return LinearGaussianSSM(F, H, Q, R)


@pytest.fixture
def nonlinear_model():
    """Create a NonlinearSSM instance (Andrieu 2010)."""
    return NonlinearSSM()


@pytest.fixture
def synthetic_data():
    """Generate synthetic data from SV model for testing."""
    np.random.seed(42)
    model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
    T = 50
    N = 100

    x_true = [0.0]
    y_obs = [float(model.beta) * np.exp(x_true[0] / 2) * np.random.normal()]

    for t in range(1, T):
        x_t = 0.91 * x_true[-1] + np.random.normal()
        x_true.append(x_t)
        y_t = float(model.beta) * np.exp(x_t / 2) * np.random.normal()
        y_obs.append(y_t)

    return {
        'x_true': np.array(x_true),
        'y_obs': np.array(y_obs),
        'T': T,
        'N': N,
    }


@pytest.fixture
def lgssm_data(lgssm_model):
    """Generate synthetic data from LGSSM for testing."""
    np.random.seed(123)
    tf.random.set_seed(123)
    T = 50

    x = tf.constant([[0.0]], dtype=tf.float32)
    x_true = []
    y_obs = []

    for _ in range(T):
        x = lgssm_model.transition(x)
        y = lgssm_model.observation(x)
        x_true.append(x.numpy().flatten())
        y_obs.append(y.numpy().flatten())

    return {
        'x_true': np.array(x_true),
        'y_obs': np.array(y_obs),
        'T': T,
    }


@pytest.fixture
def particles_and_weights(synthetic_data):
    """Create initial particles and uniform weights for testing."""
    N = synthetic_data['N']
    particles = tf.Variable(
        tf.random.normal([N], mean=0.0, stddev=1.0),
        trainable=False,
        dtype=tf.float32,
    )
    weights = tf.Variable(
        tf.ones([N]) / N,
        trainable=False,
        dtype=tf.float32,
    )
    return particles, weights


@pytest.fixture
def lgssm_analytic_results(lgssm_model, lgssm_data):
    """Run Kalman filter to get exact analytic results for LGSSM.

    Provides ground truth for testing particle filter convergence
    against the optimal filter.
    """
    from src.filters.kalman_filter import KalmanFilter

    kf = KalmanFilter(lgssm_model)
    obs = tf.constant(lgssm_data['y_obs'], dtype=tf.float32)
    x_init = tf.zeros([1], dtype=tf.float32)
    P_init = tf.eye(1, dtype=tf.float32)
    estimates, covariances = kf.run(obs, x_init, P_init)
    return {
        'estimates': estimates,
        'covariances': covariances,
        'x_true': lgssm_data['x_true'],
        'y_obs': lgssm_data['y_obs'],
    }
