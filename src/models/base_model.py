import tensorflow as tf
from abc import ABC, abstractmethod


class StateSpaceModel(ABC):
    """
    Abstract Base Class for State Space Models.

    Defines the standard interface for dynamics, observations, and likelihoods
    used by all filtering algorithms.

    Subclasses must implement:
        - transition(x, noise): State dynamics x_t = f(x_{t-1}) + noise
        - observation(x, noise): Observation y_t = h(x_t) + noise
        - observation_mean(x): Deterministic observation h(x)
        - log_likelihood(y, x): Log p(y | x)
    """

    def __init__(self, state_dim, obs_dim):
        """
        Initialize the state space model.

        Args:
            state_dim (int): Dimension of the state vector.
            obs_dim (int): Dimension of the observation vector.
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim

    @abstractmethod
    def transition(self, x, noise=None):
        """
        State transition: x_t = f(x_{t-1}, noise).

        Args:
            x (tf.Tensor): Previous state of shape (..., state_dim).
            noise (tf.Tensor, optional): Noise of shape (..., state_dim).

        Returns:
            tf.Tensor: Next state of shape (..., state_dim).
        """

    @abstractmethod
    def observation(self, x, noise=None):
        """
        Stochastic observation: y_t = h(x_t) + observation_noise.

        Args:
            x (tf.Tensor): Current state of shape (..., state_dim).
            noise (tf.Tensor, optional): Noise of shape (..., obs_dim).

        Returns:
            tf.Tensor: Observation of shape (..., obs_dim).
        """

    @abstractmethod
    def observation_mean(self, x):
        """
        Deterministic observation function h(x).

        This is the noise-free part of the observation model, used by filters
        for computing Jacobians, innovations, and log-likelihoods.

        Args:
            x (tf.Tensor): Current state of shape (..., state_dim).

        Returns:
            tf.Tensor: Predicted observation of shape (..., obs_dim).
        """

    @abstractmethod
    def log_likelihood(self, y, x):
        """
        Compute log p(y | x).

        Args:
            y (tf.Tensor): Observation.
            x (tf.Tensor): State particles of shape (N, state_dim).

        Returns:
            tf.Tensor: Log-likelihood of shape (N,).
        """

    def transition_log_pdf(self, x_curr, x_prev):
        """
        Compute log p(x_curr | x_prev) under the transition model.

        Required by particle flow filters (PFPF) for prior ratio computation.
        Subclasses that will be used with PFPF must override this method.

        Args:
            x_curr (tf.Tensor): Current state of shape (N, state_dim).
            x_prev (tf.Tensor): Previous state of shape (N, state_dim).

        Returns:
            tf.Tensor: Log transition probability of shape (N,).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement transition_log_pdf(). "
            "This is required by particle flow filters (PFPF)."
        )

    def transition_jacobian(self, x):
        """
        Compute Jacobian df/dx of the transition function via AutoDiff.

        Args:
            x (tf.Tensor): State at which to evaluate.

        Returns:
            tf.Tensor: Jacobian tensor.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            x_next = self.transition(x, noise=tf.zeros_like(x))
        return tape.jacobian(x_next, x)

    def observation_jacobian(self, x):
        """
        Compute Jacobian dh/dx of the observation function via AutoDiff.

        Args:
            x (tf.Tensor): State at which to evaluate.

        Returns:
            tf.Tensor: Jacobian tensor.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.observation_mean(x)
        return tape.jacobian(y, x)
