import tensorflow as tf
from abc import ABC, abstractmethod

class StateSpaceModel(ABC):
    """
    Abstract Base Class for State Space Models.
    Defines the interface for dynamics and observations.
    """
    
    def __init__(self, state_dim, obs_dim):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

    @abstractmethod
    def transition(self, x, noise=None):
        """x_t = f(x_{t-1}, noise)"""
        pass

    @abstractmethod
    def observation(self, x, noise=None):
        """y_t = h(x_t, noise)"""
        pass
    
    def transition_jacobian(self, x):
        """Returns df/dx evaluated at x."""
        with tf.GradientTape() as tape:
            tape.watch(x)
            x_next = self.transition(x)
        return tape.gradient(x_next, x)

    def observation_jacobian(self, x):
        """Returns dh/dx evaluated at x."""
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.observation(x)
        return tape.gradient(y, x)