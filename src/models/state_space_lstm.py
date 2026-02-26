"""
State Space LSTM Models (Zheng, 2017)

Implementation of State-Space LSTM models that combine LSTM transition dynamics
with probabilistic emissions for sequential modeling.

Reference:
    Zheng, Z., et al. (2017). "State Space LSTM Models with Particle MCMC Inference"
    arXiv preprint arXiv:1711.11179
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict
from src.models.base_model import StateSpaceModel


class StateSpaceLSTM(StateSpaceModel):
    """
    Base class for State Space LSTM models.
    
    Key components:
    - LSTM network for transition dynamics: s_t = LSTM(s_{t-1}, z_{t-1})
    - Emission distribution: p(x_t | z_t, s_t)
    - Transition distribution: p(z_t | s_t)
    
    The LSTM hidden state s_t captures long-term dependencies,
    while z_t represents the latent state at each timestep.
    """
    
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        lstm_units: int = 64,
        name: str = "StateSpaceLSTM"
    ):
        """
        Args:
            state_dim: Dimension of latent state z_t
            obs_dim: Dimension of observations x_t
            lstm_units: Number of LSTM hidden units
            name: Model name
        """
        # Note: Don't pass name to parent, it only accepts state_dim and obs_dim
        super().__init__(state_dim=state_dim, obs_dim=obs_dim)
        self.name = name
        self.lstm_units = lstm_units
        
        # Build LSTM cell and networks
        self.lstm_cell = tf.keras.layers.LSTMCell(lstm_units)
        self.build_networks()
        
        # LSTM state (hidden state h and cell state c)
        self.lstm_state = None
        
    def build_networks(self):
        """Build neural networks for transition and emission distributions."""
        raise NotImplementedError("Subclasses must implement build_networks")
    
    def transition(self, z_prev: tf.Tensor, noise: Optional[tf.Tensor] = None):
        """
        Implement abstract method from StateSpaceModel.
        Transition from z_prev to z_next via LSTM.
        
        Args:
            z_prev: Previous latent state [batch, state_dim]
            noise: Optional noise (ignored, uses stochastic sampling)
            
        Returns:
            z_next: Next latent state [batch, state_dim]
        """
        z_next, _ = self.sample_transition(z_prev, training=True)
        return z_next
    
    def observation(self, z: tf.Tensor, noise: Optional[tf.Tensor] = None):
        """
        Implement abstract method from StateSpaceModel.
        Generate observation from latent state z.
        
        Args:
            z: Latent state [batch, state_dim]
            noise: Optional noise (ignored, uses stochastic sampling)
            
        Returns:
            x: Observation [batch, obs_dim]
        """
        lstm_output = self.lstm_state[0]  # Use current LSTM state
        mean, std = self.get_emission_params(lstm_output, z)
        x = mean + std * tf.random.normal(tf.shape(mean)) if noise is None else mean + std * noise
        return x
    
    def reset_lstm_state(self, batch_size: int = 1):
        """Reset LSTM hidden state."""
        self.lstm_state = [
            tf.zeros((batch_size, self.lstm_units)),  # h
            tf.zeros((batch_size, self.lstm_units))   # c
        ]
    
    def lstm_forward(self, z_prev: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        LSTM forward step: s_t = LSTM(s_{t-1}, z_{t-1})
        
        Args:
            z_prev: Previous latent state [batch, state_dim]
            
        Returns:
            lstm_output: LSTM output (hidden state) [batch, lstm_units]
            lstm_state: New LSTM state (h, c)
        """
        if self.lstm_state is None:
            batch_size = tf.shape(z_prev)[0]
            self.reset_lstm_state(batch_size)
        
        # LSTM takes previous z as input
        lstm_output, new_state = self.lstm_cell(z_prev, self.lstm_state)
        self.lstm_state = new_state
        
        return lstm_output, new_state


class GaussianSSL(StateSpaceLSTM):
    """
    Gaussian State Space LSTM (Example 1 from Zheng, 2017).
    
    - Latent state z_t: continuous, Gaussian
    - Observations x_t: continuous, Gaussian
    - Both transition and emission are Gaussian with parameters from LSTM
    
    Transition: z_t ~ N(μ_trans(s_t), Σ_trans(s_t))
    Emission:   x_t ~ N(μ_emis(s_t, z_t), Σ_emis(s_t, z_t))
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        obs_dim: int = 2,
        lstm_units: int = 64,
        min_std: float = 0.01,
        name: str = "GaussianSSL"
    ):
        self.min_std = min_std
        super().__init__(state_dim, obs_dim, lstm_units, name)
    
    def build_networks(self):
        """Build transition and emission networks."""
        # Transition network: LSTM hidden -> z_t parameters
        self.transition_net = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.state_dim * 2)  # mean and log_std
        ], name='transition_net')
        
        # Emission network: (LSTM hidden + z_t) -> x_t parameters
        self.emission_net = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.obs_dim * 2)  # mean and log_std
        ], name='emission_net')
    
    def get_transition_params(self, lstm_output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get transition distribution parameters from LSTM output.
        
        Args:
            lstm_output: LSTM hidden state [batch, lstm_units]
            
        Returns:
            mean: Transition mean [batch, state_dim]
            std: Transition std [batch, state_dim]
        """
        params = self.transition_net(lstm_output)
        mean = params[..., :self.state_dim]
        log_std = params[..., self.state_dim:]
        std = tf.nn.softplus(log_std) + self.min_std
        return mean, std
    
    def get_emission_params(
        self, 
        lstm_output: tf.Tensor, 
        z: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get emission distribution parameters.
        
        Args:
            lstm_output: LSTM hidden state [batch, lstm_units]
            z: Latent state [batch, state_dim]
            
        Returns:
            mean: Emission mean [batch, obs_dim]
            std: Emission std [batch, obs_dim]
        """
        # Concatenate LSTM output and latent state
        inputs = tf.concat([lstm_output, z], axis=-1)
        params = self.emission_net(inputs)
        mean = params[..., :self.obs_dim]
        log_std = params[..., self.obs_dim:]
        std = tf.nn.softplus(log_std) + self.min_std
        return mean, std

    def observation_mean(self, z: tf.Tensor) -> tf.Tensor:
        """Deterministic observation function: E[x_t | z_t].

        Args:
            z: Latent state [batch, state_dim]

        Returns:
            Emission mean [batch, obs_dim]
        """
        if self.lstm_state is None:
            raise RuntimeError("LSTM state not initialized. Call reset_lstm_state() first.")
        lstm_output = self.lstm_state[0]
        mean, _ = self.get_emission_params(lstm_output, z)
        return mean

    def sample_transition(
        self,
        z_prev: tf.Tensor,
        training: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample next latent state: z_t ~ p(z_t | z_{t-1})

        Args:
            z_prev: Previous latent state [batch, state_dim]
            training: Whether in training mode

        Returns:
            z_next: Next latent state [batch, state_dim]
            log_prob: Log probability log p(z_t | z_{t-1})
        """
        lstm_output, _ = self.lstm_forward(z_prev)
        mean, std = self.get_transition_params(lstm_output)
        
        if training:
            z_next = mean + std * tf.random.normal(tf.shape(mean))
        else:
            z_next = mean
        
        # Compute log probability
        log_prob = -0.5 * tf.reduce_sum(
            tf.square((z_next - mean) / std) + 2 * tf.math.log(std) + 
            tf.math.log(2 * np.pi),
            axis=-1
        )
        
        return z_next, log_prob
    
    def log_likelihood(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Compute emission log likelihood: log p(y | x)

        Args:
            y: Observations [batch, obs_dim]
            x: Latent states [batch, state_dim]

        Returns:
            log_lik: Log likelihood [batch]
        """
        # Get LSTM output for current time step
        # Note: This assumes lstm_state is already set by transition
        lstm_output = self.lstm_state[0]  # Use hidden state

        mean, std = self.get_emission_params(lstm_output, x)

        log_lik = -0.5 * tf.reduce_sum(
            tf.square((y - mean) / std) + 2 * tf.math.log(std) +
            tf.math.log(2 * np.pi),
            axis=-1
        )
        
        return log_lik
    
    def sample_trajectory(
        self, 
        T: int, 
        z0: Optional[tf.Tensor] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a full trajectory.
        
        Args:
            T: Sequence length
            z0: Initial latent state [state_dim]
            
        Returns:
            z_seq: Latent states [T, state_dim]
            x_seq: Observations [T, obs_dim]
        """
        if z0 is None:
            z0 = tf.zeros((1, self.state_dim))
        else:
            z0 = tf.reshape(z0, (1, self.state_dim))
        
        # Reset LSTM state
        self.reset_lstm_state(batch_size=1)
        
        z_seq = []
        x_seq = []
        
        z_t = z0
        
        for t in range(T):
            # Sample next latent state
            z_t, _ = self.sample_transition(z_t, training=True)
            z_seq.append(z_t.numpy()[0])
            
            # Sample observation
            lstm_output = self.lstm_state[0]
            mean, std = self.get_emission_params(lstm_output, z_t)
            x_t = mean + std * tf.random.normal(tf.shape(mean))
            x_seq.append(x_t.numpy()[0])
        
        return np.array(z_seq), np.array(x_seq)


class TopicalSSL(StateSpaceLSTM):
    """
    Topical State Space LSTM (Example 2 from Zheng, 2017).
    
    - Latent state z_t: discrete topic indicator (categorical)
    - Observations x_t: word tokens (categorical)
    - Uses Gumbel-Softmax relaxation for differentiability
    
    For DPF-HMC compatibility, we use continuous relaxations:
    - Transition: z_t ~ Gumbel-Softmax(π_trans(s_t), τ)
    - Emission:   x_t ~ Gumbel-Softmax(π_emis(s_t, z_t), τ)
    """
    
    def __init__(
        self,
        num_topics: int = 10,
        vocab_size: int = 100,
        lstm_units: int = 64,
        temperature: float = 0.5,
        min_prob: float = 1e-8,
        name: str = "TopicalSSL"
    ):
        """
        Args:
            num_topics: Number of topics (latent state dimension)
            vocab_size: Vocabulary size (observation dimension)
            lstm_units: Number of LSTM units
            temperature: Gumbel-Softmax temperature (lower = more discrete)
            min_prob: Minimum probability for numerical stability
        """
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.min_prob = min_prob
        
        super().__init__(
            state_dim=num_topics, 
            obs_dim=vocab_size,
            lstm_units=lstm_units,
            name=name
        )
    
    def build_networks(self):
        """Build transition and emission networks."""
        # Transition network: LSTM hidden -> topic distribution
        self.transition_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_topics)  # Logits
        ], name='transition_net')
        
        # Emission network: (LSTM hidden + topic) -> word distribution
        self.emission_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.vocab_size)  # Logits
        ], name='emission_net')
    
    def gumbel_softmax_sample(
        self, 
        logits: tf.Tensor, 
        temperature: float,
        hard: bool = False
    ) -> tf.Tensor:
        """
        Sample from Gumbel-Softmax distribution.
        
        Args:
            logits: Unnormalized log probabilities [batch, dim]
            temperature: Temperature parameter
            hard: If True, return one-hot (straight-through estimator)
            
        Returns:
            samples: Soft samples [batch, dim]
        """
        # Sample Gumbel noise
        gumbel_noise = -tf.math.log(-tf.math.log(
            tf.random.uniform(tf.shape(logits), minval=self.min_prob, maxval=1.0)
        ))
        
        # Add noise and apply softmax with temperature
        y = tf.nn.softmax((logits + gumbel_noise) / temperature)
        
        if hard:
            # Straight-through estimator: hard in forward, soft in backward
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=-1, keepdims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        
        return y

    def observation_mean(self, z: tf.Tensor) -> tf.Tensor:
        """Deterministic observation function: E[x_t | z_t].

        Args:
            z: Latent topic distribution [batch, num_topics]

        Returns:
            Word distribution [batch, vocab_size]
        """
        if self.lstm_state is None:
            raise RuntimeError("LSTM state not initialized. Call reset_lstm_state() first.")
        lstm_output = self.lstm_state[0]
        inputs = tf.concat([lstm_output, z], axis=-1)
        logits = self.emission_net(inputs)
        return tf.nn.softmax(logits)

    def sample_transition(
        self,
        z_prev: tf.Tensor,
        training: bool = True,
        hard: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample next latent topic: z_t ~ p(z_t | z_{t-1})
        
        Args:
            z_prev: Previous topic distribution [batch, num_topics]
            training: Whether in training mode
            hard: Whether to use hard (one-hot) sampling
            
        Returns:
            z_next: Next topic distribution [batch, num_topics]
            log_prob: Log probability
        """
        lstm_output, _ = self.lstm_forward(z_prev)
        logits = self.transition_net(lstm_output)
        
        if training:
            z_next = self.gumbel_softmax_sample(logits, self.temperature, hard)
        else:
            z_next = tf.nn.softmax(logits)
        
        # Compute log probability
        log_probs = tf.nn.log_softmax(logits)
        log_prob = tf.reduce_sum(z_next * log_probs, axis=-1)
        
        return z_next, log_prob
    
    def log_likelihood(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Compute emission log likelihood: log p(y | x)

        Args:
            y: Observations (word one-hot or soft) [batch, vocab_size]
            x: Latent topics (one-hot or soft) [batch, num_topics]

        Returns:
            log_lik: Log likelihood [batch]
        """
        lstm_output = self.lstm_state[0]

        # Concatenate LSTM output and topic
        inputs = tf.concat([lstm_output, x], axis=-1)
        logits = self.emission_net(inputs)

        log_probs = tf.nn.log_softmax(logits)
        log_lik = tf.reduce_sum(y * log_probs, axis=-1)

        return log_lik
    
    def sample_trajectory(
        self,
        T: int,
        z0: Optional[tf.Tensor] = None,
        hard: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a full trajectory.
        
        Args:
            T: Sequence length
            z0: Initial topic distribution [num_topics]
            hard: Whether to use hard (one-hot) sampling
            
        Returns:
            z_seq: Topic sequences [T, num_topics]
            x_seq: Word sequences [T, vocab_size]
        """
        if z0 is None:
            # Start with uniform distribution
            z0 = tf.ones((1, self.num_topics)) / self.num_topics
        else:
            z0 = tf.reshape(z0, (1, self.num_topics))
        
        self.reset_lstm_state(batch_size=1)
        
        z_seq = []
        x_seq = []
        
        z_t = z0
        
        for t in range(T):
            # Sample next topic
            z_t, _ = self.sample_transition(z_t, training=True, hard=hard)
            z_seq.append(z_t.numpy()[0])
            
            # Sample word
            lstm_output = self.lstm_state[0]
            inputs = tf.concat([lstm_output, z_t], axis=-1)
            logits = self.emission_net(inputs)
            x_t = self.gumbel_softmax_sample(logits, self.temperature, hard=hard)
            x_seq.append(x_t.numpy()[0])
        
        return np.array(z_seq), np.array(x_seq)
