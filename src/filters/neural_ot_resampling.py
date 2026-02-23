"""
Neural Network-Based Optimal Transport Resampling

Implements two approaches:
1. mGradNet (Chaudhari et al. 2025): Monotone Gradient Networks for OT maps
2. FNO (Jha 2025): Fourier Neural Operator for OT plans

References:
- Chaudhari et al. (2025): "GradNetOT: Learning Optimal Transport Maps with GradNets"
- Jha (2025): "Neural Operators in Scientific Computing"
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Optional


class OTResamplingNetwork(tf.keras.Model):
    """
    Monotone Gradient Network (mGradNet) for optimal transport resampling.
    
    Learns the optimal transport map T: R^d -> R^d such that T_#μ = ν,
    where T(x) = ∇φ(x) for a convex potential φ.
    
    Key features:
    - Conditional on model parameters θ
    - Takes particle statistics as input
    - Generalizes across different SSM instances
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        name: str = "ot_resampling_network"
    ):
        super().__init__(name=name)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature encoder: processes global context
        self.feature_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', name='encoder_1'),
            tf.keras.layers.LayerNormalization(name='encoder_ln'),
            tf.keras.layers.Dense(hidden_dim, activation='relu', name='encoder_2'),
        ], name='feature_encoder')
        
        # mGradNet layers: enforce monotonicity via positive weights
        self.gradient_layers = []
        for i in range(num_layers):
            # Use softplus activation to ensure positive weights (monotonicity)
            layer = tf.keras.layers.Dense(
                hidden_dim,
                activation='softplus',
                kernel_constraint=tf.keras.constraints.NonNeg(),  # Extra safety
                name=f'grad_layer_{i}'
            )
            self.gradient_layers.append(layer)
        
        # Output layer: gradient of convex potential
        self.output_layer = tf.keras.layers.Dense(
            state_dim,
            name='output_layer'
        )
        
        # Temperature for soft assignment
        self.temperature = tf.Variable(1.0, trainable=True, name='temperature')
        
    def _construct_features(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        model_params: tf.Tensor,
        observation: tf.Tensor,
        statistics: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """
        Construct comprehensive input feature vector.
        
        Args:
            particles: (N, d) particle positions
            weights: (N,) normalized weights
            model_params: (p,) model parameter vector
            observation: (d_obs,) current observation
            statistics: dict with 'mean', 'cov', 'ess', 'weight_entropy', 'innovation'
        
        Returns:
            features: (feature_dim,) concatenated feature vector
        """
        # Flatten covariance if multidimensional
        cov_flat = tf.reshape(statistics['cov'], [-1])
        
        # Concatenate all features
        features = tf.concat([
            statistics['mean'],                          # State mean
            cov_flat,                                    # Covariance (flattened)
            [statistics['ess']],                         # Effective sample size
            [statistics['weight_entropy']],              # Weight entropy
            model_params,                                # Model parameters
            observation,                                 # Current observation
            statistics['innovation'],                    # Innovation
        ], axis=0)
        
        return features
    
    def call(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        model_params: tf.Tensor,
        observation: tf.Tensor,
        statistics: Dict[str, tf.Tensor],
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass: predict optimal transport plan.
        
        Args:
            particles: (N, d) particle positions
            weights: (N,) particle weights (normalized)
            model_params: (p,) model parameter vector
            observation: (d_obs,) current observation
            statistics: dict with 'mean', 'cov', 'ess', etc.
            training: whether in training mode
        
        Returns:
            transport_plan: (N, N) soft assignment matrix
        """
        N, d = tf.shape(particles)[0], tf.shape(particles)[1]
        
        # Construct input features (shared across all particles)
        features = self._construct_features(
            particles, weights, model_params, observation, statistics
        )
        
        # Encode global features
        global_features = self.feature_encoder(features, training=training)  # (hidden_dim,)
        
        # Concatenate particle position with global features
        particle_features = tf.concat([
            particles,  # (N, d)
            tf.tile(tf.expand_dims(global_features, 0), [N, 1])  # (N, hidden_dim)
        ], axis=-1)  # (N, d + hidden_dim)
        
        # Pass through mGradNet layers (enforce monotonicity)
        h = particle_features
        for layer in self.gradient_layers:
            h = layer(h)
            # Add residual connection for stability
            if h.shape[-1] == particle_features.shape[-1]:
                h = h + particle_features
        
        # Output: gradient of convex potential (transport map)
        transport_map = self.output_layer(h)  # (N, d)
        
        # Convert transport map to assignment matrix
        # Compute distances between mapped particles and original particles
        diff = tf.expand_dims(transport_map, 1) - tf.expand_dims(particles, 0)  # (N, N, d)
        distances = tf.reduce_sum(tf.square(diff), axis=-1)  # (N, N)
        
        # Soft assignment via softmax
        transport_plan = tf.nn.softmax(-distances / self.temperature, axis=-1)  # (N, N)
        
        return transport_plan
    
    def get_transport_map(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        model_params: tf.Tensor,
        observation: tf.Tensor,
        statistics: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """
        Get the transport map T(x) directly (for analysis/visualization).
        """
        N, d = tf.shape(particles)[0], tf.shape(particles)[1]
        features = self._construct_features(
            particles, weights, model_params, observation, statistics
        )
        global_features = self.feature_encoder(features)
        particle_features = tf.concat([
            particles,
            tf.tile(tf.expand_dims(global_features, 0), [N, 1])
        ], axis=-1)
        
        h = particle_features
        for layer in self.gradient_layers:
            h = layer(h)
        
        transport_map = self.output_layer(h)
        return transport_map


class SpectralConv2D(tf.keras.layers.Layer):
    """
    Spectral convolution layer for Fourier Neural Operator.
    Operates in frequency domain for efficient global convolution.
    """
    
    def __init__(self, modes: int, width: int, name: str = 'spectral_conv'):
        super().__init__(name=name)
        self.modes = modes
        self.width = width
        
        # Learnable Fourier weights (complex-valued)
        scale = 1.0 / (width * width)
        self.weights_real = self.add_weight(
            shape=(modes, modes, width, width),
            initializer=tf.keras.initializers.RandomNormal(stddev=scale),
            trainable=True,
            name='weights_real'
        )
        self.weights_imag = self.add_weight(
            shape=(modes, modes, width, width),
            initializer=tf.keras.initializers.RandomNormal(stddev=scale),
            trainable=True,
            name='weights_imag'
        )
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Args:
            x: (N, N, width) input tensor
        
        Returns:
            out: (N, N, width) output after spectral convolution
        """
        # FFT along both spatial dimensions
        x_complex = tf.cast(x, tf.complex64)
        x_ft = tf.signal.fft2d(x_complex)  # (N, N, width)
        
        # Extract low-frequency modes
        x_ft_low = x_ft[:self.modes, :self.modes, :]  # (modes, modes, width)
        
        # Combine real and imaginary weights
        weights_complex = tf.complex(self.weights_real, self.weights_imag)  # (modes, modes, width, width)
        
        # Spectral convolution: multiply in frequency domain
        # out_ft[i,j,m] = sum_n (x_ft[i,j,n] * weights[i,j,n,m])
        out_ft_low = tf.einsum('ijk,ijkm->ijm', x_ft_low, weights_complex)
        
        # Pad back to full size
        N = tf.shape(x)[0]
        out_ft = tf.zeros((N, N, self.width), dtype=tf.complex64)
        out_ft = tf.tensor_scatter_nd_update(
            out_ft,
            indices=[[i, j] for i in range(self.modes) for j in range(self.modes)],
            updates=tf.reshape(out_ft_low, [-1, self.width])
        )
        
        # Inverse FFT
        out_complex = tf.signal.ifft2d(out_ft)
        out_real = tf.math.real(out_complex)
        
        return out_real


class FourierOTOperator(tf.keras.Model):
    """
    Fourier Neural Operator for optimal transport resampling.
    
    Learns the solution operator: (cost, weights, ε) ↦ transport plan
    
    Key advantages:
    - Discretization invariant: train on N=100, test on N=200
    - Fast inference: O(N log N) via FFT
    - Captures global structure in frequency domain
    """
    
    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        num_layers: int = 4,
        name: str = "fourier_ot_operator"
    ):
        super().__init__(name=name)
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        
        # Lifting layer: embed input to higher dimension
        self.lift = tf.keras.layers.Dense(width, name='lift')
        
        # Fourier layers
        self.fourier_layers = []
        self.conv_layers = []
        for i in range(num_layers):
            self.fourier_layers.append(SpectralConv2D(modes, width, name=f'spectral_{i}'))
            self.conv_layers.append(tf.keras.layers.Dense(width, activation='relu', name=f'conv_{i}'))
        
        # Projection layers: map back to transport plan
        self.project = tf.keras.Sequential([
            tf.keras.layers.Dense(width, activation='relu', name='project_1'),
            tf.keras.layers.Dense(width, activation='relu', name='project_2'),
            tf.keras.layers.Dense(1, name='project_out')  # Output: transport plan values
        ], name='projection')
        
    def call(
        self,
        cost_matrix: tf.Tensor,
        source_weights: tf.Tensor,
        epsilon: float,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass: predict optimal transport plan.
        
        Args:
            cost_matrix: (N, N) cost matrix
            source_weights: (N,) source distribution
            epsilon: scalar regularization parameter
            training: whether in training mode
        
        Returns:
            transport_plan: (N, N) optimal transport plan
        """
        N = tf.shape(cost_matrix)[0]
        
        # Construct input features at each grid point
        # Stack: [cost, source_weights_i, source_weights_j, epsilon]
        weights_i = tf.tile(tf.reshape(source_weights, [N, 1]), [1, N])  # (N, N)
        weights_j = tf.tile(tf.reshape(source_weights, [1, N]), [N, 1])  # (N, N)
        epsilon_grid = tf.fill([N, N], epsilon)
        
        x = tf.stack([cost_matrix, weights_i, weights_j, epsilon_grid], axis=-1)  # (N, N, 4)
        
        # Lift to higher dimension
        x = self.lift(x)  # (N, N, width)
        
        # Fourier layers with residual connections
        for fourier_layer, conv_layer in zip(self.fourier_layers, self.conv_layers):
            x_fourier = fourier_layer(x)  # Spectral convolution
            x_conv = conv_layer(x)        # Local convolution
            x = x_fourier + x_conv + x    # Residual connection
        
        # Project to transport plan
        transport_plan = self.project(x, training=training)  # (N, N, 1)
        transport_plan = tf.squeeze(transport_plan, axis=-1)  # (N, N)
        
        # Apply softmax to satisfy marginal constraints (approximately)
        # Row normalization: sum over j gives source weights
        transport_plan = tf.nn.softmax(transport_plan, axis=-1)
        # Column normalization: sum over i gives uniform target
        transport_plan = tf.nn.softmax(transport_plan, axis=-2)
        
        return transport_plan


class DeepONetOT(tf.keras.Model):
    """
    Deep Operator Network for optimal transport.
    
    Learns the operator mapping from (cost, weights) to transport plan
    via separate branch and trunk networks.
    """
    
    def __init__(
        self,
        branch_depth: int = 3,
        trunk_depth: int = 3,
        hidden_dim: int = 128,
        name: str = "deeponet_ot"
    ):
        super().__init__(name=name)
        
        # Branch network: encodes input functions (cost, weights)
        self.branch_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', name=f'branch_{i}')
            for i in range(branch_depth)
        ] + [tf.keras.layers.Dense(hidden_dim, name='branch_out')], name='branch')
        
        # Trunk network: encodes query locations
        self.trunk_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', name=f'trunk_{i}')
            for i in range(trunk_depth)
        ] + [tf.keras.layers.Dense(hidden_dim, name='trunk_out')], name='trunk')
        
        # Bias term
        self.bias = self.add_weight(shape=(), initializer='zeros', name='bias')
        
    def call(
        self,
        cost_matrix: tf.Tensor,
        source_weights: tf.Tensor,
        query_points: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            cost_matrix: (N, N) cost matrix
            source_weights: (N,) source distribution
            query_points: (M, 2) query locations (i, j) for P[i,j]
                         If None, computes full (N, N) matrix
            training: whether in training mode
        
        Returns:
            transport_values: (M,) or (N, N) transport plan values
        """
        N = tf.shape(cost_matrix)[0]
        
        # Branch: encode the input functions
        branch_input = tf.concat([
            tf.reshape(cost_matrix, [-1]),
            source_weights
        ], axis=0)
        branch_out = self.branch_net(branch_input, training=training)  # (hidden_dim,)
        
        # If no query points specified, compute full matrix
        if query_points is None:
            # Create query points for all (i,j) pairs
            i_indices = tf.range(N, dtype=tf.float32)
            j_indices = tf.range(N, dtype=tf.float32)
            ii, jj = tf.meshgrid(i_indices, j_indices, indexing='ij')
            query_points = tf.stack([tf.reshape(ii, [-1]), tf.reshape(jj, [-1])], axis=-1)
        
        # Trunk: encode query locations
        trunk_out = self.trunk_net(query_points, training=training)  # (M, hidden_dim)
        
        # Combine via inner product
        transport_values = tf.reduce_sum(
            trunk_out * tf.expand_dims(branch_out, 0), axis=-1
        ) + self.bias
        
        # If computing full matrix, reshape
        if query_points.shape[0] == N * N:
            transport_values = tf.reshape(transport_values, [N, N])
            # Apply softmax for marginal constraints
            transport_values = tf.nn.softmax(transport_values, axis=-1)
        
        return transport_values


def compute_statistics(particles: tf.Tensor, weights: tf.Tensor) -> Dict[str, tf.Tensor]:
    """
    Compute particle statistics for neural network input.
    
    Args:
        particles: (N, d) particle positions
        weights: (N,) normalized weights
    
    Returns:
        statistics: dict with mean, cov, ess, weight_entropy
    """
    N, d = particles.shape
    
    # Weighted mean
    mean = tf.reduce_sum(weights[:, None] * particles, axis=0)
    
    # Weighted covariance
    centered = particles - mean
    if d == 1:
        cov = tf.reduce_sum(weights * tf.square(centered[:, 0]))
        cov = tf.reshape(cov, [1, 1])
    else:
        cov = tf.reduce_sum(
            weights[:, None, None] * (centered[:, :, None] * centered[:, None, :]),
            axis=0
        )
    
    # Effective sample size
    ess = 1.0 / tf.reduce_sum(tf.square(weights))
    
    # Weight entropy
    weight_entropy = -tf.reduce_sum(weights * tf.math.log(weights + 1e-10))
    
    return {
        'mean': mean,
        'cov': cov,
        'ess': ess,
        'weight_entropy': weight_entropy
    }


def neural_ot_resample(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    ot_network: tf.keras.Model,
    model_params: tf.Tensor,
    observation: tf.Tensor,
    innovation: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Fast neural network-based OT resampling.
    
    Args:
        particles: (N, d) particle positions
        log_weights: (N,) log weights
        ot_network: trained OTResamplingNetwork or FourierOTOperator
        model_params: (p,) model parameter vector
        observation: (d_obs,) current observation
        innovation: (d_obs,) innovation (obs - predicted_obs)
    
    Returns:
        resampled_particles: (N, d) resampled particles
        new_log_weights: (N,) uniform log weights
    """
    N = particles.shape[0]
    
    # Normalize weights
    weights = tf.nn.softmax(log_weights, axis=0)
    
    # Compute statistics
    stats = compute_statistics(particles, weights)
    stats['innovation'] = innovation
    
    # Neural network prediction
    if isinstance(ot_network, (OTResamplingNetwork, DeepONetOT)):
        transport_plan = ot_network(
            particles, weights, model_params, observation, stats
        )
    else:  # FourierOTOperator
        # Compute cost matrix
        diff = tf.expand_dims(particles, 1) - tf.expand_dims(particles, 0)
        cost = tf.reduce_sum(tf.square(diff), axis=-1)
        transport_plan = ot_network(cost, weights, epsilon=0.1)
    
    # Apply transport: barycentric projection
    resampled_particles = tf.matmul(transport_plan, particles, transpose_a=True) * tf.cast(N, tf.float32)
    
    # Uniform weights after resampling
    new_log_weights = tf.zeros(N, dtype=tf.float32)
    
    return resampled_particles, new_log_weights
