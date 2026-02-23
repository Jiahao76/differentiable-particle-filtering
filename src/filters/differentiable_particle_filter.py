import tensorflow as tf
from src.models.base_model import StateSpaceModel


class DifferentiableParticleFilter:
    """
    Differentiable Particle Filter with multiple resampling strategies.

    Supported resampling methods:
      - "soft": soft-resampling (mixture with uniform) with weight correction.
        Gradient quality: smooth mixture enables reliable gradient flow.
        Trade-off: mixing parameter alpha controls bias (alpha=1 is standard PF,
        alpha=0 is uniform — no learning signal). Moderate gradient variance.
      - "ot": entropy-regularized optimal transport (Sinkhorn + barycentric projection).
        Gradient quality: deterministic transport gives lowest gradient variance.
        Trade-off: epsilon controls regularization — small epsilon is closer to
        true OT but can cause vanishing gradients in Sinkhorn iterations.
      - "gumbel": relaxed categorical resampling via Gumbel-Softmax.
        Gradient quality: reparameterization trick provides unbiased gradients.
        Trade-off: temperature tau controls discreteness — low tau is accurate
        but has high gradient variance; high tau is smooth but biased.

    Gradient quality criterion for HMC suitability:
      A differentiable PF produces gradients suitable for HMC when:
      (1) Gradient SNR > 1: signal dominates noise (E[grad]^2 / Var[grad] > 1)
      (2) Gradient agreement > 0.5: cosine similarity between analytic and
          finite-difference gradients exceeds 0.5
      (3) Condition number < 1e4: Hessian is not severely ill-conditioned
      Use src.filters.gradient_diagnostics to compute these metrics.

    References:
      - Corenflos et al., 2021 (Differentiable Particle Filtering)
      - Chen et al., 2023 (DPF survey)
    """

    def __init__(
        self,
        model: StateSpaceModel,
        num_particles: int = 100,
        resampling_method: str = "soft",
        resample_threshold: float = 0.5,
        soft_mixture: float = 0.9,
        ot_epsilon: float = 0.5,
        ot_iterations: int = 50,
        ot_normalize_cost: bool = True,
        gumbel_temperature: float = 0.5,
    ):
        self.model = model
        self.N = num_particles
        self.resampling_method = resampling_method
        self.resample_threshold = resample_threshold
        self.soft_mixture = soft_mixture
        self.ot_epsilon = ot_epsilon
        self.ot_iterations = ot_iterations
        self.ot_normalize_cost = ot_normalize_cost
        self.gumbel_temperature = gumbel_temperature

    def initialize_particles(self, initial_dist_std: float = 1.0):
        return tf.random.normal(
            (self.N, self.model.state_dim),
            mean=0.0,
            stddev=initial_dist_std,
            dtype=tf.float32,
        )

    def _normalize_log_weights(self, log_weights: tf.Tensor):
        log_weights_norm = log_weights - tf.reduce_logsumexp(log_weights)
        weights = tf.exp(log_weights_norm)
        return log_weights_norm, weights

    def _soft_resample(self, particles: tf.Tensor, log_weights: tf.Tensor):
        log_weights_norm, weights = self._normalize_log_weights(log_weights)
        uniform = tf.fill((self.N,), 1.0 / float(self.N))
        mix_weights = self.soft_mixture * weights + (1.0 - self.soft_mixture) * uniform
        log_mix = tf.math.log(mix_weights + 1e-20)

        indices = tf.random.categorical(tf.reshape(log_mix, (1, -1)), self.N)
        indices = tf.reshape(indices, (-1,))
        new_particles = tf.gather(particles, indices)

        selected_log_w = tf.gather(log_weights_norm, indices)
        selected_log_mix = tf.gather(log_mix, indices)
        new_log_weights = selected_log_w - selected_log_mix
        return new_particles, new_log_weights

    def _compute_cost_matrix(self, particles: tf.Tensor):
        diff = tf.expand_dims(particles, 1) - tf.expand_dims(particles, 0)
        cost = tf.reduce_sum(tf.square(diff), axis=-1)
        if self.ot_normalize_cost:
            cost = cost / (tf.reduce_mean(cost) + 1e-8)
        return cost

    def _sinkhorn_transport(self, weights: tf.Tensor, cost: tf.Tensor):
        log_a = tf.math.log(weights + 1e-20)
        log_b = tf.math.log(tf.fill((self.N,), 1.0 / float(self.N)))
        log_k = -cost / self.ot_epsilon
        log_u = tf.zeros_like(log_a)
        log_v = tf.zeros_like(log_b)

        for _ in range(self.ot_iterations):
            log_u = log_a - tf.reduce_logsumexp(log_k + tf.reshape(log_v, (1, -1)), axis=1)
            log_v = log_b - tf.reduce_logsumexp(log_k + tf.reshape(log_u, (-1, 1)), axis=0)

        log_p = log_k + tf.reshape(log_u, (-1, 1)) + tf.reshape(log_v, (1, -1))
        return tf.exp(log_p)

    def _ot_resample(self, particles: tf.Tensor, log_weights: tf.Tensor):
        _, weights = self._normalize_log_weights(log_weights)
        cost = self._compute_cost_matrix(particles)
        transport = self._sinkhorn_transport(weights, cost)
        new_particles = tf.matmul(transport, particles, transpose_a=True) * float(self.N)
        new_log_weights = tf.zeros((self.N,), dtype=tf.float32)
        return new_particles, new_log_weights

    def _gumbel_softmax_resample(self, particles: tf.Tensor, log_weights: tf.Tensor):
        log_weights_norm, _ = self._normalize_log_weights(log_weights)
        uniform = tf.random.uniform((self.N, self.N), minval=1e-6, maxval=1.0)
        gumbel = -tf.math.log(-tf.math.log(uniform))
        logits = tf.reshape(log_weights_norm, (1, -1)) + gumbel
        soft_samples = tf.nn.softmax(logits / self.gumbel_temperature, axis=-1)
        new_particles = tf.matmul(soft_samples, particles)
        new_log_weights = tf.zeros((self.N,), dtype=tf.float32)
        return new_particles, new_log_weights

    def _resample(self, particles: tf.Tensor, log_weights: tf.Tensor):
        method = self.resampling_method.lower()
        if method == "soft":
            return self._soft_resample(particles, log_weights)
        if method == "ot":
            return self._ot_resample(particles, log_weights)
        if method == "gumbel":
            return self._gumbel_softmax_resample(particles, log_weights)
        raise ValueError(f"Unknown resampling_method: {self.resampling_method}")

    def run(self, observations: tf.Tensor, verbose: bool = True):
        """
        Run the differentiable particle filter.

        Args:
            observations: Observation sequence [T, obs_dim]
            verbose: Print progress

        Returns:
            estimates: State estimates [T, state_dim]
            ess_history: ESS at each time step [T]
            log_marginal_likelihood: Estimate of log p(y_{1:T})
        """
        T = tf.shape(observations)[0]
        particles = self.initialize_particles(initial_dist_std=1.0)
        log_weights = tf.zeros((self.N,), dtype=tf.float32)

        estimates_list = []
        ess_list = []
        log_marginal_likelihood = 0.0

        if verbose:
            print(f"Starting Differentiable PF ({self.resampling_method}) with {self.N} particles...")

        for t in range(T):
            y_curr = observations[t]

            particles = self.model.transition(particles)

            log_likelihoods = self.model.log_likelihood(y_curr, particles)
            log_likelihoods = tf.reshape(log_likelihoods, (self.N,))
            log_weights = log_weights + log_likelihoods

            # Log marginal likelihood increment: log(1/N * sum_i w_i)
            log_ml_increment = tf.reduce_logsumexp(log_weights) - tf.math.log(
                tf.cast(self.N, tf.float32)
            )
            log_marginal_likelihood += log_ml_increment

            log_weights_norm, weights = self._normalize_log_weights(log_weights)

            estimate = tf.reduce_sum(particles * tf.reshape(weights, (-1, 1)), axis=0)
            estimates_list.append(estimate)

            ess = 1.0 / tf.reduce_sum(tf.square(weights))
            ess_list.append(ess)

            if ess < (self.N * self.resample_threshold):
                particles, log_weights = self._resample(particles, log_weights)

        estimates = tf.stack(estimates_list)
        ess_history = tf.stack(ess_list)
        return estimates, ess_history, log_marginal_likelihood
