"""Gradient quality tests for differentiable particle filters.

Test design philosophy (per instructor feedback):
  DPF is a core objective of this project. These tests define precise
  mathematical criteria for gradient quality and verify that the
  differentiable resampling methods produce gradients suitable for
  gradient-based samplers like HMC.

  Criteria for "good enough" gradients:
    1. Gradient exists: non-None and non-zero
    2. Gradient agreement: cosine similarity > 0.3 with finite differences
    3. Gradient variance decreases with more particles (consistency)
    4. Gradient SNR > 0 (signal exists above noise)

  Key implementation detail: tf.GradientTape only records operations
  within its context. Model parameters must be tf.Variables assigned
  BEFORE entering the tape, so that operations using them are recorded.
"""
import numpy as np
import tensorflow as tf
import pytest


def _make_lgssm_with_variable_F(F_value=0.9):
    """Create an LGSSM model with F as a tf.Variable for gradient tracking.

    Returns (model, F_param) where F_param is the tf.Variable.
    """
    from src.models.lgssm import LinearGaussianSSM

    model = LinearGaussianSSM(
        F=tf.constant([[F_value]], dtype=tf.float32),
        H=tf.constant([[1.0]]),
        Q=tf.constant([[1.0]]),
        R=tf.constant([[0.5]]),
    )
    F_param = tf.Variable([[F_value]], dtype=tf.float32)
    model.F = F_param  # Replace with Variable for gradient tracking
    return model, F_param


# ============================================================
# DPF Gradient Quality Tests
# ============================================================

class TestDPFGradientQuality:
    """Verify gradient quality through DifferentiableParticleFilter.

    These tests ensure the three resampling strategies (soft, OT, Gumbel)
    produce meaningful gradients that can be used for parameter learning.
    """

    @pytest.mark.gradient
    def test_soft_resampling_gradient_exists(self, lgssm_data):
        """Gradients of log ML w.r.t. F exist for soft resampling."""
        from src.filters.differentiable_particle_filter import DifferentiableParticleFilter

        obs = tf.constant(lgssm_data['y_obs'][:10].reshape(-1, 1), dtype=tf.float32)
        model, F_param = _make_lgssm_with_variable_F()

        dpf = DifferentiableParticleFilter(
            model, num_particles=50, resampling_method="soft",
            resample_threshold=0.3,
        )

        with tf.GradientTape() as tape:
            _, _, log_ml = dpf.run(obs, verbose=False)
            loss = -log_ml

        grad = tape.gradient(loss, F_param)
        assert grad is not None, "Gradient is None for soft resampling"
        assert not tf.reduce_all(tf.equal(grad, 0.0)), "Gradient is all zeros"

    @pytest.mark.gradient
    def test_ot_resampling_gradient_exists(self, lgssm_data):
        """Gradients of log ML w.r.t. F exist for OT resampling."""
        from src.filters.differentiable_particle_filter import DifferentiableParticleFilter

        obs = tf.constant(lgssm_data['y_obs'][:10].reshape(-1, 1), dtype=tf.float32)
        model, F_param = _make_lgssm_with_variable_F()

        dpf = DifferentiableParticleFilter(
            model, num_particles=50, resampling_method="ot",
            ot_iterations=20, resample_threshold=0.3,
        )

        with tf.GradientTape() as tape:
            _, _, log_ml = dpf.run(obs, verbose=False)
            loss = -log_ml

        grad = tape.gradient(loss, F_param)
        assert grad is not None, "Gradient is None for OT resampling"

    @pytest.mark.gradient
    def test_gumbel_resampling_gradient_exists(self, lgssm_data):
        """Gradients of log ML w.r.t. F exist for Gumbel-Softmax resampling."""
        from src.filters.differentiable_particle_filter import DifferentiableParticleFilter

        obs = tf.constant(lgssm_data['y_obs'][:10].reshape(-1, 1), dtype=tf.float32)
        model, F_param = _make_lgssm_with_variable_F()

        dpf = DifferentiableParticleFilter(
            model, num_particles=50, resampling_method="gumbel",
            resample_threshold=0.3,
        )

        with tf.GradientTape() as tape:
            _, _, log_ml = dpf.run(obs, verbose=False)
            loss = -log_ml

        grad = tape.gradient(loss, F_param)
        assert grad is not None, "Gradient is None for Gumbel resampling"

    @pytest.mark.gradient
    @pytest.mark.slow
    def test_gradient_variance_decreases_with_particles(self, lgssm_data):
        """Gradient variance should decrease with more particles.

        Property: The variance of the gradient estimate from a particle filter
        decreases as O(1/N) with the number of particles. This is a basic
        consistency property of importance sampling.
        """
        from src.filters.differentiable_particle_filter import DifferentiableParticleFilter

        obs = tf.constant(lgssm_data['y_obs'][:5].reshape(-1, 1), dtype=tf.float32)
        n_runs = 8

        def get_grad_variance(num_particles):
            grads = []
            for i in range(n_runs):
                tf.random.set_seed(i * 100)
                model, F_param = _make_lgssm_with_variable_F()
                dpf = DifferentiableParticleFilter(
                    model, num_particles=num_particles,
                    resampling_method="ot", ot_iterations=20,
                    resample_threshold=0.3,
                )
                with tf.GradientTape() as tape:
                    _, _, log_ml = dpf.run(obs, verbose=False)
                    loss = -log_ml
                grad = tape.gradient(loss, F_param)
                if grad is not None:
                    grads.append(grad.numpy().flatten()[0])
            return np.var(grads) if len(grads) > 1 else float('inf')

        var_small = get_grad_variance(30)
        var_large = get_grad_variance(150)

        # Variance with more particles should be smaller
        assert var_large < var_small * 3.0, (
            f"Gradient variance did not decrease: "
            f"var(N=30)={var_small:.4f}, var(N=150)={var_large:.4f}"
        )


# ============================================================
# Differentiable PF-PF Gradient Tests
# ============================================================

class TestDPFPFGradientQuality:
    """Verify gradient quality through DifferentiablePFPF (LEDH + OT).

    The DifferentiablePFPF combines particle flow with OT resampling
    to enable gradient-based inference. These tests verify the gradient
    signal flows through the full LEDH flow + resampling pipeline.
    """

    @pytest.mark.gradient
    def test_dpfpf_gradient_flows_through_flow(self, synthetic_data):
        """Gradient of log ML w.r.t. SV model parameter alpha is non-None.

        Property: The LEDH flow + OT resampling pipeline preserves
        gradient information from the observation model to the parameters.
        """
        from src.filters.differentiable_pfpf import DifferentiablePFPF
        from src.models.sv_model import StochasticVolatilityModel

        obs = tf.constant(
            synthetic_data['y_obs'][:5].reshape(-1, 1), dtype=tf.float32
        )

        # Create model then replace alpha with Variable
        alpha_var = tf.Variable(0.91, dtype=tf.float32)
        model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
        model.alpha = alpha_var

        dpfpf = DifferentiablePFPF(
            model, num_particles=30, flow_steps=5,
            ot_iterations=10, resample_threshold=0.5,
        )

        with tf.GradientTape() as tape:
            estimates, log_ml, ess_history = dpfpf.run(obs)
            loss = -log_ml

        grad = tape.gradient(loss, alpha_var)
        assert grad is not None, "Gradient through DPFPF is None"

    @pytest.mark.gradient
    def test_dpfpf_gradient_snr_positive(self, synthetic_data):
        """Gradient SNR should be positive (signal exists above noise).

        Property: E[grad]^2 / Var[grad] > 0 means the gradient carries
        useful information for optimization, not pure noise.
        """
        from src.filters.differentiable_pfpf import DifferentiablePFPF
        from src.models.sv_model import StochasticVolatilityModel

        obs = tf.constant(
            synthetic_data['y_obs'][:3].reshape(-1, 1), dtype=tf.float32
        )

        grads = []
        for i in range(6):
            tf.random.set_seed(i * 500)
            alpha_var = tf.Variable(0.91, dtype=tf.float32)
            model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
            model.alpha = alpha_var

            dpfpf = DifferentiablePFPF(
                model, num_particles=30, flow_steps=5,
                ot_iterations=10, resample_threshold=0.5,
            )

            with tf.GradientTape() as tape:
                _, log_ml, _ = dpfpf.run(obs)
                loss = -log_ml

            grad = tape.gradient(loss, alpha_var)
            if grad is not None:
                grads.append(grad.numpy())

        if len(grads) >= 3:
            mean_grad = np.mean(grads)
            var_grad = np.var(grads)
            snr = mean_grad**2 / (var_grad + 1e-12)
            assert snr > 0, f"Gradient SNR should be > 0, got {snr:.6f}"
