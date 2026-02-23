"""
Compare differentiable particle filters with different resampling strategies.

Metrics:
  - RMSE (state estimation)
  - Average ESS
  - Runtime
  - Gradient variance w.r.t. model parameter (alpha)
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from models.sv_model import StochasticVolatilityModel
from filters.differentiable_particle_filter import DifferentiableParticleFilter


def simulate_sv(model, T=50, seed=0):
    tf.random.set_seed(seed)
    x = tf.zeros((1, 1), dtype=tf.float32)
    xs = []
    ys = []
    for _ in range(T):
        x = model.transition(x)
        y = model.observation(x)
        xs.append(tf.squeeze(x))
        ys.append(tf.squeeze(y))
    return tf.stack(xs), tf.stack(ys)


def compute_rmse(estimates, truth):
    diff = tf.squeeze(estimates) - tf.squeeze(truth)
    return tf.sqrt(tf.reduce_mean(tf.square(diff)))


def evaluate_method(method, observations, truth, num_particles=100):
    dpf = DifferentiableParticleFilter(
        model=method["model"],
        num_particles=num_particles,
        resampling_method=method["name"],
        soft_mixture=method.get("soft_mixture", 0.9),
        ot_epsilon=method.get("ot_epsilon", 0.5),
        ot_iterations=method.get("ot_iterations", 50),
        gumbel_temperature=method.get("gumbel_temperature", 0.5),
    )

    start = time.perf_counter()
    estimates, ess_history, _ = dpf.run(observations, verbose=False)
    runtime = time.perf_counter() - start

    rmse = compute_rmse(estimates, truth)
    avg_ess = tf.reduce_mean(ess_history)

    return rmse.numpy(), avg_ess.numpy(), runtime


def gradient_variance(method, observations, truth, num_particles=100, n_runs=5, seed=0):
    grads = []
    for i in range(n_runs):
        tf.random.set_seed(seed + i)
        model = method["model"]
        model.alpha = tf.Variable(float(model.alpha.numpy()), dtype=tf.float32)

        dpf = DifferentiableParticleFilter(
            model=model,
            num_particles=num_particles,
            resampling_method=method["name"],
            soft_mixture=method.get("soft_mixture", 0.9),
            ot_epsilon=method.get("ot_epsilon", 0.5),
            ot_iterations=method.get("ot_iterations", 50),
            gumbel_temperature=method.get("gumbel_temperature", 0.5),
        )

        with tf.GradientTape() as tape:
            estimates, _, _ = dpf.run(observations, verbose=False)
            loss = tf.reduce_mean(tf.square(tf.squeeze(estimates) - tf.squeeze(truth)))
        grad = tape.gradient(loss, model.alpha)
        grads.append(float(grad.numpy()))

    return float(np.var(grads))


def main():
    np.random.seed(0)
    tf.random.set_seed(0)

    base_model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
    truth, observations = simulate_sv(base_model, T=50, seed=0)

    methods = [
        {
            "name": "soft",
            "model": StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5),
            "soft_mixture": 0.9,
        },
        {
            "name": "gumbel",
            "model": StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5),
            "gumbel_temperature": 0.7,
        },
        {
            "name": "ot",
            "model": StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5),
            "ot_epsilon": 0.5,
            "ot_iterations": 50,
        },
    ]

    print("Differentiable PF comparison (SV model)")
    print("=" * 60)

    for method in methods:
        rmse, avg_ess, runtime = evaluate_method(method, observations, truth, num_particles=100)
        grad_var = gradient_variance(method, observations, truth, num_particles=80, n_runs=5)
        print(
            f"{method['name']:<8} | RMSE={rmse:.4f} | Avg ESS={avg_ess:.1f} | "
            f"Time={runtime:.3f}s | GradVar={grad_var:.3e}"
        )


if __name__ == "__main__":
    main()
