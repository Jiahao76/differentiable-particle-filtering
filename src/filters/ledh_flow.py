"""
Local Exact Daum-Huang (LEDH) Flow Filter
Based on Daum & Huang (2011)

Reference:
[Daum(11)] Daum, Fred, and Jim Huang. 
"Particle degeneracy: root cause and solution." SPIE, 2011.
"""
import tensorflow as tf


class LEDHFlowFilter:
    """Localized Exact Daum-Huang (LEDH) Particle Flow Filter.

    Each particle uses INDIVIDUAL flow parameters computed at its own location.
    Generic: works with any model that provides ``observation_mean()``.
    """

    def __init__(self, model, num_particles=100, flow_steps=20, step_size=0.05,
                 obs_noise_var=1.0):
        """Initialize the LEDH flow filter.

        Args:
            model: State-space model with ``transition`` and ``observation_mean``.
            num_particles: Number of particles.
            flow_steps: Number of discretization steps.
            step_size: Euler step size.
            obs_noise_var: Observation noise variance *R* used in the flow.
        """
        self.model = model
        self.num_particles = num_particles
        self.flow_steps = flow_steps
        self.epsilon = step_size
        self.R = obs_noise_var
    
    def run(self, observations):
        """
        Run LEDH flow filter with vectorized particle updates
        
        CRITICAL INSIGHT from Li(2017):
        LEDH = Local EDH means each particle has:
        1. Local linearization: H_i computed at particle i's position
        2. Local dynamics: Uses ensemble covariance P (shared)
        
        The key difference from EDH:
        - EDH: Linearizes at ensemble mean, all particles move together
        - LEDH: Linearizes at each particle's position, particles move independently
        
        Args:
            observations: Tensor of shape (T, 1)
        
        Returns:
            estimates: Estimated states at each time step
        """
        T = tf.shape(observations)[0]
        state_dim = 1
        
        # Initialize particles
        particles = tf.random.normal((self.num_particles, state_dim), dtype=tf.float32)
        estimates = tf.TensorArray(dtype=tf.float32, size=T, clear_after_read=False)
        
        for t in range(T):
            obs = observations[t]
            
            # ===== PREDICTION =====
            particles = self.model.transition(particles)
            
            # Compute ensemble covariance P (shared across all particles)
            eta_mean = tf.reduce_mean(particles, axis=0, keepdims=True)
            P = tf.reduce_mean((particles - eta_mean)**2)
            
            # Save initial particles for comparison
            particles_initial = tf.identity(particles)
            
            # ===== PARTICLE FLOW =====
            # Integrate from lambda=0 to lambda=1
            lambda_val = 0.0
            
            for j in range(self.flow_steps):
                lambda_val += self.epsilon
                
                # Key: Compute H at CURRENT particle positions (local linearization)
                with tf.GradientTape(persistent=False) as tape:
                    tape.watch(particles)
                    h_val = self.model.observation_mean(particles)

                # H: Jacobian at each particle's current location
                H = tape.gradient(h_val, particles)
                if H is None:
                    H = tf.ones_like(particles)
                
                # LEDH flow parameters using shared P but per-particle H
                denom = lambda_val * (H**2) * P + self.R
                A = -0.5 * P * (H**2) / (denom + 1e-8)
                
                # Compute residual e = h(x) - H*x
                e = h_val - H * particles
                innovation = obs - e
                
                # Flow velocity
                factor1 = 1.0 + 2.0 * lambda_val * A
                factor2 = 1.0 + lambda_val * A
                factor3 = P * H / (self.R + 1e-8)
                b = factor1 * factor2 * factor3 * innovation + A * particles
                
                # Euler step
                particles = particles + self.epsilon * (A * particles + b)
                
                # Numerical stability: clip particles
                particles = tf.clip_by_value(particles, -15.0, 15.0)
            
            # ===== ESTIMATION =====
            estimate = tf.reduce_mean(particles, axis=0)
            estimates = estimates.write(t, estimate)
        
        return estimates.stack()