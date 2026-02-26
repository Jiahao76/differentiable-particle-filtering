"""
Stochastic Particle Flow Filter implementation with TensorFlow.
Optimized for numerical stability and GPU execution.
"""

import tensorflow as tf


class TFStochasticParticleFlowFilter:
    """
    Parameterized Stochastic Particle Flow Filter (Dai21/Dai22).
    
    Flow equation:
        dx = f(x,λ)dλ + q(x,λ)dw_λ
    
    where f is the drift computed from:
        f = K_1 * ∇log p + K_2 * ∇log h
    """
    
    def __init__(self, n_dim, q_matrix, mu=0.2):
        """
        Args:
            n_dim: State dimension
            q_matrix: Diffusion matrix Q (stabilizer)
            mu: Weight for stiffness mitigation
        """
        self.n_dim = n_dim
        self.Q = tf.cast(q_matrix, dtype=tf.float32)
        self.mu = mu
        
        # Pre-compute Cholesky decomposition for efficiency
        # q = chol(Q), so Q = qq^T
        try:
            self.q_chol = tf.linalg.cholesky(self.Q + 1e-6 * tf.eye(self.n_dim))
        except tf.errors.InvalidArgumentError:
            # Fallback if Cholesky fails (non-positive-definite Q)
            self.q_chol = tf.linalg.sqrtm(self.Q + 1e-6 * tf.eye(self.n_dim))
    
    @tf.function(reduce_retracing=True)
    def step_logic(self, x, z, model, alpha, beta, alpha_dot, beta_dot, dt):
        """
        Single Euler-Maruyama step of the particle flow SDE.
        
        Computes drift f(x,λ) and diffusion term for one time step.
        
        Args:
            x: Particles [n_particles, n_dim]
            z: Measurement
            model: State-space model
            alpha, beta: Homotopy parameters at current λ
            alpha_dot, beta_dot: Derivatives dα/dλ, dβ/dλ
            dt: Time step Δλ
            
        Returns:
            x_next: Updated particles
        """
        # Ensure numerical types
        alpha = tf.cast(alpha, tf.float32)
        beta = tf.cast(beta, tf.float32)
        alpha_dot = tf.cast(alpha_dot, tf.float32)
        beta_dot = tf.cast(beta_dot, tf.float32)
        
        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch(x)
            
            with tf.GradientTape(persistent=True) as inner_tape:
                inner_tape.watch(x)
                
                # Log-densities
                log_p0 = model.log_prior(x)
                log_h = model.log_likelihood(z, x)
                
                # Homotopy log-density: log p = (α+β)*log p_0 + β*log h
                # Note: α + β = 1 for normalized homotopy
                log_p = (alpha + beta) * log_p0 + beta * log_h
            
            # First-order gradients
            grad_p0 = inner_tape.gradient(log_p0, x)
            grad_h = inner_tape.gradient(log_h, x)
            grad_p = inner_tape.gradient(log_p, x)
            
            # Check for None gradients
            if grad_p0 is None or grad_h is None or grad_p is None:
                # If gradients are None, return x unchanged
                return x
        
        # Second-order gradients (Hessians)
        hess_p = outer_tape.batch_jacobian(grad_p, x)
        hess_h = outer_tape.batch_jacobian(grad_h, x)
        
        # Clean up tape
        del outer_tape
        del inner_tape
        
        # Numerical stabilization for Hessian inversion
        # Use moderate regularization for numerical stability
        # Too much regularization (0.1) prevents convergence to truth
        n_particles = tf.shape(x)[0]
        
        # Adaptive regularization based on matrix condition
        # CRITICAL FIX: Reduced from 0.01 to 1e-6
        # base_reg=0.01 was too large and dominated small prior Hessians (~0.001),
        # causing sign flips and preventing particle drift
        base_reg = 1e-6
        reg_term = base_reg * tf.eye(self.n_dim, batch_shape=[n_particles])
        hess_p_reg = hess_p + reg_term
        
        # Use Moore-Penrose pseudoinverse instead of direct inverse
        # This handles singular/near-singular matrices gracefully
        inv_hess_p = tf.linalg.pinv(hess_p_reg)
        
        # Compute drift coefficients K_1 and K_2
        # From Dai22 Eq. (22) when α + β = 1:
        sum_ab = alpha + beta
        h_term = alpha * beta_dot - alpha_dot * beta
        
        # K_2 = -(αβ' - α'β)/(α+β) * (∇∇^T log p)^{-1}
        K2 = -(h_term / sum_ab) * inv_hess_p
        
        # K_1 = 0.5*Q*(∇∇^T log p) + (αβ' - α'β)/(2(α+β)) * (∇∇^T log p)^{-1} * (∇∇^T log h) * (∇∇^T log p)^{-1}
        #       - (α' + β')/(2(α+β)) * (∇∇^T log p)^{-1}
        k1_middle = (h_term / (2.0 * sum_ab)) * tf.linalg.matmul(
            inv_hess_p, tf.linalg.matmul(hess_h, inv_hess_p)
        )
        k1_last = ((alpha_dot + beta_dot) / (2.0 * sum_ab)) * inv_hess_p
        
        # Expand Q to batch dimension
        Q_batch = tf.expand_dims(self.Q, 0)
        Q_batch = tf.tile(Q_batch, [n_particles, 1, 1])
        
        K1 = 0.5 * Q_batch + k1_middle - k1_last
        
        # Compute drift: f = K_1 * grad_p + K_2 * grad_h
        f = tf.linalg.matvec(K1, grad_p) + tf.linalg.matvec(K2, grad_h)
        
        # Strong drift clipping to prevent divergence
        # Increased from 5.0 to 50.0 to allow larger particle movements
        # Prior has large variance (~1000), particles may need to move ~30 units
        max_drift = 50.0
        f = tf.clip_by_norm(f, clip_norm=max_drift, axes=-1)
        
        # Diffusion term: q * dw where dw ~ N(0, dt*I)
        dw = tf.random.normal(shape=tf.shape(x), stddev=tf.sqrt(dt))
        diffusion = tf.linalg.matvec(self.q_chol, dw)
        
        # Euler-Maruyama update
        x_next = x + f * dt + diffusion
        
        return x_next
    
    def flow_particles(self, particles, model, z, beta_func, n_steps=100, verbose=False):
        """
        Flow particles from prior (λ=0) to posterior (λ=1).
        
        Args:
            particles: Initial particles from prior [n_particles, n_dim]
            model: State-space model
            z: Measurement
            beta_func: Homotopy function β(λ) returning (α, β, α', β')
            n_steps: Number of integration steps
            verbose: Print progress (default: False for performance)
            
        Returns:
            particles_final: Particles at λ=1 [n_particles, n_dim]
        """
        dt = tf.constant(1.0 / n_steps, dtype=tf.float32)
        x = tf.cast(particles, dtype=tf.float32)
        
        # Integration loop
        for i in range(n_steps):
            lam = i / n_steps
            
            # Get homotopy parameters
            alpha, beta, alpha_dot, beta_dot = beta_func(lam)
            
            # Execute one step
            x = self.step_logic(x, z, model, alpha, beta, alpha_dot, beta_dot, dt)
            
            # Check for NaNs (numerical instability indicator)
            if tf.reduce_any(tf.math.is_nan(x)):
                if verbose:
                    print(f"Warning: NaNs detected at step {i}/{n_steps}")
                    print(f"  λ = {lam:.4f}, β = {float(beta):.4f}")
                break
            
            # Optional: Print progress every 20 steps (only if verbose)
            if verbose and i % 20 == 0 and i > 0:
                mean_x = tf.reduce_mean(x, axis=0)
                print(f"  λ = {lam:.2f}: mean = {mean_x.numpy()}")
        
        return x