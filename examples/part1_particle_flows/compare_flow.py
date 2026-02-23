import sys
import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sv_model import StochasticVolatilityModel
from src.filters.particle_filter import StandardParticleFilter
from src.filters.flow_pf import InvertibleFlowParticleFilter

# --- Visualization Function ---
def export_benchmark_chart(df, filename='flow_benchmark.png'):
    # Ensure filename is in results directory
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, os.path.basename(filename))
    
    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = {'Standard PF': '#95a5a6', 'PF-Flow (Li-17)': '#e74c3c'} # Grey vs Red

    # 1. Runtime
    sns.barplot(x='Method', y='Runtime (s)', data=df, ax=axes[0], palette=colors)
    axes[0].set_title('Runtime Cost (Lower is Better)', fontsize=15, fontweight='bold')
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.2f}s', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 2. RMSE (Accuracy)
    sns.barplot(x='Method', y='RMSE', data=df, ax=axes[1], palette=colors)
    axes[1].set_title('Estimation Error (RMSE) (Lower is Better)', fontsize=15, fontweight='bold')
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.4f}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    print(f"✅ Chart saved to {filepath}")

def compute_rmse(true_state, est_state):
    # Ensure inputs are flat numpy arrays
    t = np.array(true_state).flatten()
    e = np.array(est_state).flatten()
    # Align lengths if necessary (e.g. if estimates start from t=1)
    min_len = min(len(t), len(e))
    return np.sqrt(np.mean((t[:min_len] - e[:min_len])**2))

def main():
    print("="*60)
    print("Replicating Li(17): Invertible Particle Flow vs Standard PF")
    print("="*60)
    
    # 1. Setup Data
    T = 100
    model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
    
    tf.random.set_seed(42) # Fixed random seed for reproducibility
    x_true = []
    obs = []
    x = tf.random.normal((1,))
    
    # Generate synthetic data
    for _ in range(T):
        x = model.transition(x)
        y = model.observation(x)
        x_true.append(x[0])
        obs.append(y[0])
    
    x_true = np.array(x_true)
    obs_tensor = tf.reshape(tf.stack(obs), (T, 1))
    
    results = {'Method': [], 'Runtime (s)': [], 'RMSE': []}

    # ==========================================
    # CRITICAL ADJUSTMENT: Stress Testing Standard PF
    # ==========================================
    # We reduce the particle count to simulate a "sparse" environment.
    # Standard PF requires many particles to cover the state space effectively.
    # Flow PF should perform better here by guiding particles toward the likelihood.
    # ==========================================
    N_particles = 50   # Reduced from 200 to 50 to induce degeneracy in Standard PF
    flow_steps = 20    # Increased steps for smoother ODE integration
    step_size = 0.02   # Decreased step size to prevent overshooting the target
    # ==========================================

    # 2. Standard PF (Baseline)
    print(f"Running Standard PF (N={N_particles})...")
    start = time.time()
    pf = StandardParticleFilter(model, num_particles=N_particles)
    est_pf = pf.run(obs_tensor)
    
    # Handle tuple return type if applicable
    if isinstance(est_pf, tuple): 
        est_pf = est_pf[0]
        
    end = time.time()
    
    rmse_pf = compute_rmse(x_true, est_pf.numpy())
    results['Method'].append('Standard PF')
    results['Runtime (s)'].append(end - start)
    results['RMSE'].append(rmse_pf)
    
    # 3. Invertible Flow PF (Li-17 / EDH)
    print(f"Running PF-Flow (N={N_particles}, Steps={flow_steps}, eps={step_size})...")
    start = time.time()
    fpf = InvertibleFlowParticleFilter(model, num_particles=N_particles, 
                                       flow_steps=flow_steps, step_size=step_size)
    est_flow = fpf.run(obs_tensor)
    
    # Handle tuple return type if applicable
    if isinstance(est_flow, tuple): 
        est_flow = est_flow[0]
        
    end = time.time()
    
    rmse_flow = compute_rmse(x_true, est_flow.numpy())
    results['Method'].append('PF-Flow (Li-17)')
    results['Runtime (s)'].append(end - start)
    results['RMSE'].append(rmse_flow)

    # 4. Output Results
    df = pd.DataFrame(results)
    print("\nResults:")
    print(df)
    
    # Verification: Did Flow outperform PF?
    if rmse_flow < rmse_pf:
        improvement = ((rmse_pf - rmse_flow) / rmse_pf) * 100
        print(f"\n✅ SUCCESS: Flow reduced error by {improvement:.1f}%")
    else:
        print(f"\n❌ WARNING: Flow did not outperform. Consider tuning step_size.")

    # Generate Chart
    export_benchmark_chart(df)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Explicit cleanup to prevent TensorFlow 'NoneType' shutdown errors
        try:
            import gc
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass