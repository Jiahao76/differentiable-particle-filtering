import sys
import os
import time
import tracemalloc
import tensorflow as tf
import numpy as np

# --- 新增绘图库 ---
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Models
from src.models.sv_model import StochasticVolatilityModel

# Import Filters
from src.filters.particle_filter import StandardParticleFilter
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter

def export_benchmark_chart(results_data, filename='benchmark_summary.png'):
    """
    Generate a visual benchmark report (Runtime & Memory) instead of text output.
    """
    df = pd.DataFrame(results_data)
    
    # Set visual style
    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color palette (PF=Green, UKF=Blue, EKF=Red)
    colors = {'PF': '#2ecc71', 'UKF': '#3498db', 'EKF': '#e74c3c'}
    
    # ---------------------------
    # Plot 1: Runtime Performance
    # ---------------------------
    sns.barplot(x='Method', y='Runtime (s)', data=df, ax=axes[0], 
                palette=colors, order=['PF', 'UKF', 'EKF'])
    axes[0].set_title('Runtime (Lower is Better)', fontsize=15, fontweight='bold')
    axes[0].set_ylabel('Time (s)')
    axes[0].set_xlabel('')
    
    # Add value annotations
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f'{height:.4f}s', 
                         (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    # ---------------------------
    # Plot 2: Memory Efficiency
    # ---------------------------
    sns.barplot(x='Method', y='Peak Memory (MB)', data=df, ax=axes[1], 
                palette=colors, order=['PF', 'UKF', 'EKF'])
    axes[1].set_title('Peak Memory (Lower is Better)', fontsize=15, fontweight='bold')
    axes[1].set_ylabel('Memory (MB)')
    axes[1].set_xlabel('')
    
    # Add value annotations
    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(f'{height:.2f}MB', 
                         (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    # ---------------------------
    # Add Summary Text Box
    # ---------------------------
    try:
        fastest = df.loc[df['Runtime (s)'].idxmin()]
        slowest = df.loc[df['Runtime (s)'].idxmax()]
        speedup = slowest['Runtime (s)'] / fastest['Runtime (s)']
        
        summary_text = (f"SUMMARY: {fastest['Method']} is the fastest method "
                        f"({speedup:.1f}x speedup vs {slowest['Method']}).")
        
        plt.figtext(0.5, 0.02, summary_text, ha="center", fontsize=12, 
                    bbox={"facecolor":"#f1c40f", "alpha":0.2, "pad":8, "edgecolor": "none"})
    except:
        pass

    plt.tight_layout()
    # Adjust layout to make room for the text at bottom
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig(filename, dpi=300)
    print(f"\n✅ Plot generated successfully: {filename}")

def measure_performance(func, filter_name, *args, **kwargs):
    """
    Helper to measure Runtime and Peak Memory of a function call.
    Includes a warm-up step for TensorFlow graph compilation.
    """
    import gc
    gc.collect()
    
    print(f"[{filter_name}] Warming up (compiling graph)...")
    try:
        # Warm-up (Trigger JIT compilation)
        func(*args, **kwargs)
    except Exception:
        pass
    
    print(f"[{filter_name}] Running benchmark...")
    
    tracemalloc.start()
    start_time = time.time()
    
    # Actual execution
    result = func(*args, **kwargs)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    runtime = end_time - start_time
    peak_memory_mb = peak / (1024 * 1024)
    
    print(f"  > Runtime:     {runtime:.4f} sec")
    print(f"  > Peak Memory: {peak_memory_mb:.4f} MB")
    print("-" * 50)
    
    return result, runtime, peak_memory_mb

def main():
    print("=" * 60)
    print("J.P. Morgan Internship Part 1.II.d: Filter Performance Benchmark")
    print("Comparing Runtime and Memory: PF vs EKF vs UKF")
    print("=" * 60)
    
    # ------------------------------------------------------------------------
    # 1. Setup Data & Model
    # ------------------------------------------------------------------------
    print("\n1. Generating Synthetic Data (Stochastic Volatility Model)...")
    model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
    
    T = 200 
    
    tf.random.set_seed(42)
    x_t = tf.random.normal((1,))
    observations = []
    
    for _ in range(T):
        x_t = model.transition(x_t)
        x_in = tf.reshape(x_t, [1]) 
        y_t = model.observation(x_in)
        observations.append(y_t[0])
        
    obs_tensor = tf.stack(observations)
    obs_tensor = tf.reshape(obs_tensor, (T, 1))
    
    print(f"   Generated {T} time steps of data.")
    print("-" * 50)

    # ------------------------------------------------------------------------
    # 2. Benchmark Execution
    # ------------------------------------------------------------------------
    
    results = {}
    
    # Common Init
    x_init = tf.constant([0.0], dtype=tf.float32)
    P_init = tf.eye(1, dtype=tf.float32)
    Q_matrix = tf.eye(1) * (1.0**2)
    R_matrix = tf.eye(1) * (0.5**2)
    
    # --- PF ---
    pf = StandardParticleFilter(model, num_particles=1000)
    _, pf_time, pf_mem = measure_performance(
        pf.run, "PF (N=1000)", 
        observations=obs_tensor, 
        verbose=False
    )
    results['PF'] = (pf_time, pf_mem)

    # --- EKF ---
    ekf = ExtendedKalmanFilter(model)
    _, ekf_time, ekf_mem = measure_performance(
        ekf.run, "EKF",
        observations=obs_tensor,
        x_init=x_init,
        P_init=P_init,
        Q_matrix=Q_matrix,
        R_matrix=R_matrix
    )
    results['EKF'] = (ekf_time, ekf_mem)

    # --- UKF ---
    ukf = UnscentedKalmanFilter(model, alpha=1e-3, beta=2.0, kappa=0.0)
    _, ukf_time, ukf_mem = measure_performance(
        ukf.run, "UKF",
        observations=obs_tensor,
        x_init=x_init,
        P_init=P_init,
        Q=Q_matrix,
        R=R_matrix
    )
    results['UKF'] = (ukf_time, ukf_mem)

    # ------------------------------------------------------------------------
    # 3. Generate Visual Report
    # ------------------------------------------------------------------------
    print("\nProcessing results for visualization...")
    
    # Prepare data for plotting
    plot_data = {
        'Method': [],
        'Runtime (s)': [],
        'Peak Memory (MB)': []
    }
    
    for name, (r_time, mem) in results.items():
        # Clean up name for the plot label
        clean_name = name.split(' ')[0] # "PF", "EKF", "UKF"
        plot_data['Method'].append(clean_name)
        plot_data['Runtime (s)'].append(r_time)
        plot_data['Peak Memory (MB)'].append(mem)
        
    # Export Chart
    export_benchmark_chart(plot_data, filename='benchmark_summary.png')
    print("Done.")

if __name__ == "__main__":
    main()