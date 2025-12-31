"""
Debug script to diagnose LEDH performance issues
Comparing multiple LEDH implementations
"""
import sys
import os
import time
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sv_model import StochasticVolatilityModel
from src.filters.ledh_flow import LEDHFlowFilter
from src.filters.edh_flow import EDHFlowFilter
from src.filters.ledh_simple import SimpleLEDHFlowFilter, NumericalLEDHFlowFilter

def compute_rmse(true_state, est_state):
    t = np.array(true_state).flatten()
    e = np.array(est_state).flatten()
    min_len = min(len(t), len(e))
    return np.sqrt(np.mean((t[:min_len] - e[:min_len])**2))

def generate_sv_data(model, T=100, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    x_true = []
    observations = []
    x = tf.random.normal((1,), dtype=tf.float32)
    
    for _ in range(T):
        x = model.transition(x)
        y = model.observation(x)
        x_true.append(x.numpy()[0])
        observations.append(y.numpy()[0])
    
    x_true = np.array(x_true)
    obs_tensor = tf.reshape(tf.stack(observations), (T, 1))
    
    return x_true, obs_tensor

def main():
    print("="*70)
    print("LEDH vs EDH Debug Analysis")
    print("="*70)
    
    # Setup
    T = 50  # Shorter for debugging
    N_particles = 100
    flow_steps = 30
    step_size = 1.0 / flow_steps
    
    model = StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
    x_true, observations = generate_sv_data(model, T=T, seed=42)
    
    # Test with different parameters
    configs = [
        {"flow_steps": 20, "step_size": 1.0/20},
        {"flow_steps": 30, "step_size": 1.0/30},
        {"flow_steps": 50, "step_size": 1.0/50},
    ]
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Config: flow_steps={config['flow_steps']}, step_size={config['step_size']:.4f}")
        print(f"{'='*70}")
        
        # EDH Flow
        print("\nEDH Flow:")
        start = time.time()
        edh = EDHFlowFilter(model, num_particles=N_particles, 
                           flow_steps=config['flow_steps'], 
                           step_size=config['step_size'])
        est_edh = edh.run(observations)
        time_edh = time.time() - start
        rmse_edh = compute_rmse(x_true, est_edh.numpy())
        print(f"  RMSE: {rmse_edh:.4f}")
        print(f"  Time: {time_edh:.2f}s")
        
        # LEDH Flow (Current)
        print("\nLEDH Flow (Current Vectorized):")
        start = time.time()
        ledh = LEDHFlowFilter(model, num_particles=N_particles,
                             flow_steps=config['flow_steps'],
                             step_size=config['step_size'])
        est_ledh = ledh.run(observations)
        time_ledh = time.time() - start
        rmse_ledh = compute_rmse(x_true, est_ledh.numpy())
        print(f"  RMSE: {rmse_ledh:.4f}")
        print(f"  Time: {time_ledh:.2f}s")
        
        # Simplified LEDH
        print("\nLEDH Flow (Simplified Loop):")
        start = time.time()
        simple_ledh = SimpleLEDHFlowFilter(model, num_particles=N_particles,
                                          flow_steps=config['flow_steps'],
                                          step_size=config['step_size'])
        est_simple = simple_ledh.run(observations)
        time_simple = time.time() - start
        rmse_simple = compute_rmse(x_true, est_simple.numpy())
        print(f"  RMSE: {rmse_simple:.4f}")
        print(f"  Time: {time_simple:.2f}s")
        
        # Numerical (RK2) LEDH
        print("\nLEDH Flow (RK2 Numerical):")
        start = time.time()
        numerical_ledh = NumericalLEDHFlowFilter(model, num_particles=N_particles,
                                               flow_steps=config['flow_steps'],
                                               step_size=config['step_size'])
        est_numerical = numerical_ledh.run(observations)
        time_numerical = time.time() - start
        rmse_numerical = compute_rmse(x_true, est_numerical.numpy())
        print(f"  RMSE: {rmse_numerical:.4f}")
        print(f"  Time: {time_numerical:.2f}s")
        
        # Comparison
        print(f"\nComparison:")
        results = {
            'EDH': rmse_edh,
            'LEDH (Current)': rmse_ledh,
            'LEDH (Simple)': rmse_simple,
            'LEDH (RK2)': rmse_numerical
        }
        
        min_rmse = min(results.values())
        print(f"  Best: {[k for k,v in results.items() if v == min_rmse][0]} ({min_rmse:.4f})")
        
        for name, rmse in results.items():
            if name != 'EDH':
                if rmse < rmse_edh:
                    improvement = ((rmse_edh - rmse) / rmse_edh) * 100
                    print(f"  ✓ {name}: better than EDH by {improvement:.1f}%")
                else:
                    worse = ((rmse - rmse_edh) / rmse_edh) * 100
                    print(f"  ❌ {name}: worse than EDH by {worse:.1f}%")

if __name__ == "__main__":
    main()
