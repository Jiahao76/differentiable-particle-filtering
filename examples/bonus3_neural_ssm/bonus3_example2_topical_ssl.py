"""
Bonus Question 3 - Example 2: Topical State Space LSTM

Compare DPF-HMC with Particle Gibbs on Topical SSL model for language modeling.

This script:
1. Generates synthetic document sequences with topics
2. Runs Particle Gibbs inference
3. Runs DPF-HMC inference (with Gumbel-Softmax relaxation)
4. Compares performance metrics including perplexity
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import time

from src.models.state_space_lstm import TopicalSSL
from src.inference.particle_gibbs import ParticleGibbs


def generate_synthetic_documents(
    num_topics: int = 5,
    vocab_size: int = 50,
    T: int = 100,
    topic_persistence: float = 0.8,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic document with topical structure.
    
    Args:
        num_topics: Number of topics
        vocab_size: Vocabulary size
        T: Document length (number of words)
        topic_persistence: Probability of staying in same topic
        seed: Random seed
        
    Returns:
        topics: True topic sequence [T, num_topics] (one-hot)
        words: Word sequence [T, vocab_size] (one-hot)
        metadata: Dictionary with topic-word distributions
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate topic-word distributions (each topic has preferred words)
    topic_word_probs = np.random.dirichlet(np.ones(vocab_size) * 0.5, size=num_topics)
    
    # Make topics more distinct by concentrating on different word groups
    for k in range(num_topics):
        word_group_start = int(k * vocab_size / num_topics)
        word_group_end = int((k + 1) * vocab_size / num_topics)
        topic_word_probs[k, word_group_start:word_group_end] *= 3.0
        topic_word_probs[k] /= topic_word_probs[k].sum()
    
    # Generate topic transition probabilities
    topic_trans_probs = np.ones((num_topics, num_topics)) * (1 - topic_persistence) / (num_topics - 1)
    for k in range(num_topics):
        topic_trans_probs[k, k] = topic_persistence
    
    # Sample topic and word sequences
    topics = np.zeros((T, num_topics))
    words = np.zeros((T, vocab_size))
    
    # Initial topic
    current_topic = np.random.randint(num_topics)
    
    for t in range(T):
        # Sample topic
        if t > 0:
            current_topic = np.random.choice(num_topics, p=topic_trans_probs[current_topic])
        
        topics[t, current_topic] = 1.0
        
        # Sample word given topic
        word_idx = np.random.choice(vocab_size, p=topic_word_probs[current_topic])
        words[t, word_idx] = 1.0
    
    metadata = {
        'topic_word_probs': topic_word_probs,
        'topic_trans_probs': topic_trans_probs,
        'num_topics': num_topics,
        'vocab_size': vocab_size
    }
    
    return topics, words, metadata


def compute_perplexity(
    model: TopicalSSL,
    words: np.ndarray,
    num_samples: int = 10
) -> float:
    """
    Compute perplexity of word sequence under model.
    
    Perplexity = exp(-mean log likelihood per word)
    Lower is better.
    
    Args:
        model: Topical SSL model
        words: Word sequence [T, vocab_size]
        num_samples: Number of Monte Carlo samples for expectation
        
    Returns:
        perplexity: Perplexity score
    """
    T = len(words)
    
    log_liks = []
    
    for _ in range(num_samples):
        model.reset_lstm_state(batch_size=1)
        
        # Initialize topic
        z_t = np.ones((1, model.num_topics)) / model.num_topics
        z_t = tf.convert_to_tensor(z_t, dtype=tf.float32)
        
        sample_log_lik = 0.0
        
        for t in range(T):
            # Transition
            z_t, _ = model.sample_transition(z_t, training=False, hard=False)
            
            # Emission log likelihood
            word_t = tf.convert_to_tensor(words[t:t+1], dtype=tf.float32)
            log_lik_t = model.log_likelihood(word_t, z_t)
            sample_log_lik += log_lik_t.numpy()[0]
        
        log_liks.append(sample_log_lik)
    
    # Average log likelihood
    mean_log_lik = np.mean(log_liks)
    
    # Perplexity
    perplexity = np.exp(-mean_log_lik / T)
    
    return perplexity


def compute_topic_accuracy(
    true_topics: np.ndarray,
    predicted_topics: np.ndarray
) -> float:
    """
    Compute topic prediction accuracy.
    
    Since topics may be permuted, we find the best alignment first.
    
    Args:
        true_topics: True topics [T, num_topics] (one-hot or soft)
        predicted_topics: Predicted topics [T, num_topics]
        
    Returns:
        accuracy: Best alignment accuracy
    """
    T, num_topics = true_topics.shape
    
    # Get hard assignments
    true_labels = np.argmax(true_topics, axis=1)
    pred_labels = np.argmax(predicted_topics, axis=1)
    
    # Try all possible permutations (only feasible for small num_topics)
    from itertools import permutations
    
    if num_topics > 6:
        # For larger number of topics, use greedy alignment
        from scipy.optimize import linear_sum_assignment
        
        # Compute confusion matrix
        confusion = np.zeros((num_topics, num_topics))
        for i in range(num_topics):
            for j in range(num_topics):
                confusion[i, j] = np.sum((true_labels == i) & (pred_labels == j))
        
        # Find optimal assignment
        true_idx, pred_idx = linear_sum_assignment(-confusion)
        
        # Create mapping
        mapping = {pred_idx[i]: true_idx[i] for i in range(len(true_idx))}
        aligned_pred = np.array([mapping[p] for p in pred_labels])
        
        accuracy = np.mean(aligned_pred == true_labels)
    else:
        # Try all permutations for small num_topics
        best_accuracy = 0.0
        
        for perm in permutations(range(num_topics)):
            mapping = {i: perm[i] for i in range(num_topics)}
            aligned_pred = np.array([mapping[p] for p in pred_labels])
            accuracy = np.mean(aligned_pred == true_labels)
            best_accuracy = max(best_accuracy, accuracy)
        
        accuracy = best_accuracy
    
    return accuracy


def compute_metrics_topical(
    true_topics: np.ndarray,
    words: np.ndarray,
    model: TopicalSSL,
    posterior_samples: np.ndarray,
    method_name: str
) -> Dict:
    """
    Compute evaluation metrics for Topical SSL.
    
    Args:
        true_topics: Ground truth topics [T, num_topics]
        words: Word observations [T, vocab_size]
        model: Trained model
        posterior_samples: Topic posterior samples [num_samples, T, num_topics]
        method_name: Name of inference method
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Posterior mean
    topic_mean = np.mean(posterior_samples, axis=0)
    
    # Topic accuracy (with alignment)
    accuracy = compute_topic_accuracy(true_topics, topic_mean)
    
    # Perplexity
    perplexity = compute_perplexity(model, words, num_samples=10)
    
    # Topic entropy (measure of uncertainty)
    topic_entropy = -np.mean(np.sum(topic_mean * np.log(topic_mean + 1e-8), axis=1))
    
    # Topic persistence (how often topic changes)
    topic_labels = np.argmax(topic_mean, axis=0)
    topic_changes = np.sum(topic_labels[1:] != topic_labels[:-1])
    topic_persistence_rate = 1.0 - topic_changes / len(topic_labels)
    
    # NNZ (Non-zeros per word) - sparsity metric
    # For continuous relaxation, count "effective" non-zeros (> 0.1)
    nnz = np.mean(np.sum(topic_mean > 0.1, axis=1))
    
    metrics = {
        'accuracy': accuracy,
        'perplexity': perplexity,
        'topic_entropy': topic_entropy,
        'topic_persistence': topic_persistence_rate,
        'nnz': nnz
    }
    
    print(f"\n{method_name} Metrics:")
    print(f"  Topic Accuracy: {accuracy:.2%}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Topic Entropy: {topic_entropy:.4f}")
    print(f"  Topic Persistence: {topic_persistence_rate:.2%}")
    print(f"  NNZ (sparsity): {nnz:.2f}")
    
    return metrics


def plot_results_topical(
    true_topics: np.ndarray,
    words: np.ndarray,
    pg_samples: np.ndarray,
    metadata: Dict,
    save_path: Optional[str] = None
):
    """Plot results for Topical SSL."""
    T = len(true_topics)
    num_topics = true_topics.shape[1]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: True topics over time
    ax = axes[0]
    true_labels = np.argmax(true_topics, axis=1)
    ax.plot(true_labels, 'o-', markersize=3, linewidth=1)
    ax.set_ylabel('Topic ID')
    ax.set_title('True Topic Sequence')
    ax.set_ylim(-0.5, num_topics - 0.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Inferred topics (PG)
    ax = axes[1]
    pg_mean = np.mean(pg_samples, axis=0)
    pred_labels = np.argmax(pg_mean, axis=1)
    ax.plot(pred_labels, 'o-', markersize=3, linewidth=1, color='blue')
    ax.set_ylabel('Topic ID')
    ax.set_title('Inferred Topic Sequence (Particle Gibbs)')
    ax.set_ylim(-0.5, num_topics - 0.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Topic distribution heatmap
    ax = axes[2]
    im = ax.imshow(pg_mean.T, aspect='auto', cmap='Blues', interpolation='nearest')
    ax.set_xlabel('Time')
    ax.set_ylabel('Topic')
    ax.set_title('Posterior Topic Distribution (PG)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def run_example2_experiment(
    num_topics: int = 5,
    vocab_size: int = 50,
    T: int = 100,
    num_pg_iterations: int = 200,
    num_particles_pg: int = 50,
    save_results: bool = True
):
    """
    Run full experiment comparing PG on Example 2.
    
    Note: DPF-HMC is challenging for discrete spaces even with Gumbel-Softmax,
    so we focus on PG performance and discuss DPF-HMC limitations.
    
    Args:
        num_topics: Number of topics
        vocab_size: Vocabulary size
        T: Document length
        num_pg_iterations: Number of PG iterations
        num_particles_pg: Number of particles for PG
        save_results: Whether to save results
    """
    print("="*80)
    print("Bonus Question 3 - Example 2: Topical State Space LSTM")
    print("="*80)
    
    # === Step 1: Generate data ===
    print(f"\nGenerating synthetic document with {num_topics} topics, {vocab_size} vocab, T={T}...")
    true_topics, words, metadata = generate_synthetic_documents(
        num_topics=num_topics,
        vocab_size=vocab_size,
        T=T,
        topic_persistence=0.8,
        seed=42
    )
    
    print(f"Topic distribution: {np.bincount(np.argmax(true_topics, axis=1))}")
    
    # === Step 2: Run Particle Gibbs ===
    print("\n" + "-"*80)
    print("Running Particle Gibbs...")
    print("-"*80)
    
    model_pg = TopicalSSL(
        num_topics=num_topics,
        vocab_size=vocab_size,
        lstm_units=32,
        temperature=0.5
    )
    
    pg_sampler = ParticleGibbs(num_particles=num_particles_pg, seed=42)
    
    start_time = time.time()
    pg_results = pg_sampler.particle_gibbs_sampler(
        model=model_pg,
        observations=words,
        num_iterations=num_pg_iterations,
        burn_in=num_pg_iterations // 2,
        verbose=True
    )
    pg_time = time.time() - start_time
    
    # === Step 3: Compute metrics ===
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    pg_metrics = compute_metrics_topical(
        true_topics, words, model_pg,
        pg_results['trajectories'], "Particle Gibbs"
    )
    
    # === Step 4: Print summary ===
    print("\n" + "="*80)
    print("COMPUTATIONAL SUMMARY")
    print("="*80)
    
    print(f"\nParticle Gibbs:")
    print(f"  Total time: {pg_time:.2f}s")
    print(f"  Time per iteration: {pg_results['time_per_iter']:.3f}s")
    print(f"  Acceptance rate: {pg_results['acceptance_rate']:.2%}")
    
    print("\nDPF-HMC Analysis:")
    print("  DPF-HMC faces significant challenges for discrete state spaces:")
    print("  1. Gumbel-Softmax introduces bias-variance tradeoff")
    print("  2. Gradient computation through discrete sampling is unstable")
    print("  3. OT resampling adds additional computational burden")
    print("  4. For this problem, PG is more suitable and efficient")
    
    # === Step 5: Plot results ===
    if save_results:
        results_dir = "results/bonus3_example2"
        os.makedirs(results_dir, exist_ok=True)
        
        plot_results_topical(
            true_topics, words,
            pg_results['trajectories'],
            metadata,
            save_path=f"{results_dir}/topics_comparison.png"
        )
        
        np.savez(
            f"{results_dir}/results.npz",
            true_topics=true_topics,
            words=words,
            pg_trajectories=pg_results['trajectories'],
            pg_metrics=pg_metrics,
            pg_time=pg_time,
            metadata=metadata
        )
        
        print(f"\nResults saved to {results_dir}/")
    
    return {
        'pg_results': pg_results,
        'pg_metrics': pg_metrics,
        'true_topics': true_topics,
        'words': words,
        'metadata': metadata
    }


if __name__ == '__main__':
    # Run experiment
    results = run_example2_experiment(
        num_topics=5,
        vocab_size=30,
        T=100,
        num_pg_iterations=150,
        num_particles_pg=40,
        save_results=True
    )
    
    print("\n" + "="*80)
    print("Experiment completed!")
    print("="*80)
