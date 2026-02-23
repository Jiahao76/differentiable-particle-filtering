"""
Train Neural OT Networks (mGradNet and FNO)

Trains neural networks to predict optimal transport plans
for particle filter resampling.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from src.filters.neural_ot_resampling import (
    OTResamplingNetwork,
    FourierOTOperator,
    DeepONetOT,
    compute_statistics
)


def load_training_data(data_path: str, train_split: float = 0.8):
    """
    Load and split training data.
    
    Returns:
        train_data, val_data: tuples of (inputs, targets)
    """
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    
    num_examples = len(data['particles'])
    num_train = int(num_examples * train_split)
    
    # Shuffle
    indices = np.random.permutation(num_examples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    def extract_subset(indices):
        subset = {}
        for key in data.keys():
            subset[key] = data[key][indices]
        return subset
    
    train_data = extract_subset(train_indices)
    val_data = extract_subset(val_indices)
    
    print(f"✓ Loaded {num_examples} examples")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val: {len(val_indices)}")
    
    return train_data, val_data


def ot_loss(P_pred: tf.Tensor, P_true: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    """
    OT loss with marginal constraints.
    
    Args:
        P_pred: (N, N) predicted transport plan
        P_true: (N, N) ground truth transport plan
        weights: (N,) source weights
    
    Returns:
        loss: scalar loss value
    """
    # Plan matching loss
    loss_plan = tf.reduce_mean(tf.square(P_pred - P_true))
    
    # Marginal constraint losses (soft)
    N = tf.shape(P_pred)[0]
    uniform = tf.ones(N, dtype=tf.float32) / tf.cast(N, tf.float32)
    
    marginal_source = tf.reduce_sum(P_pred, axis=1)  # Should equal weights
    marginal_target = tf.reduce_sum(P_pred, axis=0)  # Should equal uniform
    
    loss_marginal_source = tf.reduce_mean(tf.square(marginal_source - weights))
    loss_marginal_target = tf.reduce_mean(tf.square(marginal_target - uniform))
    
    # Total loss
    loss = loss_plan + 0.1 * (loss_marginal_source + loss_marginal_target)
    
    return loss


def monge_ampere_loss(P_pred: tf.Tensor, particles: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    """
    Physics-informed loss: Monge-Ampère residual (simplified).
    
    For the Monge-Ampère equation: det(∇²φ) = μ/ν
    """
    # Simplified: encourage plan to match weight ratios
    N = tf.shape(P_pred)[0]
    
    # Expected weight transfer
    expected_transfer = tf.matmul(P_pred, tf.reshape(weights, [-1, 1]))
    expected_transfer = tf.reshape(expected_transfer, [-1])
    
    # Should be uniform after transport
    uniform = tf.ones(N, dtype=tf.float32) / tf.cast(N, tf.float32)
    
    loss_ma = tf.reduce_mean(tf.square(expected_transfer - uniform))
    
    return loss_ma


def train_mgradnet(
    train_data: dict,
    val_data: dict,
    state_dim: int = 1,
    hidden_dim: int = 256,
    num_layers: int = 4,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    save_dir: str = 'checkpoints'
):
    """
    Train mGradNet for OT resampling.
    """
    print("\n" + "="*70)
    print("Training mGradNet")
    print("="*70)
    
    # Create network
    network = OTResamplingNetwork(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    num_train = len(train_data['particles'])
    num_batches = (num_train + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(num_train)
        
        epoch_losses = []
        
        # Training
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_train)
            batch_indices = indices[start_idx:end_idx]
            
            with tf.GradientTape() as tape:
                batch_loss = 0.0
                
                for idx in batch_indices:
                    # Extract example
                    particles = tf.constant(train_data['particles'][idx], dtype=tf.float32)
                    weights = tf.constant(train_data['weights'][idx], dtype=tf.float32)
                    P_true = tf.constant(train_data['P_true'][idx], dtype=tf.float32)
                    model_params = tf.constant(train_data['model_params'][idx], dtype=tf.float32)
                    observation = tf.constant(train_data['observation'][idx], dtype=tf.float32)
                    
                    # Construct statistics
                    stats = {
                        'mean': tf.constant(train_data['mean'][idx], dtype=tf.float32),
                        'cov': tf.constant(train_data['cov'][idx], dtype=tf.float32),
                        'ess': tf.constant(train_data['ess'][idx], dtype=tf.float32),
                        'weight_entropy': tf.constant(train_data['weight_entropy'][idx], dtype=tf.float32),
                        'innovation': tf.constant(train_data['innovation'][idx], dtype=tf.float32)
                    }
                    
                    # Forward pass
                    P_pred = network(particles, weights, model_params, observation, stats, training=True)
                    
                    # Loss
                    loss = ot_loss(P_pred, P_true, weights)
                    loss_ma = monge_ampere_loss(P_pred, particles, weights)
                    
                    batch_loss += loss + 0.01 * loss_ma
                
                batch_loss /= len(batch_indices)
            
            # Backward pass
            gradients = tape.gradient(batch_loss, network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, network.trainable_variables))
            
            epoch_losses.append(batch_loss.numpy())
            pbar.set_postfix({'loss': f'{batch_loss.numpy():.6f}'})
        
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss = evaluate_mgradnet(network, val_data)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(save_dir, exist_ok=True)
            network.save_weights(os.path.join(save_dir, 'mgradnet_best.weights.h5'))
            print(f"  ✓ Saved best model (val_loss={val_loss:.6f})")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('mGradNet Training')
    plt.savefig(os.path.join(save_dir, 'mgradnet_training.png'), dpi=150, bbox_inches='tight')
    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.6f}")
    
    return network, train_losses, val_losses


def evaluate_mgradnet(network: OTResamplingNetwork, data: dict) -> float:
    """Evaluate mGradNet on validation set."""
    losses = []
    
    for idx in range(len(data['particles'])):
        particles = tf.constant(data['particles'][idx], dtype=tf.float32)
        weights = tf.constant(data['weights'][idx], dtype=tf.float32)
        P_true = tf.constant(data['P_true'][idx], dtype=tf.float32)
        model_params = tf.constant(data['model_params'][idx], dtype=tf.float32)
        observation = tf.constant(data['observation'][idx], dtype=tf.float32)
        
        stats = {
            'mean': tf.constant(data['mean'][idx], dtype=tf.float32),
            'cov': tf.constant(data['cov'][idx], dtype=tf.float32),
            'ess': tf.constant(data['ess'][idx], dtype=tf.float32),
            'weight_entropy': tf.constant(data['weight_entropy'][idx], dtype=tf.float32),
            'innovation': tf.constant(data['innovation'][idx], dtype=tf.float32)
        }
        
        P_pred = network(particles, weights, model_params, observation, stats, training=False)
        loss = ot_loss(P_pred, P_true, weights)
        losses.append(loss.numpy())
    
    return np.mean(losses)


def train_fno(
    train_data: dict,
    val_data: dict,
    modes: int = 16,
    width: int = 64,
    num_layers: int = 4,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    save_dir: str = 'checkpoints'
):
    """
    Train Fourier Neural Operator for OT.
    """
    print("\n" + "="*70)
    print("Training Fourier Neural Operator")
    print("="*70)
    
    # Create network
    network = FourierOTOperator(
        modes=modes,
        width=width,
        num_layers=num_layers
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    num_train = len(train_data['particles'])
    num_batches = (num_train + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        indices = np.random.permutation(num_train)
        epoch_losses = []
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_train)
            batch_indices = indices[start_idx:end_idx]
            
            with tf.GradientTape() as tape:
                batch_loss = 0.0
                
                for idx in batch_indices:
                    cost = tf.constant(train_data['cost'][idx], dtype=tf.float32)
                    weights = tf.constant(train_data['weights'][idx], dtype=tf.float32)
                    P_true = tf.constant(train_data['P_true'][idx], dtype=tf.float32)
                    epsilon = float(train_data['epsilon'][idx])
                    
                    # Forward pass
                    P_pred = network(cost, weights, epsilon, training=True)
                    
                    # Loss
                    loss = ot_loss(P_pred, P_true, weights)
                    batch_loss += loss
                
                batch_loss /= len(batch_indices)
            
            gradients = tape.gradient(batch_loss, network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, network.trainable_variables))
            
            epoch_losses.append(batch_loss.numpy())
            pbar.set_postfix({'loss': f'{batch_loss.numpy():.6f}'})
        
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss = evaluate_fno(network, val_data)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(save_dir, exist_ok=True)
            network.save_weights(os.path.join(save_dir, 'fno_best.weights.h5'))
            print(f"  ✓ Saved best model (val_loss={val_loss:.6f})")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('FNO Training')
    plt.savefig(os.path.join(save_dir, 'fno_training.png'), dpi=150, bbox_inches='tight')
    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.6f}")
    
    return network, train_losses, val_losses


def evaluate_fno(network: FourierOTOperator, data: dict) -> float:
    """Evaluate FNO on validation set."""
    losses = []
    
    for idx in range(len(data['cost'])):
        cost = tf.constant(data['cost'][idx], dtype=tf.float32)
        weights = tf.constant(data['weights'][idx], dtype=tf.float32)
        P_true = tf.constant(data['P_true'][idx], dtype=tf.float32)
        epsilon = float(data['epsilon'][idx])
        
        P_pred = network(cost, weights, epsilon, training=False)
        loss = ot_loss(P_pred, P_true, weights)
        losses.append(loss.numpy())
    
    return np.mean(losses)


def main():
    parser = argparse.ArgumentParser(description='Train neural OT networks')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data (.npz file)')
    parser.add_argument('--method', type=str, choices=['mgradnet', 'fno', 'both'],
                       default='both', help='Which method to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for mGradNet')
    parser.add_argument('--fno_modes', type=int, default=16,
                       help='Number of Fourier modes for FNO')
    parser.add_argument('--fno_width', type=int, default=64,
                       help='Width for FNO')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    print("="*70)
    print("Neural OT Network Training")
    print("="*70)
    print(f"Data: {args.data}")
    print(f"Method: {args.method}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print()
    
    # Load data
    train_data, val_data = load_training_data(args.data)
    
    # Train
    if args.method in ['mgradnet', 'both']:
        train_mgradnet(
            train_data, val_data,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir
        )
    
    if args.method in ['fno', 'both']:
        train_fno(
            train_data, val_data,
            modes=args.fno_modes,
            width=args.fno_width,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir
        )
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == '__main__':
    main()
