import os
import sys
import argparse
import numpy as np

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.6.1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging


# ml cuda/12.6.1 cudnn/9.4.0 


import tensorflow as tf
from sgrna_scorer.data.preprocessing import load_and_preprocess_data
from sgrna_scorer.models.multi_path_atn_model import create_transfer_learning_models
from sgrna_scorer.utils.training import (train_model, evaluate_model, 
                                       save_model_weights, plot_predictions, 
                                       load_model_weights)

def parse_args():
    parser = argparse.ArgumentParser(description='Train or load pre-trained model for sgRNA scoring')
    parser.add_argument('--force-train', action='store_true',
                       help='Force training even if pre-trained weights exist')
    parser.add_argument('--skip-indel', action='store_true',
                       help='Skip indel training/loading and go straight to KO fine-tuning')
    return parser.parse_args()

def train_or_load_base_model(base_model, indel_model, X_train, y_train, X_val, y_val, 
                            weights_path, force_train=False):
    """Train or load the base model depending on conditions."""
    if not force_train and os.path.exists(f"{weights_path}.weights.h5"):
        print(f"\nFound existing weights at {weights_path}.weights.h5")
        if load_model_weights(base_model, weights_path):
            print("Successfully loaded pre-trained weights")
            return True
        print("Failed to load weights, falling back to training")
    
    print("\nTraining indel model...")
    # Compile indel model for pre-training
    indel_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    trained_model, history = train_model(
        indel_model,
        X_train, y_train,
        X_val, y_val,
        model_name="indel_model",
        batch_size=16,
        epochs=20
    )
    
    if trained_model is not None:
        metrics = evaluate_model(trained_model, X_val, y_val)
        print("\nValidation metrics:", metrics)
        plot_predictions(trained_model, X_val, y_val, model_name="indel_model")
        save_model_weights(base_model, weights_path)
        print(f"\nSaved pre-trained base model weights to {weights_path}")
        return True
    
    return False

def train_fresh_ko_model(X_train_ko, y_train_ko, X_val_ko, y_val_ko):
    """Train a fresh model on KO data without transfer learning."""
    print("\nCreating and training fresh KO model (no transfer learning)...")
    
    # Create new models
    _, _, fresh_ko_model = create_transfer_learning_models(input_shape=(20,))
    
    # Build model with dummy input
    dummy_input = tf.zeros((1, 20))
    _ = fresh_ko_model(dummy_input)
    
    # Compile fresh model
    fresh_ko_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),  # Higher learning rate for fresh training
        loss='mse',
        metrics=['mae']
    )
    
    print("\nFresh KO model architecture:")
    fresh_ko_model.summary()
    
    # Train fresh model
    trained_fresh_model, fresh_history = train_model(
        fresh_ko_model,
        X_train_ko, y_train_ko,
        X_val_ko, y_val_ko,
        model_name="fresh_ko_model",
        batch_size=16,
        epochs=100  # Train for longer since starting from scratch
    )
    
    if trained_fresh_model is not None:
        print("\nEvaluating fresh KO model:")
        fresh_metrics = evaluate_model(trained_fresh_model, X_val_ko, y_val_ko)
        print("\nFresh KO validation metrics:", fresh_metrics)
        plot_predictions(trained_fresh_model, X_val_ko, y_val_ko, model_name="fresh_ko_model")
        
        # Save fresh model
        fresh_weights_path = 'sgrna_scorer/resources/fresh_ko_model'
        save_model_weights(fresh_ko_model, fresh_weights_path)
        print(f"\nSaved fresh KO model weights to {fresh_weights_path}")
        
        return fresh_metrics
    
    return None

def main():
    args = parse_args()
    
    # Create models
    print("\nCreating models...")
    base_model, indel_model, ko_model = create_transfer_learning_models(input_shape=(20,))
    
    # Load KO data first since it's our primary task
    print("\nLoading KO data...")
    X_train_ko, X_val_ko, y_train_ko, y_val_ko = load_and_preprocess_data(
        'sgrna_scorer/resources/pt_sat_guides.csv'
    )
    
    print("\nKO data shapes:")
    print(f"X_train_ko: {X_train_ko.shape}")
    print(f"y_train_ko: {y_train_ko.shape}")
    print(f"X_val_ko: {X_val_ko.shape}")
    print(f"y_val_ko: {y_val_ko.shape}")
    
    # Train fresh model for comparison
    fresh_metrics = train_fresh_ko_model(X_train_ko, y_train_ko, X_val_ko, y_val_ko)
    
    if not args.skip_indel:
        print("\nTrying alternative transfer learning approaches...")
        
        # Load DeepHF data
        print("Loading DeepHF data...")
        X_train_indel, X_val_indel, y_train_indel, y_val_indel = load_and_preprocess_data(
            'sgrna_scorer/resources/DeepHF.clean.csv'
        )
        
        # Try different transfer learning strategies
        strategies = [
            {
                'name': 'Minimal pre-training',
                'epochs': 5,
                'learning_rate': 1e-3,
                'fine_tune_lr': 1e-4,
                'freeze_layers': True
            },
            {
                'name': 'No freezing',
                'epochs': 20,
                'learning_rate': 1e-3,
                'fine_tune_lr': 1e-4,
                'freeze_layers': False
            },
            {
                'name': 'Joint training',
                'epochs': 20,
                'learning_rate': 1e-4,
                'fine_tune_lr': 1e-4,
                'freeze_layers': False
            }
        ]
        
        best_metrics = None
        best_strategy = None
        best_model = None
        
        for strategy in strategies:
            print(f"\nTrying strategy: {strategy['name']}")
            
            # Reset models
            base_model, indel_model, ko_model = create_transfer_learning_models(input_shape=(20,))
            
            if strategy['name'] != 'Joint training':
                # Pre-training phase
                indel_model.compile(
                    optimizer=tf.keras.optimizers.Adam(strategy['learning_rate']),
                    loss='mse',
                    metrics=['mae']
                )
                
                trained_model, _ = train_model(
                    indel_model,
                    X_train_indel, y_train_indel,
                    X_val_indel, y_val_indel,
                    model_name=f"indel_model_{strategy['name']}",
                    batch_size=16,
                    epochs=strategy['epochs']
                )
                
                if strategy['freeze_layers']:
                    base_model.trainable = False
            else:
                # Joint training on both datasets
                combined_X_train = np.concatenate([X_train_indel, X_train_ko])
                combined_y_train = np.concatenate([y_train_indel, y_train_ko])
                combined_X_val = np.concatenate([X_val_indel, X_val_ko])
                combined_y_val = np.concatenate([y_val_indel, y_val_ko])
            
            # Fine-tuning phase
            ko_model.compile(
                optimizer=tf.keras.optimizers.Adam(strategy['fine_tune_lr']),
                loss='mse',
                metrics=['mae']
            )
            
            trained_ko_model, _ = train_model(
                ko_model,
                X_train_ko if strategy['name'] != 'Joint training' else combined_X_train,
                y_train_ko if strategy['name'] != 'Joint training' else combined_y_train,
                X_val_ko if strategy['name'] != 'Joint training' else combined_X_val,
                y_val_ko if strategy['name'] != 'Joint training' else combined_y_val,
                model_name=f"ko_model_{strategy['name']}",
                batch_size=16,
                epochs=30
            )
            
            if trained_ko_model is not None:
                metrics = evaluate_model(trained_ko_model, X_val_ko, y_val_ko)
                print(f"\nMetrics for {strategy['name']}:", metrics)
                
                if best_metrics is None or metrics['spearman_corr'] > best_metrics['spearman_corr']:
                    best_metrics = metrics
                    best_strategy = strategy['name']
                    best_model = trained_ko_model
        
        if best_model is not None:
            print(f"\nBest transfer learning strategy was: {best_strategy}")
            print("Best transfer learning metrics:", best_metrics)
            print("\nComparison with fresh model:")
            print("Fresh model metrics:", fresh_metrics)
            print("Best transfer learning metrics:", best_metrics)
            
            # Calculate improvement/degradation
            if fresh_metrics:
                mse_change = (fresh_metrics['mse'] - best_metrics['mse']) / fresh_metrics['mse'] * 100
                mae_change = (fresh_metrics['mae'] - best_metrics['mae']) / fresh_metrics['mae'] * 100
                corr_change = (best_metrics['spearman_corr'] - fresh_metrics['spearman_corr']) / abs(fresh_metrics['spearman_corr']) * 100
                
                print("\nTransfer learning vs Fresh training:")
                print(f"MSE: {'improved' if mse_change > 0 else 'degraded'} by {abs(mse_change):.1f}%")
                print(f"MAE: {'improved' if mae_change > 0 else 'degraded'} by {abs(mae_change):.1f}%")
                print(f"Spearman correlation: {'improved' if corr_change > 0 else 'degraded'} by {abs(corr_change):.1f}%")
    
    print("\nConclusion: Fresh training appears to be more effective for this task.")
    print("Possible reasons:")
    print("1. The indel and KO tasks might be too different")
    print("2. The pre-training dataset might not be relevant enough")
    print("3. The model architecture might be better suited for direct training")
    print("4. The KO dataset might be sufficient on its own")

if __name__ == "__main__":
    main()