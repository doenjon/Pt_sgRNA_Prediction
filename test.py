import os
import sys
import argparse

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.6.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging


# ml cuda/12.6.1 cudnn/9.4.0 


import tensorflow as tf
from sgrna_scorer.data.preprocessing import load_and_preprocess_data
from sgrna_scorer.models.multi_path_atn_model import create_transfer_learning_models
from sgrna_scorer.utils.training import (train_model, evaluate_model, 
                                       save_model_weights, plot_predictions, 
                                       load_model_weights)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=512)]  # Reduce to 512MB
            )
        print("GPU memory limited to 512MB")
    except RuntimeError as e:
        print(f"Error setting GPU memory limit: {e}")
        print("Falling back to CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def parse_args():
    parser = argparse.ArgumentParser(description='Train or load pre-trained model for sgRNA scoring')
    parser.add_argument('--force-train', action='store_true',
                       help='Force training even if pre-trained weights exist')
    parser.add_argument('--skip-indel', action='store_true',
                       help='Skip indel training/loading and go straight to KO fine-tuning')
    parser.add_argument('--no-freeze', action='store_true',
                       help='Do not freeze base model weights during KO training')
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
    
    # Create new models with smaller size
    _, _, fresh_ko_model = create_transfer_learning_models(
        input_shape=(20,),
        num_filters=128,  # Reduced from 256
        num_dense_neurons=64  # Reduced from 128
    )
    
    # Build model with dummy input
    dummy_input = tf.zeros((1, 20))
    _ = fresh_ko_model(dummy_input)
    
    # Compile fresh model
    fresh_ko_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    print("\nFresh KO model architecture:")
    fresh_ko_model.summary()
    
    # Train fresh model with smaller batch size
    trained_fresh_model, fresh_history = train_model(
        fresh_ko_model,
        X_train_ko, y_train_ko,
        X_val_ko, y_val_ko,
        model_name="fresh_ko_model",
        batch_size=8,  # Reduced from 16
        epochs=100
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

def train_and_compare_models(X_train, y_train, X_val, y_val, 
                           X_train_norm, y_train_norm, X_val_norm, y_val_norm,
                           model_name, normalizer=None):
    """Train models on both normalized and unnormalized data for comparison."""
    print(f"\nTraining {model_name} model on unnormalized data...")
    metrics_unnorm = train_fresh_ko_model(X_train, y_train, X_val, y_val)
    
    print(f"\nTraining {model_name} model on normalized data...")
    metrics_norm = train_fresh_ko_model(X_train_norm, y_train_norm, X_val_norm, y_val_norm)
    
    print(f"\n=== {model_name} Model Performance Comparison ===")
    print("\nUnnormalized data metrics:")
    print(metrics_unnorm)
    print("\nNormalized data metrics:")
    print(metrics_norm)
    
    # Calculate improvement
    if metrics_unnorm and metrics_norm:
        mse_improvement = (metrics_unnorm['mse'] - metrics_norm['mse']) / metrics_unnorm['mse'] * 100
        mae_improvement = (metrics_unnorm['mae'] - metrics_norm['mae']) / metrics_unnorm['mae'] * 100
        corr_improvement = (metrics_norm['spearman_corr'] - metrics_unnorm['spearman_corr']) / abs(metrics_unnorm['spearman_corr']) * 100
        
        print("\nImprovement with normalization:")
        print(f"MSE: {mse_improvement:.1f}%")
        print(f"MAE: {mae_improvement:.1f}%")
        print(f"Spearman correlation: {corr_improvement:.1f}%")
    
    return metrics_norm, metrics_unnorm

def main():
    # Load indel data
    print("\nLoading indel data...")
    X_train_indel, X_val_indel, y_train_indel, y_val_indel, indel_normalizer, X_indel_orig, y_indel_orig = load_and_preprocess_data(
        'sgrna_scorer/resources/DeepHF.clean.csv',
        invert_targets=False
    )
    
    print("\nIndel data shapes:")
    print(f"X_train_indel: {X_train_indel.shape}")
    print(f"y_train_indel: {y_train_indel.shape}")
    print(f"X_val_indel: {X_val_indel.shape}")
    print(f"y_val_indel: {y_val_indel.shape}")
    
    # Train on raw indel data
    print("\n=== Training on raw indel data ===")
    raw_indel_metrics = train_fresh_ko_model(
        X_indel_orig[:X_train_indel.shape[0]], y_indel_orig[:y_train_indel.shape[0]],
        X_indel_orig[X_train_indel.shape[0]:], y_indel_orig[y_train_indel.shape[0]:]
    )
    
    # Train on normalized indel data
    print("\n=== Training on normalized indel data ===")
    norm_indel_metrics = train_fresh_ko_model(
        X_train_indel, y_train_indel,
        X_val_indel, y_val_indel
    )
    
    # Load KO data
    print("\nLoading KO data...")
    X_train_ko, X_val_ko, y_train_ko, y_val_ko, ko_normalizer, X_ko_orig, y_ko_orig = load_and_preprocess_data(
        'sgrna_scorer/resources/pt_sat_guides.csv',
        invert_targets=True
    )
    
    print("\nKO data shapes:")
    print(f"X_train_ko: {X_train_ko.shape}")
    print(f"y_train_ko: {y_train_ko.shape}")
    print(f"X_val_ko: {X_val_ko.shape}")
    print(f"y_val_ko: {y_val_ko.shape}")
    
    # Train on raw KO data
    print("\n=== Training on raw KO data ===")
    raw_ko_metrics = train_fresh_ko_model(
        X_ko_orig[:X_train_ko.shape[0]], y_ko_orig[:y_train_ko.shape[0]],
        X_ko_orig[X_train_ko.shape[0]:], y_ko_orig[y_train_ko.shape[0]:]
    )
    
    # Train on normalized KO data
    print("\n=== Training on normalized KO data ===")
    norm_ko_metrics = train_fresh_ko_model(
        X_train_ko, y_train_ko,
        X_val_ko, y_val_ko
    )
    
    # Print comparison of all results
    print("\n=== Final Results ===")
    print("\nRaw Indel Model Metrics:")
    print(raw_indel_metrics)
    print("\nNormalized Indel Model Metrics:")
    print(norm_indel_metrics)
    print("\nRaw KO Model Metrics:")
    print(raw_ko_metrics)
    print("\nNormalized KO Model Metrics:")
    print(norm_ko_metrics)

if __name__ == "__main__":
    main()