import os
import sys
import argparse
from datetime import datetime

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.6.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging


# ml cuda/12.6.1 cudnn/9.4.0 


import tensorflow as tf
from sgrna_scorer.data.preprocessing import load_and_preprocess_data
from sgrna_scorer.models.multi_path_atn_model import create_transfer_learning_models, plot_model_diagram
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
        print("Error setting GPU memory limit:", str(e))
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

def train_or_load_base_model(base_model, X_train, y_train, X_val, y_val, 
                            weights_path, force_train=False):
    """Train or load the base model depending on conditions."""
    if not force_train and os.path.exists(f"{weights_path}.weights.h5"):
        print(f"\nFound existing weights at {weights_path}.weights.h5")
        if load_model_weights(base_model, weights_path):
            print("Successfully loaded pre-trained weights")
            return True
        print("Failed to load weights, falling back to training")
    
    print("\nTraining base model on indel data...")
    # Compile base model for pre-training
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,  # High initial learning rate
        clipnorm=1.0
    )
    
    # Compile with simple loss and metrics for single output
    base_model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Add callbacks with consistent stopping conditions
    base_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f'{weights_path}_checkpoint.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=5,  
            min_lr=1e-5,
            min_delta=0.001
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            min_delta=0.001
        )
    ]
    
    trained_model, history = train_model(
        base_model,
        X_train, y_train,
        X_val, y_val,
        model_name="base_model",
        batch_size=64,
        epochs=250,  # Max epochs
        callbacks=base_callbacks,
        use_base_callbacks=False
    )
    
    if trained_model is not None:
        metrics = evaluate_model(trained_model, X_val, y_val)
        print("\nValidation metrics:", metrics)
        plot_predictions(trained_model, X_val, y_val, model_name="base_model")
        save_model_weights(base_model, weights_path)
        print(f"\nSaved pre-trained base model weights to {weights_path}")
        return True
    
    return False

def train_fresh_ko_model(X_train_ko, y_train_ko, X_val_ko, y_val_ko):
    """Train a fresh model on KO data without transfer learning."""
    print("\nCreating and training fresh KO model (no transfer learning)...")
    
    # Create new KO model with dual path
    _, fresh_ko_model = create_transfer_learning_models(
        input_shape=(20,),
        num_filters=128,
        num_dense_neurons=64
    )
    
    # Build model with dummy input
    dummy_input = tf.zeros((1, 20))
    _ = fresh_ko_model(dummy_input)
    
    # Compile fresh model with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,  # High initial learning rate
        clipnorm=1.0
    )
    
    fresh_ko_model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    print("\nFresh KO model architecture:")
    fresh_ko_model.summary()
    
    # Add callbacks with consistent stopping conditions
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'sgrna_scorer/resources/checkpoints/fresh_ko_model_{epoch:02d}.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=7,  # Reduced from 20 to 5
            min_lr=1e-5,
            min_delta=0.001
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            min_delta=0.001
        )
    ]
    
    # Train fresh model
    trained_fresh_model, fresh_history = train_model(
        fresh_ko_model,
        X_train_ko, y_train_ko,
        X_val_ko, y_val_ko,
        model_name="fresh_ko_model",
        batch_size=32,
        epochs=250,  # Max epochs
        callbacks=callbacks,
        use_base_callbacks=False
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
        'sgrna_scorer/resources/crispron.clean.csv',
        invert_targets=False
    )
    
    print("\nIndel data shapes:")
    print(f"X_train_indel: {X_train_indel.shape}")
    print(f"y_train_indel: {y_train_indel.shape}")
    print(f"X_val_indel: {X_val_indel.shape}")
    print(f"y_val_indel: {y_val_indel.shape}")
    
    # # Train on normalized indel data
    # print("\n=== Training on normalized indel data ===")
    # norm_indel_metrics = train_fresh_ko_model(
    #     X_train_indel, y_train_indel,
    #     X_val_indel, y_val_indel
    # )
    
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
    
    # # Train on normalized KO data
    # print("\n=== Training on normalized KO data ===")
    # norm_ko_metrics = train_fresh_ko_model(
    #     X_train_ko, y_train_ko,
    #     X_val_ko, y_val_ko
    # )
    
    # # Print normalized model results
    # print("\n=== Normalized Model Results ===")
    # print("\nNormalized Indel Model Metrics:")
    # print(norm_indel_metrics)
    # print("\nNormalized KO Model Metrics:")
    # print(norm_ko_metrics)
    
    # Train fresh model for comparison
    print("\n=== Training Fresh Model (No Transfer Learning) ===")
    fresh_metrics = train_fresh_ko_model(
        X_train_ko, y_train_ko,
        X_val_ko, y_val_ko
    )
    
    # Now do transfer learning with normalized data
    print("\n=== Starting Transfer Learning on Normalized Data ===")
    args = parse_args()
    
    # Create models for transfer learning
    print("\nCreating models for transfer learning...")
    base_model, ko_model = create_transfer_learning_models(input_shape=(20,))
    
    # Plot the model diagrams with expanded nested models
    plot_model_diagram(base_model, filename='base_model_diagram.png')
    plot_model_diagram(ko_model, filename='ko_model_diagram.png')
    
    # Build models with dummy input
    dummy_input = tf.zeros((1, 20))
    _ = ko_model(dummy_input)
    
    # Train or load base model on normalized indel data
    weights_path = 'sgrna_scorer/resources/pretrained_base_weights'
    if not train_or_load_base_model(base_model, 
                                  X_train_indel, y_train_indel,
                                  X_val_indel, y_val_indel,
                                  weights_path, args.force_train):
        print("Error: Failed to train or load base model")
        sys.exit(1)
    
    # Fine-tune on normalized KO data
    print("\nStarting KO model fine-tuning...")
    
    # Initial training with frozen base
    base_model.trainable = False
    print("Base model weights are initially frozen")
    
    ko_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),  # Initial learning rate
        loss='mse',
        metrics=['mae']
    )
    
    print("\nKO model architecture:")
    ko_model.summary()
    
    # Define transfer_callbacks before using them
    transfer_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'sgrna_scorer/resources/checkpoints/ko_model_transfer_{epoch:02d}.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/ko_model_transfer_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=5,  # Reduced from 20 to 5
            min_lr=1e-5,
            min_delta=0.001
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            min_delta=0.001
        )
    ]
    
    # Train with frozen base
    trained_ko_model, ko_history = train_model(
        ko_model,
        X_train_ko, y_train_ko,
        X_val_ko, y_val_ko,
        model_name="ko_model_transfer_initial",
        batch_size=64,
        epochs=250,  # Initial training with frozen base
        callbacks=transfer_callbacks,
        use_base_callbacks=False
    )
    
    # Gradual unfreezing
    print("\nStarting unfreezing all layers at once...")
    base_model.trainable = True  # Unfreeze the entire model
    ko_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),  # Lower learning rate for fine-tuning
        loss='mse',
        metrics=['mae']
    )
    
    # Train with the entire model unfrozen
    trained_ko_model, ko_history = train_model(
        ko_model,
        X_train_ko, y_train_ko,
        X_val_ko, y_val_ko,
        model_name="ko_model_transfer_final",
        batch_size=64,
        epochs=250,  
        callbacks=transfer_callbacks,
        use_base_callbacks=False
    )
    
    # Final evaluation
    if trained_ko_model is not None:
        print("\nFinal evaluation after gradual unfreezing:")
        transfer_metrics = evaluate_model(trained_ko_model, X_val_ko, y_val_ko)
        print("\nTransfer learning KO validation metrics:", transfer_metrics)
        plot_predictions(trained_ko_model, X_val_ko, y_val_ko, 
                        model_name="ko_model_transfer_final")
        
        # Compare approaches
        print("\n=== Final Comparison ===")
        print("\nFresh Model (No Transfer Learning):")
        print(fresh_metrics)
        print("\nTransfer Learning Model:")
        print(transfer_metrics)
        
        # Calculate improvement
        if fresh_metrics and transfer_metrics:
            mse_improvement = (fresh_metrics['mse'] - transfer_metrics['mse']) / fresh_metrics['mse'] * 100
            mae_improvement = (fresh_metrics['mae'] - transfer_metrics['mae']) / fresh_metrics['mae'] * 100
            corr_improvement = (transfer_metrics['spearman_corr'] - fresh_metrics['spearman_corr']) / abs(fresh_metrics['spearman_corr']) * 100
            
            print("\nTransfer Learning Improvement:")
            print(f"MSE: {mse_improvement:.1f}%")
            print(f"MAE: {mae_improvement:.1f}%")
            print(f"Spearman correlation: {corr_improvement:.1f}%")

if __name__ == "__main__":
    main()