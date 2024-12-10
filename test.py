import os
import sys
import argparse

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
    
    # Build models with dummy input
    dummy_input = tf.zeros((1, 20))
    _ = ko_model(dummy_input)
    
    if not args.skip_indel:
        # Load DeepHF data for pre-training (no inversion)
        print("Loading DeepHF data for pre-training...")
        X_train_indel, X_val_indel, y_train_indel, y_val_indel, _ = load_and_preprocess_data(
            'sgrna_scorer/resources/DeepHF.clean.csv',
            invert_targets=False
        )
        
        print("\nIndel data shapes:")
        print(f"X_train_indel: {X_train_indel.shape}")
        print(f"y_train_indel: {y_train_indel.shape}")
        print(f"X_val_indel: {X_val_indel.shape}")
        print(f"y_val_indel: {y_val_indel.shape}")
        
        # Train or load base model
        weights_path = 'sgrna_scorer/resources/pretrained_base_weights'
        if not train_or_load_base_model(base_model, indel_model, 
                                      X_train_indel, y_train_indel,
                                      X_val_indel, y_val_indel,
                                      weights_path, args.force_train):
            print("Error: Failed to train or load base model")
            sys.exit(1)
    else:
        # Just load pre-trained weights
        weights_path = 'sgrna_scorer/resources/pretrained_base_weights'
        if not load_model_weights(base_model, weights_path):
            print("Error: Could not load pre-trained weights and --skip-indel was specified")
            sys.exit(1)
    
    # Load KO data with target inversion
    print("\nLoading KO data for fine-tuning...")
    X_train_ko, X_val_ko, y_train_ko, y_val_ko, ko_normalizer = load_and_preprocess_data(
        'sgrna_scorer/resources/pt_sat_guides.csv',
        invert_targets=True
    )
    
    print("\nKO data shapes:")
    print(f"X_train_ko: {X_train_ko.shape}")
    print(f"y_train_ko: {y_train_ko.shape}")
    print(f"X_val_ko: {X_val_ko.shape}")
    print(f"y_val_ko: {y_val_ko.shape}")
    
    # Train fresh model for comparison
    fresh_metrics = train_fresh_ko_model(X_train_ko, y_train_ko, X_val_ko, y_val_ko)
    
    # Fine-tune transfer learning model
    print("\nStarting KO model fine-tuning (transfer learning)...")
    if not args.no_freeze:
        base_model.trainable = False
        print("Base model weights are frozen")
    else:
        print("Base model weights are trainable")
    
    ko_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='mse',
        metrics=['mae']
    )
    
    print("\nKO model architecture:")
    ko_model.summary()
    
    trained_ko_model, ko_history = train_model(
        ko_model,
        X_train_ko, y_train_ko,
        X_val_ko, y_val_ko,
        model_name="ko_model",
        batch_size=16,
        epochs=30
    )
    
    if trained_ko_model is not None:
        print("\nEvaluating initial KO model:")
        ko_metrics = evaluate_model(trained_ko_model, X_val_ko, y_val_ko, ko_normalizer)
        print("\nInitial KO validation metrics:", ko_metrics)
        plot_predictions(trained_ko_model, X_val_ko, y_val_ko, model_name="ko_model_initial", normalizer=ko_normalizer)
        
        # Gradual unfreezing and continued training
        print("\nStarting gradual unfreezing and continued training...")
        
        # Get all layers from the base model
        base_layers = [layer for layer in base_model.layers if 'batch_normalization' not in layer.name.lower()]
        
        # Gradually unfreeze layers from top to bottom
        for i in range(len(base_layers) - 1, -1, -1):
            print(f"\nUnfreezing layer: {base_layers[i].name}")
            base_layers[i].trainable = True
            
            # Recompile with lower learning rate
            ko_model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),  # Even lower learning rate for fine-tuning
                loss='mse',
                metrics=['mae']
            )
            
            # Train for a few epochs with unfrozen layer
            trained_ko_model, ko_history = train_model(
                ko_model,
                X_train_ko, y_train_ko,
                X_val_ko, y_val_ko,
                model_name=f"ko_model_unfreeze_{i}",
                batch_size=16,
                epochs=30  # Fewer epochs per layer
            )
            
            # Evaluate after each unfreezing
            if trained_ko_model is not None:
                print(f"\nEvaluating model after unfreezing {base_layers[i].name}:")
                ko_metrics = evaluate_model(trained_ko_model, X_val_ko, y_val_ko, ko_normalizer)
                print(f"\nValidation metrics after unfreezing {base_layers[i].name}:", ko_metrics)
        
        # Final evaluation and saving
        print("\nFinal evaluation after gradual unfreezing:")
        final_metrics = evaluate_model(trained_ko_model, X_val_ko, y_val_ko, ko_normalizer)
        print("\nFinal KO validation metrics:", final_metrics)
        plot_predictions(trained_ko_model, X_val_ko, y_val_ko, model_name="ko_model_final", normalizer=ko_normalizer)
        
        # Compare results
        print("\nModel Comparison:")
        print("Fresh model (no transfer learning):")
        print(fresh_metrics)
        print("\nTransfer learning model:")
        print(final_metrics)
        
        # Calculate improvement
        if fresh_metrics and final_metrics:
            mse_improvement = (fresh_metrics['mse'] - final_metrics['mse']) / fresh_metrics['mse'] * 100
            mae_improvement = (fresh_metrics['mae'] - final_metrics['mae']) / fresh_metrics['mae'] * 100
            corr_improvement = (final_metrics['spearman_corr'] - fresh_metrics['spearman_corr']) / abs(fresh_metrics['spearman_corr']) * 100
            
            print("\nImprovement with transfer learning:")
            print(f"MSE: {mse_improvement:.1f}%")
            print(f"MAE: {mae_improvement:.1f}%")
            print(f"Spearman correlation: {corr_improvement:.1f}%")
        
        # Save final model
        ko_weights_path = 'sgrna_scorer/resources/ko_model_final'
        save_model_weights(ko_model, ko_weights_path)
        print(f"\nSaved final fine-tuned KO model weights to {ko_weights_path}")

if __name__ == "__main__":
    main()