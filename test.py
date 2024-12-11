import os
import sys
import argparse
from datetime import datetime

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.6.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import tensorflow as tf
from sgrna_scorer.data.preprocessing import load_and_preprocess_data
from sgrna_scorer.models.multi_path_atn_model import create_model
from sgrna_scorer.utils.training import (train_model, evaluate_model, 
                                       save_model_weights, plot_predictions)

# Global metrics dictionary
all_metrics = {}

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

def main():
    global all_metrics
    
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
    
    # Create and train a model directly on KO data
    print("\nCreating and training a model directly on KO data...")
    model = create_model(input_shape=(20,))
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Add callbacks for training stability
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            min_delta=0.001
        )
    ]
    
    # Train the model
    trained_model, model_history = train_model(
        model,
        X_train_ko, y_train_ko,
        X_val_ko, y_val_ko,
        model_name="direct_model",
        batch_size=32,
        epochs=100,
        callbacks=callbacks
    )
    
    if trained_model is not None:
        print("\nEvaluating direct model:")
        direct_metrics = evaluate_model(trained_model, X_val_ko, y_val_ko)
        all_metrics['direct'] = direct_metrics
        print("\nDirect model validation metrics:", direct_metrics)
        plot_predictions(trained_model, X_val_ko, y_val_ko, model_name="direct_model")
        
        model_weights_path = 'sgrna_scorer/resources/direct_model'
        save_model_weights(trained_model, model_weights_path)
        print(f"\nSaved direct model weights to {model_weights_path}")
    
    return direct_metrics

def pretrain_base_model():
    global all_metrics
    
    # Load initial dataset
    X_train, X_val, y_train, y_val, _, _, _ = load_and_preprocess_data(
        'sgrna_scorer/resources/DeepHF.clean.csv'
    )
    
    print("\nPre-training base model on DeepHF data...")
    base_model = create_model(input_shape=(20,))
    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    # Train the base model
    history = base_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32
    )
    
    # Evaluate base model
    base_metrics = base_model.evaluate(X_val, y_val)
    all_metrics['base'] = {'mse': base_metrics[0], 'mae': base_metrics[1]}
    
    print("\nBase model metrics on DeepHF data:")
    print(f"MSE: {base_metrics[0]:.4f}")
    print(f"MAE: {base_metrics[1]:.4f}")
    
    # Save the base model weights
    os.makedirs('sgrna_scorer/resources', exist_ok=True)
    base_model.save_weights('sgrna_scorer/resources/base_model_weights.weights.h5')
    return base_metrics

def finetune_model():
    global all_metrics
    
    # Load target dataset
    X_train, X_val, y_train, y_val, _, _, _ = load_and_preprocess_data(
        'sgrna_scorer/resources/pt_sat_guides.csv'
    )
    
    print("\nFine-tuning model on KO data...")
    model = create_model(input_shape=(20,))
    model.load_weights('sgrna_scorer/resources/base_model_weights.weights.h5')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='mse',
        metrics=['mae']
    )
    
    # Fine-tune the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32
    )
    
    # Evaluate fine-tuned model
    finetuned_metrics = model.evaluate(X_val, y_val)
    all_metrics['finetuned'] = {'mse': finetuned_metrics[0], 'mae': finetuned_metrics[1]}
    
    print("\nFine-tuned model metrics on KO data:")
    print(f"MSE: {finetuned_metrics[0]:.4f}")
    print(f"MAE: {finetuned_metrics[1]:.4f}")
    
    # Save the fine-tuned model weights
    model.save_weights('sgrna_scorer/resources/fine_tuned_model_weights.weights.h5')
    return finetuned_metrics

if __name__ == "__main__":
    # Train all models and collect metrics
    direct_metrics = main()
    
    # Pre-training and fine-tuning
    print("\n=== Starting Transfer Learning Pipeline ===")
    base_metrics = pretrain_base_model()
    finetuned_metrics = finetune_model()
    
    # Print comprehensive performance summary
    print("\n=== Final Performance Summary ===")
    print("\nDirect Training on KO Data:")
    print(f"MSE: {all_metrics['direct']['mse']:.4f}")
    print(f"MAE: {all_metrics['direct']['mae']:.4f}")
    print(f"Spearman correlation: {all_metrics['direct']['spearman_corr']:.4f}")
    
    print("\nBase Model on DeepHF Data:")
    print(f"MSE: {all_metrics['base']['mse']:.4f}")
    print(f"MAE: {all_metrics['base']['mae']:.4f}")
    
    print("\nFine-tuned Model on KO Data:")
    print(f"MSE: {all_metrics['finetuned']['mse']:.4f}")
    print(f"MAE: {all_metrics['finetuned']['mae']:.4f}")
    
    # Calculate improvements
    mse_improvement = ((all_metrics['direct']['mse'] - all_metrics['finetuned']['mse']) / 
                      all_metrics['direct']['mse'] * 100)
    mae_improvement = ((all_metrics['direct']['mae'] - all_metrics['finetuned']['mae']) / 
                      all_metrics['direct']['mae'] * 100)
    
    print("\nTransfer Learning Improvement:")
    print(f"MSE Improvement: {mse_improvement:.1f}%")
    print(f"MAE Improvement: {mae_improvement:.1f}%")