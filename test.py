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
    
    # Create and train a model
    print("\nCreating and training a model...")
    model = create_model(input_shape=(20,))
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,  # Initial learning rate
        clipnorm=1.0  # Add gradient clipping
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
        model_name="model",
        batch_size=32,
        epochs=100,
        callbacks=callbacks
    )
    
    if trained_model is not None:
        print("\nEvaluating model:")
        model_metrics = evaluate_model(trained_model, X_val_ko, y_val_ko)
        print("\nModel validation metrics:", model_metrics)
        plot_predictions(trained_model, X_val_ko, y_val_ko, model_name="model")
        
        # Save model
        model_weights_path = 'sgrna_scorer/resources/model'
        save_model_weights(trained_model, model_weights_path)
        print(f"\nSaved model weights to {model_weights_path}")

if __name__ == "__main__":
    main()