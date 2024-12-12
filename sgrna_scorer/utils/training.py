import tensorflow as tf
from datetime import datetime
import os
from scipy.stats import spearmanr


def create_callbacks(model_name):
    """Create training callbacks for monitoring and checkpointing."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create directories if they don't exist
    checkpoint_dir = 'sgrna_scorer/resources/checkpoints'
    log_dir = 'logs'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'{checkpoint_dir}/{model_name}_{timestamp}.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'{log_dir}/{model_name}_{timestamp}',
            histogram_freq=1
        )
    ]
    return callbacks

import matplotlib.pyplot as plt

def plot_training_history(history, model_name):
    """Plot and save the training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    
    # Save the plot as a PNG file
    plot_path = f'logs/{model_name}_training_history.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training history plot saved to {plot_path}")

def train_model(model, train_data, val_data, batch_size=32, epochs=100, use_feature=False, additional_callbacks=None, fold_num=None):
    """Train the model with optional feature data."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
    ]
    
    if additional_callbacks:
        callbacks.extend(additional_callbacks)
    
    if use_feature:
        X_train, feature_train, y_train = train_data
        X_val, feature_val, y_val = val_data
        
        print(f"Training data shapes: X_train: {X_train.shape}, feature_train: {feature_train.shape}, y_train: {y_train.shape}")
        print(f"Validation data shapes: X_val: {X_val.shape}, feature_val: {feature_val.shape}, y_val: {y_val.shape}")
        
        history = model.fit(
            [X_train, feature_train],
            y_train,
            validation_data=([X_val, feature_val], y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
    else:
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        print(f"Training data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Validation data shapes: X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
    
    return history

def plot_predictions(model, X_val, y_val, model_name="model", normalizer=None):
    """Plot predicted vs actual values in normalized space."""
    if isinstance(X_val, list):  # If features are included
        val_inputs = {
            'sequence_input': X_val[0],
            'feature_input': X_val[1]
        }
        predictions = model.predict(val_inputs).flatten()
    else:  # No features
        predictions = model.predict({'sequence_input': X_val}).flatten()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val, predictions, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], 
             [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('Actual Values (Normalized)')
    plt.ylabel('Predicted Values (Normalized)')
    plt.title('Predicted vs Actual Values (Normalized Space)')
    plt.grid(True)
    
    # Print value ranges
    print("\nValue ranges in plot (normalized space):")
    print(f"Actual values range: {y_val.min():.3f} to {y_val.max():.3f}")
    print(f"Predicted values range: {predictions.min():.3f} to {predictions.max():.3f}")
    
    # Save the plot
    plot_path = f'logs/{model_name}_predictions_vs_actuals.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Predicted vs Actual plot saved to {plot_path}")


def evaluate_model(model, X_val, y_val, normalizer=None):
    """Evaluate model performance on validation data."""
    if isinstance(X_val, list):
        val_inputs = {
            'sequence_input': X_val[0],
            'feature_input': X_val[1]
        }
        print(f"Evaluation input shapes: {val_inputs['sequence_input'].shape}, {val_inputs['feature_input'].shape}")
        predictions = model.predict(val_inputs).flatten()
        results = model.evaluate(val_inputs, y_val, verbose=0)
    else:
        print(f"Evaluation input shape: {X_val.shape}")
        predictions = model.predict({'sequence_input': X_val}).flatten()
        results = model.evaluate({'sequence_input': X_val}, y_val, verbose=0)
    
    metrics = {
        'mse': results[0],
        'mae': results[1]
    }
    
    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr(y_val, predictions)
    metrics['spearman_corr'] = spearman_corr
    
    # Sample predictions in normalized space
    print("\nSample predictions vs actual (in normalized space):")
    for pred, actual in zip(predictions[:5], y_val[:5]):
        print(f"Predicted: {pred:.3f}, Actual: {actual:.3f}")
    
    return metrics

def save_model_weights(model, filepath):
    """Save model weights with appropriate directory creation."""
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)
    model.save_weights(f"{filepath}.weights.h5") 

def load_model_weights(model, filepath):
    """Load model weights from a file."""
    try:
        model.load_weights(f"{filepath}.weights.h5")
        print(f"Successfully loaded weights from {filepath}.weights.h5")
        return True
    except Exception as e:
        print(f"Error loading weights from {filepath}.weights.h5: {str(e)}")
        return False 