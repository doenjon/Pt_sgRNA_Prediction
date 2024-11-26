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

def train_model(model, X_train, y_train, X_val, y_val, 
                model_name="model", batch_size=32, epochs=50):
    """Train a model with monitoring and callbacks."""
    callbacks = create_callbacks(model_name)
    
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        plot_training_history(history, model_name)
        
        return model, history
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None

def plot_predictions(model, X_val, y_val, model_name="model"):
    """Plot predicted vs actual values."""
    predictions = model.predict(X_val)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val, predictions, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.grid(True)
    
    # Save the plot as a PNG file
    plot_path = f'logs/{model_name}_predictions_vs_actuals.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Predicted vs Actual plot saved to {plot_path}")


def evaluate_model(model, X_val, y_val):
    """Evaluate model performance on validation data."""
    results = model.evaluate(X_val, y_val, verbose=0)
    metrics = {
        'mse': results[0],
        'mae': results[1]
    }
    
    # Calculate Spearman correlation
    predictions = model.predict(X_val)
    spearman_corr, _ = spearmanr(y_val, predictions)
    metrics['spearman_corr'] = spearman_corr
    
    # Sample predictions
    sample_predictions = predictions[:5]
    print("\nSample predictions vs actual:")
    for pred, actual in zip(sample_predictions, y_val[:5]):
        print(f"Predicted: {pred[0]:.3f}, Actual: {actual:.3f}")
    
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