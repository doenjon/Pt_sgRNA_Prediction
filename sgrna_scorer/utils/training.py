import tensorflow as tf
from datetime import datetime

def create_callbacks():
    """Create training callbacks for monitoring and checkpointing."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/model_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    return callbacks

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """Train the model with early stopping and logging.
    
    Args:
        model: Compiled Keras model
        X_train: Training sequences
        y_train: Training scores
        X_val: Validation sequences
        y_val: Validation scores
        batch_size: Training batch size
        epochs: Maximum number of epochs
        
    Returns:
        tuple: Trained model and training history
    """
    callbacks = create_callbacks()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_val, y_val):
    """Evaluate model performance on validation data.
    
    Args:
        model: Trained Keras model
        X_val: Validation sequences
        y_val: Validation scores
        
    Returns:
        dict: Evaluation metrics
    """
    results = model.evaluate(X_val, y_val, verbose=0)
    metrics = {
        'mse': results[0],
        'mae': results[1]
    }
    return metrics