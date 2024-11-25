import os
import sys

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.6.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import tensorflow as tf
from sgrna_scorer.data.preprocessing import load_and_preprocess_data
from sgrna_scorer.models.base_model import BaseSequenceModel


# Load data
print("Loading data...")
X_train, X_val, y_train, y_val = load_and_preprocess_data('sgrna_scorer/resources/DeepHF.clean.csv')

print("\nData shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")

# Create model with explicit input shape
print("\nCreating model...")
model = BaseSequenceModel(input_shape=(20,))

# Build model
dummy_input = tf.zeros((1, 20))  # Create dummy input to build model
_ = model(dummy_input)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='mse',
    metrics=['mae']
)

# Print model summary
model.summary()

print("\nStarting training...")
# Train with smaller batch size
try:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=16,  # Reduced batch size
        verbose=1
    )
    
    # Make some predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_val[:5], batch_size=16)
    print("\nSample predictions vs actual:")
    for pred, actual in zip(predictions, y_val[:5]):
        print(f"Predicted: {pred[0]:.3f}, Actual: {actual:.3f}")
        
except Exception as e:
    print(f"Error during training: {str(e)}")