import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_filters=128, num_dense_neurons=64, dropout_rate=0.4, num_heads=4):
    """Create a simplified model for sgRNA scoring with convolution and multi-head attention."""
    
    input_sequence = layers.Input(shape=input_shape)
    
    # Initial normalization and embedding
    x = layers.Normalization()(input_sequence)
    x = layers.Embedding(input_dim=5, output_dim=32)(x)
    
    # Convolutional layer to capture motifs
    x = layers.Conv1D(num_filters, kernel_size=5, activation='relu', 
                      kernel_initializer='he_normal', padding='same')(x)
    
    # Multi-head attention
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_filters)(x, x)
    
    # Global pooling
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense layers
    x = layers.Dense(num_dense_neurons, activation='relu', 
                     kernel_initializer='he_normal')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    output = layers.Dense(1, activation='linear')(x)
    
    return models.Model(inputs=input_sequence, outputs=output, name='model')