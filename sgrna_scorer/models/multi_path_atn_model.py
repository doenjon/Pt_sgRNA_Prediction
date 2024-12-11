import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_filters=128, num_dense_neurons=64, dropout_rate=0.4):
    """Create a model for sgRNA scoring with three convolutional paths."""
    
    input_sequence = layers.Input(shape=input_shape)
    
    # Initial embedding
    x = layers.Embedding(input_dim=5, output_dim=32)(input_sequence)
    
    # Path 1: Small motifs
    path1 = layers.Conv1D(num_filters, kernel_size=3, activation='relu', 
                          kernel_initializer='he_normal', padding='same')(x)
    path1 = layers.GlobalMaxPooling1D()(path1)
    path1 = layers.Dropout(dropout_rate)(path1)
    
    # Path 2: Larger motifs
    path2 = layers.Conv1D(num_filters, kernel_size=5, activation='relu', 
                          kernel_initializer='he_normal', padding='same')(x)
    path2 = layers.GlobalMaxPooling1D()(path2)
    path2 = layers.Dropout(dropout_rate)(path2)
    
    # Path 3: Position-specific motifs
    path3 = layers.Conv1D(num_filters, kernel_size=5, activation='relu', 
                          kernel_initializer='he_normal', padding='same')(x)
    path3 = layers.GlobalMaxPooling1D()(path3)
    path3 = layers.Dropout(dropout_rate)(path3)
    
    # Concatenate paths
    concatenated = layers.Concatenate()([path1, path2, path3])
    
    # Dense layers
    x = layers.Dense(num_dense_neurons, activation='relu', 
                     kernel_initializer='he_normal')(concatenated)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    output = layers.Dense(1, activation='linear')(x)
    
    return models.Model(inputs=input_sequence, outputs=output, name='model')