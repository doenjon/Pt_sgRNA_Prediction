import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(sequence_input_shape, use_feature=False, num_filters=128, num_dense_neurons=64, dropout_rate=0.4):
    """
    Create a model for sgRNA scoring with three convolutional paths and optional position feature.
    
    Args:
        sequence_input_shape: Shape of the sequence input (length,)
        use_feature: Whether to use the position feature
        num_filters: Number of filters in Conv1D layers
        num_dense_neurons: Number of neurons in dense layers
        dropout_rate: Dropout rate
    """
    # Sequence input branch
    sequence_input = layers.Input(shape=sequence_input_shape, name='sequence_input')
    x = layers.Embedding(input_dim=5, output_dim=32)(sequence_input)
    
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
    
    # Concatenate CNN paths
    concatenated = layers.Concatenate()([path1, path2, path3])
    
    if use_feature:
        # Position feature input
        feature_input = layers.Input(shape=(1,), name='feature_input')
        # Combine sequence features with position feature
        concatenated = layers.Concatenate()([concatenated, feature_input])
    
    # Dense layers
    x = layers.Dense(num_dense_neurons, activation='relu', 
                     kernel_initializer='he_normal')(concatenated)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    output = layers.Dense(1, activation='linear')(x)
    
    if use_feature:
        model = models.Model(inputs=[sequence_input, feature_input], outputs=output, name='model_with_feature')
    else:
        model = models.Model(inputs=sequence_input, outputs=output, name='model_without_feature')
    
    return model