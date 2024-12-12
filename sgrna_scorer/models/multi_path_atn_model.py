import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(sequence_input_shape, num_filters=128, num_dense_neurons=64, dropout_rate=0.4, use_feature=False):
    """Create a model for sgRNA scoring with one additional feature.
    
    Args:
        sequence_input_shape: Shape of the input sequence
        num_filters: Number of filters in convolutional layers
        num_dense_neurons: Number of neurons in dense layers
        dropout_rate: Dropout rate for regularization
        use_feature: Boolean to control whether to include the feature
    """
    
    # Main sequence input
    sequence_input = layers.Input(shape=sequence_input_shape, name='sequence_input')
    print(f"Sequence input shape: {sequence_input_shape}")
    
    # Initial embedding for sequence
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
    print(f"Concatenated shape before feature input: {concatenated.shape}")
    
    if use_feature:
        # Single feature input
        feature_input = layers.Input(shape=(1,), name='feature_input')
        print("Feature input shape: (1,)")
        
        # Concatenate sequence features with raw feature
        concatenated = layers.Concatenate()([concatenated, feature_input])
        print(f"Concatenated shape after adding feature input: {concatenated.shape}")
        model_inputs = [sequence_input, feature_input]
    else:
        model_inputs = sequence_input
    
    # Dense layers processing all features together
    x = layers.Dense(num_dense_neurons, activation='relu', 
                     kernel_initializer='he_normal')(concatenated)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    output = layers.Dense(1, activation='linear')(x)
    
    return models.Model(inputs=model_inputs, outputs=output, name='model')