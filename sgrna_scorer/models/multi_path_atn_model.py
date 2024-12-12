import tensorflow as tf
from tensorflow.keras import layers, models

class DebugLayer(layers.Layer):
    def __init__(self, name):
        super().__init__()
        self.layer_name = name
    
    def call(self, inputs):
        tf.print(f"\n{self.layer_name} stats:", 
                "\nmin:", tf.reduce_min(inputs),
                "\nmax:", tf.reduce_max(inputs),
                "\nmean:", tf.reduce_mean(inputs),
                "\nstd:", tf.math.reduce_std(inputs))
        return inputs

def create_base_feature_extractor(input_shape, num_filters=32, input_dim=5, 
                                kernel_size=5, pool_size=2, dropout_rate=0.4):
    """Create a simplified base feature extractor."""
    
    input_sequence = layers.Input(shape=input_shape)
    
    # Initial normalization and embedding
    x = layers.Normalization()(input_sequence)
    x = layers.Embedding(input_dim=input_dim, output_dim=44, input_length=input_shape[0])(x)
    
    # Single convolution block
    x = layers.Conv1D(num_filters, kernel_size, activation='relu', 
                     kernel_initializer='he_normal')(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Flatten for output
    features = layers.Flatten()(x)
    
    # Add a dense layer to produce a single output
    output = layers.Dense(1, activation='linear')(features)
    
    # Create two models: one for training and one for feature extraction
    training_model = models.Model(inputs=input_sequence, outputs=output, name='base_model')
    feature_model = models.Model(inputs=input_sequence, outputs=features, name='feature_extractor')
    
    return training_model, feature_model

def create_ko_specific_path(input_shape, num_filters=64, input_dim=5,
                          kernel_size=5, pool_size=2, dropout_rate=0.4):
    """Create a fresh path for learning KO-specific features."""
    
    input_sequence = layers.Input(shape=input_shape)
    
    # Initial normalization and embedding
    x = layers.Normalization()(input_sequence)
    x = layers.Embedding(input_dim=input_dim, output_dim=32, input_length=input_shape[0])(x)
    
    # Convolutional layers
    x = layers.Conv1D(num_filters, kernel_size, activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(num_filters, kernel_size, activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Flatten for output
    features = layers.Flatten()(x)
    
    return models.Model(inputs=input_sequence, outputs=features, name='ko_specific_path')

def create_dual_path_model(base_feature_model, input_shape, num_dense_neurons=128, dropout_rate=0.4):
    """Create a model with parallel pre-trained and fresh paths."""
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Pre-trained path (frozen initially)
    pretrained_features = base_feature_model(inputs)  # Get features directly
    
    # Fresh KO-specific path
    ko_path = create_ko_specific_path(input_shape)
    ko_features = ko_path(inputs)
    
    # Combine features from both paths
    combined_features = layers.Concatenate()([pretrained_features, ko_features])
    
    # Dense layers for final prediction
    x = layers.Dense(num_dense_neurons, activation='relu',
                    kernel_initializer='he_normal')(combined_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(num_dense_neurons // 2, activation='relu',
                    kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='linear')(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name='dual_path_model')

def create_transfer_learning_models(input_shape, num_filters=256, input_dim=5,
                                  kernel_size=5, pool_size=2, num_dense_neurons=128,
                                  dropout_rate=0.4):
    """Create all models needed for transfer learning."""
    
    # Create base feature extractor and feature model
    base_model, feature_model = create_base_feature_extractor(
        input_shape=input_shape,
        num_filters=num_filters,
        input_dim=input_dim,
        kernel_size=kernel_size,
        pool_size=pool_size,
        dropout_rate=dropout_rate
    )
    
    # Create separate KO model with dual path
    ko_model = create_dual_path_model(
        base_feature_model=feature_model,  # Pass the feature model
        input_shape=input_shape,
        num_dense_neurons=num_dense_neurons,
        dropout_rate=dropout_rate
    )
    
    return base_model, ko_model