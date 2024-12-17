import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

class MultiPathAttnModel(tf.keras.Model):
    def __init__(self, use_extra_feature=True):
        super(MultiPathAttnModel, self).__init__()
        self.use_extra_feature = use_extra_feature
        
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(5, 256)
        
        # First conv path
        self.conv1 = tf.keras.layers.Conv1D(256, 5, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling1D(2)
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.conv2 = tf.keras.layers.Conv1D(256, 5, padding='same', activation='relu')
        
        # Second conv path
        self.conv3 = tf.keras.layers.Conv1D(256, 5, padding='same', activation='relu')
        
        # Attention block
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
        # Dense layers with dropouts
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.4)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.4)
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        if self.use_extra_feature:
            x, extra_feature = inputs
        else:
            x = inputs
            
        # Embedding
        x = self.embedding(x)
        
        # First path
        path1 = self.conv1(x)
        path1 = self.pool1(path1)
        path1 = self.dropout1(path1, training=training)
        path1 = self.conv2(path1)
        
        # Second path
        path2 = self.conv3(x)
        
        # Attention mechanism
        attn_output = self.attention(path1, path2, path2)
        attn_output = self.layer_norm(attn_output + path1)
        
        # Flatten and concatenate
        x = self.flatten(attn_output)
        
        # Add extra feature if used
        if self.use_extra_feature:
            x = tf.concat([x, extra_feature], axis=1)
        
        # Dense layers
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        x = self.dropout3(x, training=training)
        x = self.dense3(x)
        
        return self.output_layer(x)

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

def plot_model_diagram():
    """Plot and save the model diagram."""
    model = create_model(sequence_input_shape=(60,), use_feature=False)
    plot_model(model, to_file='multi_path_model.png', show_shapes=True, show_layer_names=True)
    print("Model diagram saved to multi_path_model.png")

# Call this function to generate the model diagram
plot_model_diagram()