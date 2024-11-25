import tensorflow as tf
from tensorflow.keras import layers, models

class BaseSequenceModel(tf.keras.Model):
    """Base sgRNA sequence prediction model.
    
    Args:
        input_shape: Tuple defining input sequence dimensions
        num_filters: Number of convolutional filters
        kernel_size: Size of the convolutional kernel
    """
    def __init__(self, input_shape, num_filters=128, kernel_size=5):
        super().__init__()
        self.embedding = layers.Embedding(5, 32, input_length=input_shape[0])
        self.conv1 = layers.Conv1D(num_filters, kernel_size, activation='relu')
        self.pool1 = layers.MaxPooling1D(2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)
    
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.output_layer(x)
