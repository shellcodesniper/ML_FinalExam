import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

class TSBaseModel(keras.Model):
  def __init__(self, **kwargs):
    super(TSBaseModel, self).__init__(**kwargs)
    self.input_layer = layers.InputLayer(input_shape=(None,13))
    self.lstm_1 = layers.LSTM(26, return_sequences=True)
    self.lstm_2 = layers.LSTM(13)
    self.dense_1 = layers.Dense(32, activation='relu')
    self.dense_2 = layers.Dense(16, activation='leaky_relu')
    self.outputL = layers.Dense(1)

  def call(self, inputs):
    x = self.input_layer(inputs)

    # NOTE : demansion epand
    x = tf.expand_dims(x, axis=0)
    x = self.lstm_1(x)
    x = self.lstm_2(x)
    x = self.dense_1(x)
    x = self.dense_2(x)


    return self.outputL(x)
