import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)
from tensorflow import keras
import tensorflow.keras.layers as layers

class TSBaseModel(keras.Model):
  def __init__(self, **kwargs):
    super(TSBaseModel, self).__init__(**kwargs)
    self.input_layer = layers.InputLayer(input_shape=(13,))
    self.lstms = [
      layers.LSTM(64, return_sequences=True),
      layers.LSTM(64, return_sequences=True),
    ],
    self.dense_1 = layers.Dense(64, activation='relu')
    self.dense_2 = layers.Dense(10)

  def call(self, inputs):
    x = self.input_layer(inputs)
    x = self.lstms[0](x)
    x = self.lstms[1](x)
    x = self.dense_1(inputs)
    return self.dense_2(x)
