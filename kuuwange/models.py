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
  def __init__(self):
    super().__init__()
    self.dense1 = layers.Dense(128, activation = 'relu')
    self.dense2 = layers.Dense(64, activation = 'relu')
    self.dense3 = layers.Dense(32, activation = 'relu')
    self.dense4 = layers.Dense(1, activation = 'relu')



  def call(self, inputs, training = False):
    x = self.dense1(inputs)
    x = self.dense2(x)
    if training:
      x = layers.Dropout(0.2)(x)
    x = self.dense3(x)
    x = self.dense4(x)
    return layers.Dense(1, activation = 'relu')(x)

