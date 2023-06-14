import tensorflow as tf
from tensorflow import keras

class ShopModel(keras.Model):
  def __init__(self, shop_id):
    super(ShopModel, self).__init__()
    self.shop_id = shop_id
    self.dense1 = keras.layers.Dense(64, activation='relu')
    self.dense2 = keras.layers.Dense(64, activation='relu')
    self.dense3 = keras.layers.Dense(64, activation='relu')
    self.dense4 = keras.layers.Dense(1)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    x = self.dense3(x)
    return self.dense4(x)

