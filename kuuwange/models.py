import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow_decision_forests as tfdf
from tensorflow_decision_forests.keras import Task

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

class TSBaseModel(keras.Model):
  def __init__(self, **kwargs):
    super(TSBaseModel, self).__init__(**kwargs)
    self.input_layer = layers.InputLayer(input_shape=(None,13))
    self.lstm_1 = layers.LSTM(13, return_sequences=True)
    self.lstm_2 = layers.LSTM(6)
    self.dense_1 = layers.Dense(32, activation='relu')
    self.dense_2 = layers.Dense(16, activation='relu')
    self.outputL = layers.Dense(1)

  def call(self, inputs, training=False):
    x = self.input_layer(inputs)
    # NOTE : demansion epand
    x = tf.expand_dims(x, axis=0)
    x = self.lstm_1(x)
    if training:
      x = tf.nn.dropout(x, 0.2)
    x = self.lstm_2(x)
    if training:
      x = tf.nn.dropout(x, 0.2)
    x = self.dense_1(x)
    x = self.dense_2(x)


    return self.outputL(x)

def gradientBoostingModel(seed):
  model_GBT = tfdf.keras.GradientBoostedTreesModel(
    task=Task.REGRESSION,
    random_seed=seed,
    num_trees=10,
    num_threads=4,
    max_depth=10,
  )
  return model_GBT

def randomForstRegressionModel(seed):
  model = tfdf.keras.RandomForestModel(
    task=Task.REGRESSION,
    random_seed=seed,
    num_trees=10,
    num_threads=4,
    max_depth=10,
  )
  return model
