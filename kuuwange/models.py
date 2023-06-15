import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow_decision_forests as tfdf
from tensorflow_decision_forests.keras import Task

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# HACK : Callback Functions
def get_callback(name):
  save_path = f"datas/weight_{name}.keras"
  log_path = f"datas/log_{name}.log"

  es = tf.keras.callbacks.EarlyStopping(
    monitor='loss', verbose=1, mode='auto',
    baseline=None, restore_best_weights=True, patience=12
  )
  mc = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_path,  save_best_only=True
  )

  rlr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1, patience=10, verbose=2,
    mode='auto'
  )
  csv_logger = tf.keras.callbacks.CSVLogger(log_path)
  return [es, mc, rlr, csv_logger]









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
  model = tfdf.keras.GradientBoostedTreesModel(
    task=Task.REGRESSION,
    random_seed=seed,
    num_trees=10,
    # num_threads=4,
    max_depth=10,
  )
  model.compile(
    metrics=["mse", "mae"],
  )
  return model

def randomForstRegressionModel(seed):
  model = tfdf.keras.RandomForestModel(
    task=Task.REGRESSION,
    random_seed=seed,
    num_trees=10,
    # num_threads=4,
    max_depth=10,
  )
  model.compile(
    metrics=["mse", "mae"],
  )
  return model
