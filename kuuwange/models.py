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
    monitor='rmse', verbose=1, mode='auto',
    baseline=None, restore_best_weights=True, patience=12
  )
  mc = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_path,  save_best_only=True
  )

  rlr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='rmse', factor=0.1, patience=10, verbose=2,
    mode='auto'
  )
  csv_logger = tf.keras.callbacks.CSVLogger(log_path)
  return [es, mc, rlr, csv_logger]





# TODO : Concat Models into One

def concatModel(tree_seed, input_shape=(16,)):
  Model_GBT = gradientBoostingModel(tree_seed)
  Model_RFR = randomForstRegressionModel(tree_seed)
  Model_rs = RegressionModel(input_shape=(input_shape[0]+2,))


  input_layer = layers.Input(shape=input_shape)

  output_gbt = Model_GBT(input_layer)
  output_gbt = layers.Reshape((1,))(output_gbt)
  output_rfr = Model_RFR(input_layer)
  output_rfr = layers.Reshape((1,))(output_rfr)
  output_layers = layers.concatenate([input_layer, output_gbt, output_rfr])
  output_layers = layers.Reshape((input_shape[0]+2,))(output_layers)

  Model_Concated = keras.Model(inputs=input_layer, outputs=Model_rs(output_layers))
  return [Model_GBT, Model_RFR, Model_rs, Model_Concated]





class RegressionModel(keras.Model):
  def __init__(self, input_shape=(16,), **kwargs):
    super(RegressionModel, self).__init__(**kwargs)
    self.input_layer = layers.InputLayer(input_shape=input_shape)
    self.dense_1 = layers.Dense(128, activation='relu')
    self.dense_2 = layers.Dense(256, activation='relu')
    self.dense_3 = layers.Dense(512, activation='relu')
    self.dense_4= layers.Dense(128, activation='relu')



    self.outputL = layers.Dense(1, activation='linear')



  def call(self, inputs, training=False):
    x = self.input_layer(inputs)

    # NOTE : demansion epand
    x = tf.reshape(x, [-1, 16])
    if training:
      x = tf.nn.dropout(x, 0.2)
    x = self.dense_1(x)
    x = self.dense_2(x)
    x = self.dense_3(x)
    x = self.dense_4(x)


    return self.outputL(x)


def gradientBoostingModel(seed):
  model = tfdf.keras.GradientBoostedTreesModel(
    task=Task.REGRESSION,
    random_seed=seed,
    num_trees=32,
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
    num_trees=32,
    # num_threads=4,
    max_depth=10,
  )
  model.compile(
    metrics=["mse", "mae"],
  )
  return model

