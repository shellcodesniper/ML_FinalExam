import tensorflow as tf
tf.config.set_soft_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

import kuuwange.models as Model
import kuuwange as MY
from kuuwange.loaders import Loaders
import time


# NOTE : 기존에 사용하던, Callbacks


def main():

  tree_seed = 9932 # NOTE : 같은 결과를 얻기 위해
  model_GBT = Model.gradientBoostingModel(tree_seed)

  model_RFR = Model.randomForstRegressionModel(tree_seed)

  train_generator = Loaders(True).as_generator(batch_size=10000, shuffle=True) # TYPE : Train Dataset Generator
  test_generator = Loaders(False).as_generator(batch_size=10000, shuffle=True) # TYPE : Test Dataset Generator

  for data in train_generator:
    [x_train, y_train] = data
    result_GBT = model_GBT.fit(x_train, y_train, verbose=1, callbacks=Model.get_callback('gbt'))
    result_RFR = model_RFR.fit(x_train, y_train, verbose=1, callbacks=Model.get_callback('rfr'))

    print (result_GBT["mse"], result_RFR["mse"])
    time.sleep(2)









if __name__ == "__main__":
  main()
