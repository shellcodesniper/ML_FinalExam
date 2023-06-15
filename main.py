import tensorflow as tf
import warnings
 
warnings.filterwarnings(action='ignore')
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

  (predict_x, _)= Loaders(False).as_raw_set()
  train_generator = Loaders(True).as_generator(batch_size=100000, shuffle=True) # TYPE : Train Dataset Generator
  (x_test, y_test) = Loaders(True).get_validation_set() # TYPE : Train Dataset Generator

  for data in train_generator:
    [x_train, y_train] = data
    # TODO : Train
    model_GBT.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('gbt'))
    model_RFR.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('rfr'))

    # INFO : Summary
    # model_GBT.summary()
    # model_RFR.summary()

    # TODO : Evalulate
    result_GBT = model_GBT.evaluate(x_test, y_test, return_dict=True)
    result_RFR = model_RFR.evaluate(x_test, y_test, return_dict=True)

    print("=====================================")
    print(f"Result GBT : {result_GBT}")
    print(f"Result RFR : {result_RFR}")
    print("=====================================")

    predict_y_GBT = model_GBT.predict(predict_x)
    predict_y_RFR = model_RFR.predict(predict_x)

    print (predict_y_GBT)
    print (predict_y_RFR)










if __name__ == "__main__":
  main()
