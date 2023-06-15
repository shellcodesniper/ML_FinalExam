import tensorflow as tf
import warnings
 
warnings.filterwarnings(action='ignore')
tf.config.set_soft_device_placement(False)
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
import time, platform, os


# NOTE : 기존에 사용하던, Callbacks

IS_MAC_OS = os.name == 'posix' and platform.system() == 'Darwin'

def main():

  tree_seed = 9932 # NOTE : 같은 결과를 얻기 위해
  model_GBT = Model.gradientBoostingModel(tree_seed)

  model_RFR = Model.randomForstRegressionModel(tree_seed)

  predict_loader = Loaders(False)

  (predict_x, _)= predict_loader.as_raw_set()
  (x_test, y_test) = Loaders(True).get_validation_set() # TYPE : Train Dataset Generator


  # NOTE : Train All Data (Epoch 1)
  (x_train, y_train) = Loaders(True).as_raw_set()

  if IS_MAC_OS:
    x_train = x_train[:10]
    y_train = y_train[:10]
    model_GBT.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('gbt'))
    model_RFR.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('rfr'))

    if os.path.exists('datas/model_GBT.h5'):
      model_GBT.load_weights('datas/model_GBT.h5')
    if os.path.exists('datas/model_RFR.h5'):
      model_RFR.load_weights('datas/model_RFR.h5')
  else:
    model_GBT.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('gbt'))
    model_RFR.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('rfr'))

    # TODO : Summary
    model_GBT.summary()
    model_RFR.summary()

    result_GBT = model_GBT.evaluate(x_test, y_test, return_dict=True)
    result_RFR = model_RFR.evaluate(x_test, y_test, return_dict=True)

    print("=====================================")
    print(f"Result GBT : {result_GBT}")
    print(f"Result RFR : {result_RFR}")
    print("=====================================")

    model_GBT.save_weights('datas/model_GBT.h5')
    model_RFR.save_weights('datas/model_RFR.h5')


  # TODO : Training
  predict_y_GBT = model_GBT.predict(predict_x)
  predict_y_RFR = model_RFR.predict(predict_x)

  restored_x = predict_x
  # restored_x = predict_scaler.inverse_transform(predict_x)

  restored_y_GBT = predict_y_GBT
  restored_y_RFR = predict_y_RFR

  result_list = []

  print (restored_x[0])
  for i in range(len(restored_x)):
    idx = int(restored_x[i][0])
    data1 = restored_y_GBT[i]
    data2 = restored_y_RFR[i]


    print (idx, data1, data2)










if __name__ == "__main__":
  main()
