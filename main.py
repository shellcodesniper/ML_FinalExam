from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas, warnings, math, random
 
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
import numpy as np


# NOTE : 기존에 사용하던, Callbacks

IS_MAC_OS = platform.system() == 'Darwin'

def main():
  DataLoader = Loaders(True)

  # NOTE : DATA PREPARE
  x_scaler = DataLoader.x_scaler
  y_scaler = DataLoader.y_scaler



  # NOTE : Train Data Load
  (x_test, y_test) = DataLoader.get_validation_set() # TYPE : Train Dataset Generator
  (x_train, y_train) = DataLoader.as_raw_set()
  (predict_x, _)= Loaders(False).as_raw_set()



  # NOTE : Prepare Models
  tree_seed = random.randint(0,999999)  # NOTE : 같은 결과를 얻기 위해
  [model_GBT, model_RFR, model_RS, Model_Concated] = Model.concatModel(tree_seed, (16, ))






  # NOTE : Train All Load Weight
  if IS_MAC_OS:
    if os.path.exists('datas/model_GBT.h5') and os.path.exists('datas/model_RFR.h5'):
      x_train = x_train[:10]
      y_train = y_train[:10]
      model_GBT(x_train, y_train)
      model_RFR(x_train, y_train)
      model_GBT.load_weights('datas/model_GBT.h5')
      model_RFR.load_weights('datas/model_RFR.h5')
    else:
      model_GBT.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('gbt'))
      model_RFR.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('rfr'))

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


  model_RS.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])


  # TODO : Ocastration
  Model_Concated.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

  Model_Concated.summary()

  Model_Concated.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('concated'))




  # predict_y_GBT = model_GBT.predict(predict_x)
  # predict_y_RFR = model_RFR.predict(predict_x)
  #
  # restored_x = x_scaler.inverse_transform(predict_x)
  #
  # restored_y_GBT = y_scaler.inverse_transform(predict_y_GBT)
  # restored_y_RFR = y_scaler.inverse_transform(predict_y_RFR)

  # result_list = []
  #
  # print (restored_x[0])
  # for i in range(len(restored_x)):
  #   idx = int(restored_x[i][0])
  #   data1 = restored_y_GBT[i]
  #   data2 = restored_y_RFR[i]
  #
  #
  #   print (idx, data1, data2)
  #   # result_list.append(int((data1+data2) / 2))
  #   import math
  #   result_list.append(math.ceil(data2))
  # submission = pandas.read_csv('datas/sample_submission.csv')
  # submission['sales'] = result_list
  # submission.to_csv('datas/submission.csv', index=False)











if __name__ == "__main__":
  main()
