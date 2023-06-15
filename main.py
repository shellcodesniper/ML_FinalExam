from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas, warnings, math, random
 
warnings.filterwarnings(action='ignore')
tf.config.set_soft_device_placement(True)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     tf.config.experimental.set_memory_growth(gpus[0], True)
#   except RuntimeError as e:
#     print(e)

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
      x = x_train
      y = y_train
      x_train = x_train[:10]
      y_train = y_train[:10]
      model_GBT(x_train, y_train)
      model_RFR(x_train, y_train)
      x_train = x
      y_train = y
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


  model_RS.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mae', 'mse'],
  )


  # TODO : Ocastration
  Model_Concated.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

  model_GBT.build((None, 16))
  model_GBT.summary()
  model_RFR.build((None, 16))
  model_RFR.summary()

  model_RS.build((None, 18))
  model_RS.summary()

  Model_Concated.build((None, 16))
  Model_Concated.summary()

  concat_history = Model_Concated.fit(
    x_train, y_train,
    verbose=0,
    epochs=100,
    batch_size=32,
    callbacks=Model.get_callback('concated'),
  )

  print (concat_history.history['loss'])

  collaborated_result = Model_Concated.evaluate(x_test, y_test, return_dict=True)
  print("=====================================")
  print(f"Result Concated : {collaborated_result}")

  # NOTE : Predict

  predict_y = Model_Concated.predict(predict_x)

  restored_x = x_scaler.inverse_transform(predict_x)
  restored_y = y_scaler.inverse_transform(predict_y)


  import math
  result_list = []
  # print (restored_x[0])
  for i in range(len(restored_x)):
    # idx = int(restored_x[i][0])
    res = restored_y[i]

    result_list.append(math.floor(res))

  submission = pandas.read_csv('datas/sample_submission.csv')
  submission['sales'] = result_list
  submission.to_csv('datas/submission.csv', index=False)
  















if __name__ == "__main__":
  main()
