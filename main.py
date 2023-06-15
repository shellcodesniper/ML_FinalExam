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
import time


# NOTE : 기존에 사용하던, Callbacks


def main():

  tree_seed = 9932 # NOTE : 같은 결과를 얻기 위해
  model_GBT = Model.gradientBoostingModel(tree_seed)

  model_RFR = Model.randomForstRegressionModel(tree_seed)

  predict_loader = Loaders(False)
  predict_scaler = predict_loader.get_scaler()

  (predict_x, _)= predict_loader.as_raw_set()
  (x_test, y_test) = Loaders(True).get_validation_set() # TYPE : Train Dataset Generator

  # train_generator = Loaders(True).as_generator(batch_size=100000, shuffle=True) # TYPE : Train Dataset Generator
  # for data in train_generator:
  #   [x_train, y_train] = data
  #   # TODO : Train
  #   model_GBT.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('gbt'))
  #   model_RFR.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('rfr'))
  #
  #   # INFO : Summary
  #   # model_GBT.summary()
  #   # model_RFR.summary()
  #
  #   # TODO : Evalulate
  #   result_GBT = model_GBT.evaluate(x_test, y_test, return_dict=True)
  #   result_RFR = model_RFR.evaluate(x_test, y_test, return_dict=True)
  #
  #   print("=====================================")
  #   print(f"Result GBT : {result_GBT}")
  #   print(f"Result RFR : {result_RFR}")
  #   print("=====================================")

  # NOTE : Train All Data (Epoch 1)
  (x_train, y_train) = Loaders(True).as_raw_set()
  model_GBT.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('gbt'))
  model_RFR.fit(x_train, y_train, verbose=0, callbacks=Model.get_callback('rfr'))

  model_GBT.save('datas/model_GBT.h5')
  model_RFR.save('datas/model_RFR.h5')


  predict_y_GBT = model_GBT.predict(predict_x)
  predict_y_RFR = model_RFR.predict(predict_x)
  restored_y_GBT = predict_scaler.inverse_transform(predict_y_GBT)
  restored_y_RFR = predict_scaler.inverse_transform(predict_y_RFR)

  result_list = []
  for i in range(len(predict_y_GBT)):
    idx = predict_x[i][0][0]
    data = restored_y_GBT[i][0]

    print (idx, data)










if __name__ == "__main__":
  main()
