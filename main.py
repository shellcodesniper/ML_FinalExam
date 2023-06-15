import tensorflow as tf
tf.config.set_soft_device_placement(True)
# tf.debugging.set_log_device_placement(True)
# SET GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

import kuuwange.models as Model
import kuuwange as MY
from kuuwange.loaders import Loader
import math, time
import numpy as np

es = tf.keras.callbacks.EarlyStopping(
  monitor='loss', verbose=1, mode='auto',
  baseline=None, restore_best_weights=True, patience=12
)
mc = tf.keras.callbacks.ModelCheckpoint(
  filepath='datas/ckp.keras',  save_best_only=True
)

rlr = tf.keras.callbacks.ReduceLROnPlateau(
  monitor='loss', factor=0.1, patience=10, verbose=2,
  mode='auto'
)
csv_logger = tf.keras.callbacks.CSVLogger('datas/training.log')

# NOTE : 기존에 사용하던, Callbacks


def main():

  # train_dataset = Loader.as_randomforest_dataset()

  # NOTE : Model
  # model = Model.RandomForestModel()
  

  tree_seed = 9932 # NOTE : 같은 결과를 얻기 위해
  model_GBT = Model.gradientBoostingModel(tree_seed)

  model_RFR = Model.randomForstRegressionModel(tree_seed)

  generator = Loader.as_generator(batch_size=10000, shuffle=True)

  for data in generator:
    [x_train, y_train] = data
    result_GBT = model_GBT.fit(x_train, y_train, verbose=1, callbacks=[es, mc, rlr, csv_logger])
    result_RFR = model_RFR.fit(x_train, y_train, verbose=1, callbacks=[es, mc, rlr, csv_logger])

    print (result_GBT.print(), result_RFR.print())
    time.sleep(2)

  # kfold = KFold(n_splits=10)


  # cv_results = cross_val_score(model, x_pred, y_pred, scoring="neg_mean_squared_error", cv=10)


  # model.compile(
  #   # optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
  #   # loss=tf.keras.losses.MeanSquaredError(),
  #   metrics=['mse', 'mae'],
  # )

  # model.build(input_shape=(None, 13))
  #
  # model.summary()


  for _i in range(int(3054348 / 250)):
    shuffled = train_dataset.shuffle(250)
    history = model.fit(shuffled, batch_size=5, epochs=50, verbose="auto", shuffle=False, callbacks=[es, mc, rlr, csv_logger])

    # show history
    # print (history.history.keys())
    # "Loss"
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])

    # TODO : Evaluation 
    result = model.evaluate(shuffled, batch_size=5, verbose='auto', )
    print (result)

    predict = model.predict(shuffled, batch_size=5, verbose='auto', )
    # TODO : Save Model
  # NOTE : Show History









if __name__ == "__main__":
  main()
