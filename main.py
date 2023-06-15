import tensorflow as tf
from tensorflow_decision_forests.keras import Task
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
import math
import numpy as np
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

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
  

  
  model = tfdf.keras.GradientBoostedTreesModel(
    task=Task.REGRESSION,
    num_trees=10,
    num_threads=4,
    max_depth=10,
  )

  (x_data, y_data) = Loader.as_raw_set()
  x_train = x_data[:10000]
  y_train = np.ravel(y_data[:10000],  order = 'C')

  model.fit(x_train, y_train, verbose=2)

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
