import tensorflow as tf
# tf.config.set_soft_device_placement(True)
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

es = tf.keras.callbacks.EarlyStopping(
  monitor='loss', verbose=1, mode='auto',
  baseline=None, restore_best_weights=True, patience=12
)
mc = tf.keras.callbacks.ModelCheckpoint(
  filepath='datas/predict.keras',
  save_best_only=True
)

rlr = tf.keras.callbacks.ReduceLROnPlateau(
  monitor='loss', factor=0.1, patience=10, verbose=2,
  mode='auto'
)
csv_logger = tf.keras.callbacks.CSVLogger('datas/training.log')

# NOTE : 기존에 사용하던, Callbacks


def main():

  train_dataset = Loader.get_trainset()

  # NOTE : Model
  model = Model.TSBaseModel()

  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss=tf.keras.losses.MeanSquaredLogarithmicError(),
    metrics=['mse', 'mae'],
  )

  model.build(input_shape=(None, 13))

  model.summary()


  for _i in range(int(3054348 / 1000)):
    shuffled = train_dataset.shuffle(1000)
    history = model.fit(shuffled, batch_size=10, epochs=50, verbose="auto", shuffle=True, callbacks=[es, mc, rlr, csv_logger])

    # show history
    # print (history.history.keys())
    # "Loss"
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])

    # TODO : Evaluation 
    result = model.evaluate(shuffled, batch_size=10, verbose='auto', )
    print (result)
    # TODO : Save Model
  # NOTE : Show History









if __name__ == "__main__":
  main()
