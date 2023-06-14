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


def main():

  train_dataset = Loader.get_trainset()

  # NOTE : Model
  model = Model.TSBaseModel()

  model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
  )

  # dataset_to_numpy = list(train_dataset.as_numpy_iterator())
  # shape = tf.shape(dataset_to_numpy)
  # print(shape)

  model.build(input_shape=(None, 13))

  model.summary()

  shuffled = train_dataset.shuffle(1000)
  for (batch, (x, y)) in enumerate(shuffled.take(10)):
    print (batch)
    print(x.shape, y.shape)
    model.fit(x, y, epochs=10, batch_size=10)







if __name__ == "__main__":
  main()
