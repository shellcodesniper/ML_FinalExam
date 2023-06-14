import kuuwange as MY
from kuuwange.loaders import Loader
import tensorflow as tf

import kuuwange.models as Model


def main():

  train_dataset = Loader.get_trainset()

  # NOTE : Model
  model = Model.TSBaseModel()

  model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
  )

  dataset_to_numpy = list(train_dataset.as_numpy_iterator())
  shape = tf.shape(dataset_to_numpy)
  print(shape)

  model.build(input_shape=(None, 14))

  model.summary()

  model.fit(
    train_dataset,
    shuffle=True,
    batch_size=2,
    epochs=10,
  )







if __name__ == "__main__":
  main()
