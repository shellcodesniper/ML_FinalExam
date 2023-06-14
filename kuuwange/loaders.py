import os, glob, tqdm, numpy as np, pandas as pd
BASE_PATH = 'data/'

class _Loaders:
  def __init__(self):
    self.train_raw = pd.read_csv(
    os.path.join(BASE_PATH, 'train.csv'),
      usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
      dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
        'onpromotion': 'uint32',
      },
      parse_dates=['date'],
      infer_datetime_format=True,
    )
    self.train_raw['date'] = self.train_raw.date.dt.to_period('D')
    self.train_raw = self.train_raw.set_index(['store_nbr', 'family', 'date']).sort_index()

    self.test_raw = pd.read_csv(
      os.path.join(BASE_PATH, 'test.csv'),
      dtype={
          'store_nbr': 'category',
          'family': 'category',
          'onpromotion': 'uint32',
      },
      parse_dates=['date'],
      infer_datetime_format=True,
    )
    self.test_raw['date'] = self.test_raw.date.dt.to_period('D')
    self.test_raw = self.test_raw.set_index(['store_nbr', 'family', 'date']).sort_index()

  def get_train(self):
    return self.train_raw
  
  def get_test(self):
    return self.test_raw





Loader = _Loaders()
