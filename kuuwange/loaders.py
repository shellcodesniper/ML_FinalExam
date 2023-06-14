import os, glob, tqdm, pickle
import numpy as np, pandas as pd
BASE_PATH = 'datas/'

class _Loaders:
  def __init__(self):
    if (
      os.path.exists(os.path.join(BASE_PATH, 'train.pkl'))
      and os.path.exists(os.path.join(BASE_PATH, 'test.pkl'))
      and os.path.exists(os.path.join(BASE_PATH, 'stores.pkl'))
      and os.path.exists(os.path.join(BASE_PATH, 'oil.pkl'))
      and os.path.exists(os.path.join(BASE_PATH, 'transactions.pkl'))
      and os.path.exists(os.path.join(BASE_PATH, 'holidays.pkl'))
    ):
      self.train_raw = pickle.load(open(os.path.join(BASE_PATH, 'train.pkl'), 'rb'))
      self.test_raw = pickle.load(open(os.path.join(BASE_PATH, 'test.pkl'), 'rb'))
      self.store_info = pickle.load(open(os.path.join(BASE_PATH, 'stores.pkl'), 'rb'))
      self.oil_info = pickle.load(open(os.path.join(BASE_PATH, 'oil.pkl'), 'rb'))
      self.transact = pickle.load(open(os.path.join(BASE_PATH, 'transactions.pkl'), 'rb'))
      self.holidays = pickle.load(open(os.path.join(BASE_PATH, 'holidays.pkl'), 'rb'))
      return
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
    )
    self.test_raw['date'] = self.test_raw.date.dt.to_period('D')
    self.test_raw = self.test_raw.set_index(['store_nbr', 'family', 'date']).sort_index()

    self.store_info = pd.read_csv(
      os.path.join(BASE_PATH, 'stores.csv'),
      usecols=['store_nbr', 'city', 'state', 'type', 'cluster'],
      dtype={
        'store_nbr': 'category',
        'city': 'category',
        'state': 'category',
        'type': 'category',
        'cluster': 'category',
      },
    )

    self.oil_info = pd.read_csv(
      os.path.join(BASE_PATH, 'oil.csv'),
      dtype={
        'dcoilwtico': 'float32',
      },
      parse_dates=['date'],
    )

    self.transact = pd.read_csv(
      os.path.join(BASE_PATH, 'transactions.csv'),
      dtype={
        'store_nbr': 'category',
      },
      parse_dates=['date'],
    )

    self.holidays = pd.read_csv(
      os.path.join(BASE_PATH, 'holidays_events.csv'),
      dtype={
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
      },
      parse_dates=['date'],
    )


    pickle.dump(self.train_raw, open(os.path.join(BASE_PATH, 'train.pkl'), 'wb'))
    pickle.dump(self.test_raw, open(os.path.join(BASE_PATH, 'test.pkl'), 'wb'))
    pickle.dump(self.store_info, open(os.path.join(BASE_PATH, 'stores.pkl'), 'wb'))
    pickle.dump(self.oil_info, open(os.path.join(BASE_PATH, 'oil.pkl'), 'wb'))
    pickle.dump(self.transact, open(os.path.join(BASE_PATH, 'transactions.pkl'), 'wb'))
    pickle.dump(self.holidays, open(os.path.join(BASE_PATH, 'holidays.pkl'), 'wb'))

  def get_train(self):
    return self.train_raw
  
  def get_test(self):
    return self.test_raw

  def get_stores(self):
    return self.store_info

  def get_oil(self):
    return self.oil_info

  def get_transactions(self):
    return self.transact

  def get_holidays(self):
    return self.holidays





Loader = _Loaders()
