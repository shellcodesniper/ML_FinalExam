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
      usecols=['id','store_nbr', 'family', 'date', 'sales', 'onpromotion'],
      dtype={
        'id': 'uint32',
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
      usecols=['id','store_nbr', 'family', 'date', 'onpromotion'],
      dtype={
          'id': 'uint32',
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
    # self.store_info['date'] = self.store_info.date.dt.to_period('D')

    self.oil_info = pd.read_csv(
      os.path.join(BASE_PATH, 'oil.csv'),
      dtype={
        'dcoilwtico': 'float32',
      },
      parse_dates=['date'],
    )

    self.oil_info['date'] = self.oil_info.date.dt.to_period('D')

    self.transact = pd.read_csv(
      os.path.join(BASE_PATH, 'transactions.csv'),
      dtype={
        'store_nbr': 'category',
      },
      parse_dates=['date'],
    )
    self.transact['date'] = self.transact.date.dt.to_period('D')

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
    self.holidays['date'] = self.holidays.date.dt.to_period('D')


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

  def get_merged(self, train = True):
    base = self.get_stores().reset_index().rename(columns={'type': 'store_type', 'cluster': 'store_cluster'})
    
    if train:
      data = self.get_train().rename(columns={'id': 'class_id'}).reset_index()
      print (data.dtypes)

      train_cols = ["class_id", "family", "sales", "onpromotion"]
      for c in train_cols:
        base[c] = [None for i in range(0, base.shape[0])]

      for i in range(0, data.shape[0]):
        target_date = data.iloc[i].date
        for c in train_cols:
          item = data.iloc[i][c]
          base.loc[(base.store_nbr == data.iloc[i].store_nbr) & (base.family == data.iloc[i].family), c] = item

          

    # merged = self.get_stores().reset_index().rename(columns={'type': 'store_type', 'cluster': 'store_cluster'})
    #
    #
    # if train:
    #   total_sales = self.get_train().reset_index().groupby(['store_nbr', 'date']).agg({'sales': 'sum', 'onpromotion': 'sum', 'family': 'count'}).reset_index()
    #   print (total_sales.shape[0])
    #   merged = merged.merge(total_sales, on='store_nbr', how='right', suffixes=('', '_ts'))
    # else:
    #   merged = merged.merge(self.get_test().reset_index().groupby(['store_nbr', 'date']).agg({'sales': 'sum', 'onpromotion': 'sum', 'family': 'count'}).reset_index(), on='store_nbr', how='left', suffixes=('', '_ts')).drop_columns('index_ts')
    #
    # holiday = self.get_holidays().reset_index().rename(columns={'type': 'holiday_type', 'locale': 'holiday_locale', 'locale_name': 'holiday_locale_name', 'description': 'holiday_description'})
    # merged = merged.merge(holiday, on='date', how='left', suffixes=('', '_holiday'))
    #
    # transact = self.get_transactions().reset_index()
    # merged = merged.merge(transact, on=('store_nbr', 'date'), how='left', suffixes=('', '_transact'))
    #
    #
    # # merged['date'] = pd.to_datetime(merged['date'].astype(str), format='%Y-%m-%d') # TYPE : Conversion
    #
    # oil_prices = self.get_oil().reset_index().rename(columns={'dcoilwtico': 'oil_price'})
    # merged = merged.merge(oil_prices, on='date', how='left', suffixes=('', '_oil'))
    # merged['oil_price'] = merged['oil_price'].fillna(method='bfill')
    #
    #
    # # RESULT : Reordering
    # merged_columns: list[str] = merged.columns.values.tolist()
    # for col in ['store_nbr', 'date', 'sales']:
    #   merged_columns.remove(col)
    # merged_columns = ['store_nbr', 'date'] + merged_columns + ['sales']
    # merged = merged.reindex(columns=merged_columns)

    return merged






Loader = _Loaders()
