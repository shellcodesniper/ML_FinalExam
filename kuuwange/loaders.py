import os, glob, tqdm, pickle
import numpy as np, pandas as pd
BASE_PATH = 'datas/'

def str_to_int(holiday):
  if holiday == 'None':
    return 0
  elif holiday == 'Holiday':
    return 1
  elif holiday == 'Transfer':
    return 2
  elif holiday == 'Additional':
    return 3
  elif holiday == 'Bridge':
    return 4
  else:
    return 5

class _Loaders:
  def __init__(self):
    self.test = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))

  def get_test(self):
    return self.test

  def get_merged(self):
    oil = pd.read_csv('datas/oil.csv')
    holidays = pd.read_csv('datas/holidays_events.csv')
    stores = pd.read_csv('datas/stores.csv')
    transactions = pd.read_csv('datas/transactions.csv')
    train = pd.read_csv('datas/train.csv')

    md = pd.merge(train, oil, how = 'left', on='date')
    md = pd.merge(md, holidays, how = 'left',on = 'date')
    md = pd.merge(md, transactions, how ='left', on =['date','store_nbr'])
    md = pd.merge(md, stores, how = 'left', on = 'store_nbr')
    md.rename(columns={'type_x':'holiday_type', 'type_y':'store_type'}, inplace = True)

    # TODO : string to enum or int
    md['holiday_type'] = md['holiday_type'].fillna('None')
    md['holiday_type'] = md['holiday_type'].apply(str_to_int)

    print(md.head())


    return md 






Loader = _Loaders()
