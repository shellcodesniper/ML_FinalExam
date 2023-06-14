from io import StringIO
import os, glob, tqdm, pickle
import numpy as np, pandas as pd
import tensorflow as tf
BASE_PATH = 'datas/'

# TODO : Nan Value 채우고(해야함?), 학습 가능하도록 string -> numeric 변환.
def replace_string(datas):
  unique_dict = sorted(datas.dropna().unique())
  unique_dict = { unique_dict[i] : i for i in range(len(unique_dict))}
  # NOTE : Fill Nan with -1!
  unique_dict['Nan'] = -1
  datas = datas.fillna('Nan').apply(lambda x : unique_dict[x])
  return datas.copy()

class _Loaders:
  def __init__(self):
    self.test = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))

  def get_test(self):
    return self.test

  def get_merged(self):
    if os.path.exists('datas/merged.pkl'):
      return pickle.load(open('datas/merged.pkl', 'rb'))

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

    # TODO : String ( object ) -> Numeric 으로 변환
    for col in md.columns:
      if col in  ['id', 'date']:
        continue

      if md[col].dtype == 'object':
        md[col] = replace_string(md[col])

    print(f"NULL(B):\n{md.isnull().sum()}")

    # TODO : oil -> 전날 가격이 없는 경우, 전전날 가격으로 채우기
    md['dcoilwtico'] = md['dcoilwtico'].fillna(lambda x: x.rolling(3).mean())
    # md['dcoilwtico'] = md['dcoilwtico'].fillna(method='bfill') # NOTE : 그 외값은, 앞선 가격으로 채워버리기.


    # TODO : transactions -> 거래량 추론
    md['transactions'] = md.groupby(['store_nbr','holiday_type'])['transactions'].transform(lambda x: x.fillna(x.mean())) # NOTE : 같은 상점의 holiday_type 이 같은경우를 평균을 내서 채움.
    md['transactions'] = md.groupby(['store_nbr'])['transactions'].transform(lambda x: x.fillna(x.mean())) # NOTE : 그 외값은, 같은 상점의 평균으로 채움.

    print(f"NULL(A):\n{md.isnull().sum()}")

    # TODO : Nan Value 채우기

    csv_buf = StringIO()
    md.to_csv(csv_buf, sep=",", index=True, mode="wt", encoding="UTF-8")
    csv_buf.seek(0)
    md = pd.read_csv(csv_buf)
    pickle.dump(md, open('datas/merged.pkl', 'wb'))

    return md 

  def get_trainset(self):
    pre_processed = self.get_merged()

    pre_processed['date'] = pre_processed['date'].apply(lambda x : x.replace('-','')).astype(int) # NOTE : 날짜를 숫자로 변환

    x_train = pre_processed.drop(['sales', 'state', 'description', 'transferred'], axis=1)
    y_train = pre_processed['sales']

    dataset = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))

    return dataset






Loader = _Loaders()
