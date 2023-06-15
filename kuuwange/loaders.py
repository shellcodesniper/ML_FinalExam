from io import StringIO
import os, glob, tqdm, pickle
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf
BASE_PATH = 'datas/'

# TODO : Nan Value 채우고(해야함?), 학습 가능하도록 string -> numeric 변환.
def replace_string(datas):
  unique_dict = sorted(datas.dropna().unique())
  unique_dict = { unique_dict[i] : i for i in range(len(unique_dict))}
  # NOTE : Fill Nan with -1!
  unique_dict['Nan'] = -1
  datas = datas.fillna('Nan').apply(lambda x : unique_dict[x])
  return datas.copy()

class Loaders:
  def __init__(self, IS_TRAIN=True):
    self.base = pd.read_csv(os.path.join(BASE_PATH, 'test.csv')) if not IS_TRAIN else pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
    self.merged_path = os.path.join(BASE_PATH, 'merged.pkl') if not IS_TRAIN else os.path.join(BASE_PATH, 'merged_train.pkl')

    self.scaler = StandardScaler(
      with_std=True,
      with_mean=True,
      copy=True
    )

  def get_merged(self):
    if os.path.exists(self.merged_path):
      return pickle.load(open(self.merged_path, 'rb'))

    oil = pd.read_csv('datas/oil.csv')
    holidays = pd.read_csv('datas/holidays_events.csv')
    stores = pd.read_csv('datas/stores.csv')
    transactions = pd.read_csv('datas/transactions.csv')
    base_csv = self.base.copy()

    # TODO : oil -> 전날 가격이 없는 경우, 전전날 가격으로 채우기
    X = oil['dcoilwtico'].values.reshape((203,6))
    imputer = KNNImputer(n_neighbors=5, weights = 'distance')
    z = imputer.fit_transform(X)
    z = z.reshape((1218,1))
    oil['dcoilwtico'] = z

    md = pd.merge(base_csv, oil, how = 'left', on='date')
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

    md['dcoilwtico']= md['dcoilwtico'].fillna(method='bfill')

    # TODO : transactions -> 거래량 추론
    md['transactions'] = md.groupby(['store_nbr','holiday_type'])['transactions'].transform(lambda x: x.fillna(x.mean())) # NOTE : 같은 상점의 holiday_type 이 같은경우를 평균을 내서 채움.
    md['transactions'] = md.groupby(['store_nbr'])['transactions'].transform(lambda x: x.fillna(x.mean())) # NOTE : 그 외값은, 같은 상점의 평균으로 채움.

    print(f"NULL(A):\n{md.isnull().sum()}")

    # TODO : Nan Value 채우기

    csv_buf = StringIO()
    md.to_csv(csv_buf, sep=",", index=False, mode="wt", encoding="UTF-8")
    csv_buf.seek(0)
    md = pd.read_csv(csv_buf)
    pickle.dump(md, open('datas/merged.pkl', 'wb'))

    return md 

  def as_raw_set(self):
    pre_processed = self.get_merged()
    pre_processed = pre_processed.reset_index().set_index('id')


    pre_processed['date'] = pre_processed['date'].str.replace('-', '').astype(int)
    # pre_processed['dcoilwtico'] = pre_processed['dcoilwtico'].str.replace(',', '').astype(float)


    # x_train = pre_processed.drop(['sales', 'state', 'description', 'transferred'], axis=1)
    x_train = pre_processed.drop(['sales'], axis=1) # TYPE : Drop Just sales
    y_train = pre_processed[['sales']] # TYPE : 2-D Required.
    # print (x_train.head())

    # x_train = x_train.to_numpy()
    # y_train = y_train.to_numpy()


    # NOTE : Scaling

    x_train = self.scaler.fit_transform(x_train)
    y_train = self.scaler.fit_transform(y_train)
    print ("============ X, Y ===============")
    print ("X[0]:", x_train[0], x_train.shape,"\nY[0]:", y_train[0], y_train.shape)
    print ("===========================")

    # print (x_train.shape, y_train.shape)

    # y_train = y_train.reshape((-1,1))


    return (x_train, y_train)

  def as_generator(self, batch_size = 1, shuffle = True):
    (x_train, y_train) = self.as_raw_set()
    for i in range(0, len(x_train), batch_size):
      collection_x = []
      collection_y = []
      for j in range(batch_size):
        if shuffle:
          idx = np.random.randint(0, len(x_train))
          collection_x.append(x_train[idx])
          collection_y.append(y_train[idx])
        else:
          collection_x.append(x_train[i])
          collection_y.append(y_train[i])
      y_set = np.array(collection_y)
      y_set= np.ravel(y_set,  order = 'C')

      yield (np.array(collection_x), y_set)


  def as_dataset(self):
    (x_train, y_train) = self.as_raw_set()
    dataset = tf.data.Dataset.from_tensors((x_train, y_train))

    return dataset
  
  def get_sample(self):
    (x_train, y_train) = self.as_raw_set()
    return (x_train[0], y_train[0])

  def as_randomforest_dataset(self):
    pre_processed = self.get_merged()
    pre_processed['date'] = pre_processed['date'].str.replace('-', '').astype(int)

    predict_label = 'sales'

    train_set = pre_processed 

    pre_processed['sales'] = self.scaler.fit_transform(pre_processed['sales']) # NOTE: Scaling
    return tfdf.keras.pd_dataframe_to_tf_dataset(
      train_set,
      label=predict_label
    )
