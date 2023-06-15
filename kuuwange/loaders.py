from io import StringIO
import os, glob, tqdm, pickle, datetime
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf
BASE_PATH = 'datas/'

X_SCALER = StandardScaler(
    with_std=True,
    with_mean=True,
)

Y_SCALER = StandardScaler(
  with_std=True,
  with_mean=True,
)
SCALER_FITTED = False


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
    global X_SCALER, Y_SCALER
    print (f'[Loader] : {"TRAIN" if IS_TRAIN else "TEST"}')
    self.base = pd.read_csv(os.path.join(BASE_PATH, 'test.csv')) if not IS_TRAIN else pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
    self.merged_path = os.path.join(BASE_PATH, 'merged.pkl') if not IS_TRAIN else os.path.join(BASE_PATH, 'merged_train.pkl')
    self.is_train = IS_TRAIN
    self.x_scaler = X_SCALER
    self.y_scaler = Y_SCALER

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

    if (self.is_train):
      md['sales'] = md['sales'].astype('float32')

    

    # TODO : String ( object ) -> Numeric 으로 변환
    for col in md.columns:
      if col in  ['id', 'date']:
        continue

      if md[col].dtype == 'object':
        md[col] = replace_string(md[col])

    print(f"NULL(B):\n{md.isnull().sum()}")

    md['dcoilwtico']= md['dcoilwtico'].fillna(method='bfill')
    md['dcoilwtico'] = md['dcoilwtico'].astype('float32')

    # TODO : 1. 같은 날짜의 해당 상점 평균 거래량으로 채우기
    md['transactions'] = md['transactions'].fillna(md.groupby(['date', 'store_nbr'])['transactions'].transform('mean'))

    # TODO : 2. 같은 공휴일의 해당 상점 평균 거래량으로 채우기
    md['transactions'] = md['transactions'].fillna(md.groupby(['holiday_type', 'store_nbr'])['transactions'].transform('mean'))

    # TODO : 3. 같은 공휴일의 전체 상점 평균 거래량으로 채우기
    md['transactions'] = md['transactions'].fillna(md.groupby('holiday_type')['transactions'].transform('mean'))

    # TODO : 4. 같은 상점의 전체 평균 거래량으로 채우기
    md['transactions'] = md['transactions'].fillna(md.groupby('store_nbr')['transactions'].transform('mean'))

    # TODO : 5. 전체 평균 거래량으로 채우기
    md['transactions'] = md['transactions'].fillna(md['transactions'].mean())

    print(f"NULL(A):\n{md.isnull().sum()}")

    # TODO : Nan Value 채우기

    csv_buf = StringIO()
    md.to_csv(csv_buf, sep=",", index=False, mode="wt", encoding="UTF-8")
    csv_buf.seek(0)
    md = pd.read_csv(csv_buf)
    pickle.dump(md, open(self.merged_path, 'wb'))

    return md 

  def as_raw_set(self):
    global SCALER_FITTED
    pre_processed = self.get_merged()
    print(f"[결측치({'TRAIN' if self.is_train else 'TEST'})]")
    print(pre_processed.isnull().sum())
    print("---------------------------")
    pre_processed = pre_processed.reset_index().set_index('id')
    pre_processed['date'] = pre_processed['date'].str.replace('-', '').astype(int)

    x_train = pd.DataFrame()
    y_train = pd.DataFrame()

    if (self.is_train):
      x_train = pre_processed.drop(['sales'], axis=1) # TYPE : Drop Just sales
      y_train = pre_processed[['sales']] # TYPE : 2-D Required.
    else:
      x_train = pre_processed.copy()
      y_train = pd.DataFrame(np.zeros((len(x_train), 1)))


    # NOTE : Scaling
    # x_train = self.scaler.fit_transform(x_train)
    if (not SCALER_FITTED):
      self.x_scaler.fit(x_train)
      self.y_scaler.fit(y_train)
      SCALER_FITTED = True

    x_train = self.x_scaler.transform(x_train)
    y_train = self.y_scaler.transform(y_train)
    y_train = np.ravel(y_train.reshape(-1, 1),  order = 'C')

    print ("============ X, Y ===============")
    print ("X[0]:", x_train[0], x_train.shape)
    print ("Y[0]:", y_train[0], y_train.shape)
    print ("===========================")



    return (x_train, y_train)

  def get_validation_set(self):
    (x_train, y_train) = self.as_raw_set()

    TRAIN_SET_SIZE = int(len(x_train) * 0.8)
    x_set = x_train[TRAIN_SET_SIZE+1:]
    y_set = y_train[TRAIN_SET_SIZE+1:]

    y_set = np.ravel(y_set,  order = 'C')
    return (x_set, y_set)


  def as_generator(self, batch_size = 1, shuffle = True):
    (x_train, y_train) = self.as_raw_set()

    TRAIN_SET_SIZE = int(len(x_train) * 0.8)
    x_train = x_train[:TRAIN_SET_SIZE]
    y_train = y_train[:TRAIN_SET_SIZE]

    for i in range(0, len(x_train), batch_size):
      collection_x = []
      collection_y = []

      for j in range(batch_size):
        idx = np.random.randint(0, len(x_train)) if shuffle else i + j
        collection_x.append(x_train[idx])
        collection_y.append(y_train[idx])

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
