import tensorflow as tf
from tensorflow import keras
import kuuwange as MY
from kuuwange.loaders import Loader

import kuuwange.models as Model


def main():

  merged_data = Loader.get_merged()

  shop_by_shop = merged_data.groupby('store_nbr')

  # date_by_date = merged_data.groupby('date')

  shop_names = shop_by_shop.groups.keys()

  # dates = date_by_date.groups.keys()

  # NOTE : Want to Predict 'sales' for each shop

  # print (shop_names)
  # print (dates)


  DATASET_X = []
  DATASET_Y = []

  # NOTE : Want to Predict 'sales' for each shop
  # print (shop_data['sales'].values)




if __name__ == "__main__":
  main()
