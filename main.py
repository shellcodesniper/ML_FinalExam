import tensorflow as tf
from tensorflow import keras
import kuuwange as MY
from kuuwange.loaders import Loader

from kuwange.models import ShopModel


def main():

  merged_data = Loader.get_merged()

  shop_by_shop = merged_data.groupby('store_nbr')

  date_by_date = merged_data.groupby('date')

  shop_names = shop_by_shop.groups.keys()

  dates = date_by_date.groups.keys()

  print (shop_names)
  print (dates)

  shopModel = MY.ShopModel(shop_names)


if __name__ == "__main__":
  main()
