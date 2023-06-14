if __name__ != "__main__":
  raise Exception("This script is not meant to be imported.")

from kuuwange.loaders import Loader

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.subplots as ms
import plotly.express as px
import datetime, os, sys, time, glob
import dash, webbrowser
import dash_mantine_components as dmc


# plot total average sale vs lags for train data (total -> mean of all stores and families)

train_data = Loader.get_train()
