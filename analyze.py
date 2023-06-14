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

train_data: pd.DataFrame = Loader.get_train()
store_info: pd.DataFrame = Loader.get_stores()
oil_info: pd.DataFrame = Loader.get_oil()
transact: pd.DataFrame = Loader.get_transactions()
holidays: pd.DataFrame = Loader.get_holidays()

plot_data = train_data.groupby(['date']).agg({'sales': 'sum'}).reset_index()
plot_data['date'] = pd.to_datetime(plot_data['date'].astype(str), format='%Y-%m-%d')

plot_data = plot_data.merge(oil_info, on='date', how='left')
plot_data = plot_data.merge(transact, on='date', how='left')
plot_data = plot_data.merge(store_info, on='store_nbr', how='left')

fig = ms.make_subplots(rows=2, cols=2, subplot_titles=("Total Sales", "Oil Price", "Transactions", "Holidays"))

fig.add_trace(
  go.Scatter(x=plot_data['date'], y=plot_data['sales'], name='Total Sales'),
  row=1, col=1
)
fig.add_trace(
  go.Scatter(x=plot_data['date'], y=plot_data['dcoilwtico'], name='Oil Price'),
  row=1, col=2
)

fig.add_trace(
  go.Scatter(x=plot_data['date'], y=plot_data['transactions'], name='Transactions'),
  row=2, col=1
)

fig.update_layout(height=600, width=800, title_text="Total Sales vs Oil Price vs Transactions vs Holidays")

fig.show()

mfig = ms.make_subplots(rows=4, cols=1, specs=[[{'rowspan':3}],[None],[None], [{'rowspan': 1}]], shared_xaxes=True, horizontal_spacing=0.03, vertical_spacing=0.01)

# NOTE : Base Data
mfig.add_trace(
  go.Scatter(x=plot_data['date'], y=plot_data['sales'], name='Total Sales'),
  row=1, col=1
)

# NOTE : Transaction Data
mfig.add_trace(
  go.Scatter(x=plot_data['date'], y=plot_data['transactions'], name='Transactions'),
  row=4, col=1
)

mfig.update_layout(autosize=True, xaxis1_rangeslider_visible=False, xaxis2_rangeslider_visible=False, margin=dict(l=50, r=50, t=50, b=50), template='seaborn')
  # dfig.update_layout(xaxis=dict(type="category", tickformat="%H%M"), xaxis2=dict(type="category", tickformat="%H%M"))
mfig.update_xaxes(tickformat='%Y.%m.%d', zeroline=True, zerolinewidth=1, zerolinecolor='black', showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black', mirror=True)
mfig.update_yaxes(tickformat=',d', zeroline=True, zerolinewidth=1, zerolinecolor='black', showgrid=True, gridwidth=2, gridcolor='lightgray',showline=True,linewidth=2, linecolor='black', mirror=True)
mfig.update_traces(xhoverformat='%Y.%m.%d') 
mfig.update_layout(width=1920, height=800)
mfig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
mfig.update_layout(hovermode="x unified")

mfig.show()

