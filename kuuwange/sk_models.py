from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

COLLAPSED_MODEL_PIPELINE = []

_linearRegressor = LinearRegression()
_lasso = Lasso()
_elasticNet = ElasticNet()
_decisionTreeRegressor = DecisionTreeRegressor()
_kNeighborsRegressor = KNeighborsRegressor()
_gradientBoostingRegressor = GradientBoostingRegressor()
_randomForestRegressor = RandomForestRegressor()
_histGradientBoostingRegressor = HistGradientBoostingRegressor()


pipeline = [
  ('LinearRegressor', _linearRegressor),
  ('Lasso', _lasso),
  ('ElasticNet', _elasticNet),
  ('DecisionTreeRegressor', _decisionTreeRegressor),
  ('KNeighborsRegressor', _kNeighborsRegressor),
  ('GradientBoostingRegressor', _gradientBoostingRegressor),
  ('RandomForestRegressor', _randomForestRegressor),
  ('HistGradientBoostingRegressor', _histGradientBoostingRegressor),
]

for (name, model) in pipeline:
  COLLAPSED_MODEL_PIPELINE.append((name, model))
