# src/models.py

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def get_random_forest(params=None):
    if params is None:
        params = {"n_estimators": 100, "max_depth": None, "random_state": 42}
    return RandomForestRegressor(**params)

def get_gradient_boosting(params=None):
    if params is None:
        params = {"n_estimators": 100, "max_depth": 3, "random_state": 42}
    return GradientBoostingRegressor(**params)

def get_linear_regression():
    return LinearRegression()

def get_svr(params=None):
    if params is None:
        params = {"kernel": "rbf", "C": 1.0, "epsilon": 0.2}
    return SVR(**params)
