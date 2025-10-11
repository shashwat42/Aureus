from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from config import RANDOM_FOREST_PARAMS


def get_random_forest(params=None):
    if params is None: params = RANDOM_FOREST_PARAMS
    return RandomForestRegressor(**params)

def get_gradient_boosting(params=None):
    if params is None: params = {"n_estimators": 100, "max_depth": 3, "random_state": 42}
    return GradientBoostingRegressor(**params)

def get_linear_regression():
    return LinearRegression()

def get_svr(params=None):
    if params is None: params = {"kernel": "rbf", "C": 1.0, "epsilon": 0.2}
    return SVR(**params)

def get_keras_regressor(input_dim):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model
