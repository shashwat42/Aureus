import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from config import RANDOM_FOREST_PARAMS, XGBOOST_PARAMS, GRADIENT_BOOSTING_PARAMS, LIGHTGBM_PARAMS

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def get_random_forest(params=None):
    if params is None:
        params = RANDOM_FOREST_PARAMS
    return RandomForestRegressor(**params)

def get_gradient_boosting(params=None):
    if params is None:
        params = GRADIENT_BOOSTING_PARAMS
    return GradientBoostingRegressor(**params)

def get_xgboost(params=None):
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")
    if params is None:
        params = XGBOOST_PARAMS
    return xgb.XGBRegressor(**params)

def get_lightgbm(params=None):
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")
    if params is None:
        params = LIGHTGBM_PARAMS
    return lgb.LGBMRegressor(**params)

def get_ridge(alpha=1.0):
    return Ridge(alpha=alpha, random_state=42)
