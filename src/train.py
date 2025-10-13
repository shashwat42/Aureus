import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from config import TRAIN_FEATURES_FILE, TRAIN_CLEAN_FILE, ENSEMBLE_MODEL_FILE
from models import get_xgboost, get_lightgbm, get_random_forest
from ensemble import StackingEnsemble

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return 100 * np.mean(diff)

def main():
    print("="*60)
    print("ENSEMBLE STACKING TRAINING")
    print("="*60)
    
    X = np.load(TRAIN_FEATURES_FILE)
    train_df = pd.read_csv(TRAIN_CLEAN_FILE)
    y = np.log1p(train_df['price'].values)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    base_models = [
        ('XGBoost', get_xgboost()),
        ('LightGBM', get_lightgbm()),
        ('RandomForest', get_random_forest())
    ]
    
    meta_model = Ridge(alpha=10.0, random_state=42)
    ensemble = StackingEnsemble(base_models, meta_model)
    ensemble.fit(X_train, y_train)
    
    print("Evaluating on validation set...")
    y_pred_log = ensemble.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_val_original = np.expm1(y_val)
    y_pred = np.clip(y_pred, 0.01, None)
    
    mae = mean_absolute_error(y_val_original, y_pred)
    r2 = r2_score(y_val_original, y_pred)
    smape_score = smape(y_val_original, y_pred)
    
    print("="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"MAE:   ${mae:.2f}")
    print(f"R²:    {r2:.3f}")
    print(f"SMAPE: {smape_score:.2f}%")
    print("="*60)
    
    joblib.dump(ensemble, ENSEMBLE_MODEL_FILE)
    print(f"\nModel saved to {ENSEMBLE_MODEL_FILE}")

if __name__ == "__main__":
    main()
