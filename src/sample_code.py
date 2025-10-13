import os
import numpy as np
import pandas as pd
import joblib
from config import TEST_FEATURES_FILE, TEST_CLEAN_FILE, ENSEMBLE_MODEL_FILE, TRAIN_CLEAN_FILE
from ensemble import StackingEnsemble

def smape_optimized_correction(predictions, train_prices):
    train_q25 = np.percentile(train_prices, 25)
    train_q75 = np.percentile(train_prices, 75)
    
    corrected = predictions.copy()
    
    low_mask = predictions < train_q25
    corrected[low_mask] *= 1.03
    
    mid_mask = (predictions >= train_q25) & (predictions < train_q75)
    corrected[mid_mask] *= 0.99
    
    high_mask = predictions >= train_q75
    corrected[high_mask] *= 0.95
    
    return corrected

def predict_prices():
    X_test = np.load(TEST_FEATURES_FILE)
    test_df = pd.read_csv(TEST_CLEAN_FILE)
    train_df = pd.read_csv(TRAIN_CLEAN_FILE)
    
    ensemble = joblib.load(ENSEMBLE_MODEL_FILE)
    
    y_pred_log = ensemble.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred_corrected = smape_optimized_correction(y_pred, train_df['price'].values)
    y_pred_corrected = np.clip(y_pred_corrected, 0.01, None)
    
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': y_pred_corrected
    })
    
    output_filename = os.path.join('outputs', 'test_predictions.csv')
    os.makedirs('outputs', exist_ok=True)
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")

if __name__ == "__main__":
    predict_prices()
