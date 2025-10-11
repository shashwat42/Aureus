# src/predict.py

import numpy as np
import pandas as pd
import os
import joblib

FEATURES_DIR = 'outputs'
DATASET_DIR = 'dataset'
TEST_FEATURES_FILE = os.path.join(FEATURES_DIR, 'test_features.npy')
TEST_FILE = os.path.join(DATASET_DIR, 'test_cleaned.csv')
MODEL_FILE = os.path.join(FEATURES_DIR, 'rf_model.joblib')
OUTPUT_FILE = os.path.join(FEATURES_DIR, 'test_predictions.csv')

X_test = np.load(TEST_FEATURES_FILE)
test_df = pd.read_csv(TEST_FILE)

model = joblib.load(MODEL_FILE)
y_pred = model.predict(X_test)

output_df = pd.DataFrame({
    'id': test_df['id'],
    'predicted_price': y_pred
})

output_df.to_csv(OUTPUT_FILE, index=False)
print(f"Predictions saved to {OUTPUT_FILE}")
