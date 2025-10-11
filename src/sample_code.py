import os
import numpy as np
import pandas as pd
import joblib
from config import TEST_FEATURES_FILE, TEST_CLEAN_FILE, MODEL_FILE

def predict_prices():
    X_test = np.load(TEST_FEATURES_FILE)
    test_df = pd.read_csv(TEST_CLEAN_FILE)

    model = joblib.load(MODEL_FILE)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': y_pred
    })

    output_filename = os.path.join('outputs', 'test_predictions.csv')
    output_df.to_csv(output_filename, index=False)

    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")

if __name__ == "__main__":
    predict_prices()
