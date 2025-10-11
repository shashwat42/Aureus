import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from config import TRAIN_FEATURES_FILE, TRAIN_CLEAN_FILE, MODEL_FILE
from models import get_random_forest

X = np.load(TRAIN_FEATURES_FILE)
train_df = pd.read_csv(TRAIN_CLEAN_FILE)
y = train_df['price'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = get_random_forest()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"Validation MAE: {mae:.2f}")
print(f"Validation R2: {r2:.2f}")

joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
