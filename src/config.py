# src/config.py

# Data directories and file paths
DATASET_DIR = 'dataset'
OUTPUTS_DIR = 'outputs'

TRAIN_FILE = f"{DATASET_DIR}/train.csv"
TEST_FILE = f"{DATASET_DIR}/test.csv"
TRAIN_CLEAN_FILE = f"{DATASET_DIR}/train_cleaned.csv"
TEST_CLEAN_FILE = f"{DATASET_DIR}/test_cleaned.csv"
TRAIN_FEATURES_FILE = f"{OUTPUTS_DIR}/train_features.npy"
TEST_FEATURES_FILE = f"{OUTPUTS_DIR}/test_features.npy"

MODEL_FILE = f"{OUTPUTS_DIR}/rf_model.joblib"
PREDICTIONS_FILE = f"{OUTPUTS_DIR}/test_predictions.csv"

# Feature engineering config
TFIDF_MAX_FEATURES = 200

# Model parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42
}

# Parallel processes for image download
NUM_PARALLEL_DOWNLOADS = 100
