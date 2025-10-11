import os

DATASET_DIR = 'dataset'
OUTPUTS_DIR = 'outputs'

TRAIN_FILE = os.path.join(DATASET_DIR, 'train.csv')
TEST_FILE = os.path.join(DATASET_DIR, 'test.csv')
TRAIN_CLEAN_FILE = os.path.join(DATASET_DIR, 'train_cleaned.csv')
TEST_CLEAN_FILE = os.path.join(DATASET_DIR, 'test_cleaned.csv')
TRAIN_FEATURES_FILE = os.path.join(OUTPUTS_DIR, 'train_features.npy')
TEST_FEATURES_FILE = os.path.join(OUTPUTS_DIR, 'test_features.npy')

MODEL_FILE = os.path.join(OUTPUTS_DIR, 'rf_model.joblib')
PREDICTIONS_FILE = os.path.join(OUTPUTS_DIR, 'test_predictions.csv')

IMAGES_DIR = 'images'

TFIDF_MAX_FEATURES = 200

KEYWORDS = [
    'fresh', 'natural', 'organic', 'quality', 'light', 'premium', 'protein',
    'gluten_free', 'vegan', 'non_gmo', '100percent', 'pure', 'gourmet',
    'original', 'keto', 'authentic', 'all_natural', 'sugar_free', 'whole_grain'
]

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42
}

NUM_PARALLEL_DOWNLOADS = 8
