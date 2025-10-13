import os

DATASET_DIR = 'dataset'
OUTPUTS_DIR = 'outputs'
IMAGES_DIR = 'images'

TRAIN_FILE = os.path.join(DATASET_DIR, 'train.csv')
TEST_FILE = os.path.join(DATASET_DIR, 'test.csv')
TRAIN_CLEAN_FILE = os.path.join(DATASET_DIR, 'train_cleaned.csv')
TEST_CLEAN_FILE = os.path.join(DATASET_DIR, 'test_cleaned.csv')

TRAIN_FEATURES_FILE = os.path.join(OUTPUTS_DIR, 'train_features.npy')
TEST_FEATURES_FILE = os.path.join(OUTPUTS_DIR, 'test_features.npy')

MODEL_FILE = os.path.join(OUTPUTS_DIR, 'rf_model.joblib')
XGBOOST_MODEL_FILE = os.path.join(OUTPUTS_DIR, 'xgb_model.joblib')
ENSEMBLE_MODEL_FILE = os.path.join(OUTPUTS_DIR, 'ensemble_model.joblib')
PREDICTIONS_FILE = os.path.join(OUTPUTS_DIR, 'test_predictions.csv')

TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 2)

KEYWORDS = [
    'fresh', 'natural', 'organic', 'quality', 'light', 'premium', 'protein',
    'gluten_free', 'vegan', 'non_gmo', '100percent', 'pure', 'gourmet',
    'original', 'keto', 'authentic', 'all_natural', 'sugar_free', 'whole_grain',
    'fat_free', 'low_fat', 'low_sodium', 'dairy_free', 'kosher', 'halal',
    'paleo', 'grain_free', 'soy_free', 'nut_free', 'lactose_free'
]

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 25,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": 'sqrt',
    "random_state": 42,
    "n_jobs": -1
}

XGBOOST_PARAMS = {
    'n_estimators': 450,
    'max_depth': 6,
    'learning_rate': 0.025,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'reg_alpha': 0.4,
    'reg_lambda': 2.5,
    'gamma': 0.15,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'n_estimators': 500,
    'max_depth': 7,
    'learning_rate': 0.02,
    'num_leaves': 40,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 3.0,
    'min_child_samples': 30,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42
}

TEST_SIZE = 0.15
RANDOM_STATE = 42
N_FOLDS = 5

REMOVE_OUTLIERS = True
OUTLIER_LOWER_PERCENTILE = 1.0
OUTLIER_UPPER_PERCENTILE = 99.0

NUM_PARALLEL_DOWNLOADS = 8
