import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from config import TRAIN_CLEAN_FILE, TEST_CLEAN_FILE, TRAIN_FEATURES_FILE, TEST_FEATURES_FILE, KEYWORDS, TFIDF_MAX_FEATURES, OUTPUTS_DIR

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image

if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)

def extract_image_features(img_path, model, target_size=(224,224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        return features.flatten()
    except Exception:
        return np.zeros(model.output_shape[1])

train = pd.read_csv(TRAIN_CLEAN_FILE)
test = pd.read_csv(TEST_CLEAN_FILE)

tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
X_train_text = tfidf.fit_transform(train['catalog_content']).toarray()
X_test_text = tfidf.transform(test['catalog_content']).toarray()

img_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

train_img_paths = [os.path.join('images', os.path.basename(url)) for url in train['image_link']]
X_train_img = np.array([extract_image_features(p, img_model) for p in train_img_paths])

test_img_paths = [os.path.join('images', os.path.basename(url)) for url in test['image_link']]
X_test_img = np.array([extract_image_features(p, img_model) for p in test_img_paths])

feature_cols = ['weight', 'content_length'] + KEYWORDS
X_train_struct = train[feature_cols].astype(float).values
X_test_struct = test[feature_cols].astype(float).values

X_train = np.hstack([X_train_text, X_train_struct, X_train_img])
X_test = np.hstack([X_test_text, X_test_struct, X_test_img])

np.save(TRAIN_FEATURES_FILE, X_train)
np.save(TEST_FEATURES_FILE, X_test)

print("Feature engineering complete with image features. Features saved.")
