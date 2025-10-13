import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
from config import (
    TRAIN_CLEAN_FILE, TEST_CLEAN_FILE,
    TRAIN_FEATURES_FILE, TEST_FEATURES_FILE,
    KEYWORDS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, OUTPUTS_DIR
)

try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Image features will be skipped.")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

def extract_image_features(img_path, model, target_size=(224, 224)):
    """Extracts EfficientNet features from an image."""
    try:
        if not os.path.exists(img_path):
            return None
        
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        return None

def get_image_features(df, images_dir='images'):
    """Extract image features for all samples"""
    if not TENSORFLOW_AVAILABLE:
        print("Skipping image features (TensorFlow not available)")
        return np.zeros((len(df), 1))
    
    print("Loading EfficientNetB0 model...")
    base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    feature_dim = 1280
    
    image_features = []
    has_valid_image = []
    
    print(f"Extracting image features for {len(df)} samples...")
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing image {idx}/{len(df)}...")
        
        if pd.notnull(row['image_link']) and row['image_link'] != '':
            img_filename = os.path.basename(row['image_link'])
            img_path = os.path.join(images_dir, img_filename)
            features = extract_image_features(img_path, base_model)
            
            if features is not None:
                image_features.append(features)
                has_valid_image.append(1)
            else:
                image_features.append(np.zeros(feature_dim))
                has_valid_image.append(0)
        else:
            image_features.append(np.zeros(feature_dim))
            has_valid_image.append(0)
    
    image_features = np.array(image_features)
    has_valid_image = np.array(has_valid_image).reshape(-1, 1)
    
    print(f"✅ Image features shape: {image_features.shape}")
    print(f"   Valid images: {has_valid_image.sum()}/{len(df)}")
    
    return np.hstack([image_features, has_valid_image])

def get_text_features(train_df, test_df):
    """Extract TF-IDF features from catalog content"""
    print("Extracting TF-IDF features...")
    
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )
    
    train_text = train_df['catalog_content_clean'].fillna('')
    test_text = test_df['catalog_content_clean'].fillna('')
    
    train_tfidf = vectorizer.fit_transform(train_text).toarray()
    test_tfidf = vectorizer.transform(test_text).toarray()
    
    # Save vectorizer
    joblib.dump(vectorizer, os.path.join(OUTPUTS_DIR, 'tfidf_vectorizer.joblib'))
    
    print(f"✅ TF-IDF features shape: {train_tfidf.shape}")
    print(f"   (Reduced to {TFIDF_MAX_FEATURES} features for better generalization)")
    return train_tfidf, test_tfidf

def get_structured_features(df):
    """Extract structured numeric and categorical features"""
    feature_cols = [
        'content_length', 'word_count', 'avg_word_length',
        'num_count', 'avg_number', 'max_number',
        'weight', 'quantity', 'has_image'
    ] + KEYWORDS
    
    # Create derived features
    df['weight_per_quantity'] = df['weight'] / (df['quantity'] + 1)
    df['price_indicator'] = df['num_count'] * df['avg_number']
    df['keyword_count'] = df[KEYWORDS].sum(axis=1)
    df['content_density'] = df['word_count'] / (df['content_length'] + 1)
    
    feature_cols.extend(['weight_per_quantity', 'price_indicator', 'keyword_count', 'content_density'])
    
    features = df[feature_cols].fillna(0).values
    return features

def build_features():
    """Main feature engineering pipeline"""
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE (SMAPE-OPTIMIZED)")
    print("=" * 60)
    
    # Load cleaned data
    print("\n1. Loading cleaned datasets...")
    train_df = pd.read_csv(TRAIN_CLEAN_FILE)
    test_df = pd.read_csv(TEST_CLEAN_FILE)
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Extract text features
    print("\n2. Extracting text features (reduced for generalization)...")
    train_text_features, test_text_features = get_text_features(train_df, test_df)
    
    # Extract structured features
    print("\n3. Extracting structured features...")
    train_structured = get_structured_features(train_df)
    test_structured = get_structured_features(test_df)
    print(f"   Structured features shape: {train_structured.shape}")
    
    # Extract image features
    print("\n4. Extracting image features...")
    train_image_features = get_image_features(train_df, 'images')
    test_image_features = get_image_features(test_df, 'images')
    
    # Combine all features
    print("\n5. Combining all features...")
    X_train = np.hstack([
        train_text_features,
        train_structured,
        train_image_features
    ])
    
    X_test = np.hstack([
        test_text_features,
        test_structured,
        test_image_features
    ])
    
    # Scale features
    print("\n6. Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(OUTPUTS_DIR, 'feature_scaler.joblib'))
    
    # Save features
    print("\n7. Saving features...")
    np.save(TRAIN_FEATURES_FILE, X_train)
    np.save(TEST_FEATURES_FILE, X_test)
    
    print("\n" + "=" * 60)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Final feature dimensions:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")


if __name__ == "__main__":
    build_features()
