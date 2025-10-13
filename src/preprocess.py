import pandas as pd
import re
import numpy as np
from config import (
    TRAIN_FILE, TEST_FILE, TRAIN_CLEAN_FILE, TEST_CLEAN_FILE, 
    KEYWORDS, REMOVE_OUTLIERS, OUTLIER_LOWER_PERCENTILE, OUTLIER_UPPER_PERCENTILE
)

def extract_weight(text):
    """Enhanced weight extraction with more patterns"""
    if pd.isnull(text):
        return 0.0
    
    text = text.lower()
    

    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:kg|kilogram)',
        r'(\d+(?:\.\d+)?)\s*(?:g|gram|gm)(?:\s|$|,)',
        r'(\d+(?:\.\d+)?)\s*(?:oz|ounce)',
        r'(\d+(?:\.\d+)?)\s*(?:lb|pound)',
        r'(\d+(?:\.\d+)?)\s*(?:ml|milliliter)',
        r'(\d+(?:\.\d+)?)\s*(?:l|liter)(?:\s|$|,)',
    ]
    
    weights = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            weights.extend([float(m) for m in matches])
    

    pack_match = re.search(r'pack\s+of\s+(\d+)', text)
    pack_quantity = int(pack_match.group(1)) if pack_match else 1
    
    if weights:
        return max(weights) * pack_quantity
    
    return 0.0

def extract_quantity(text):
    """Extract item pack quantity"""
    if pd.isnull(text):
        return 1
    
    text = text.lower()
    patterns = [
        r'pack\s+of\s+(\d+)',
        r'(\d+)\s+pack',
        r'(\d+)\s+count',
        r'(\d+)\s+pieces',
        r'(\d+)\s+units'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    
    return 1

def extract_numeric_features(text):
    """Extract various numeric features"""
    if pd.isnull(text):
        return {'num_count': 0, 'avg_number': 0, 'max_number': 0}
    
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    numbers = [float(n) for n in numbers]
    
    return {
        'num_count': len(numbers),
        'avg_number': np.mean(numbers) if numbers else 0,
        'max_number': max(numbers) if numbers else 0
    }

def extract_flags(text, keywords):
    """Extract keyword flags"""
    if pd.isnull(text):
        return {key: 0 for key in keywords}
    
    text = text.lower()
    return {key: int(key.replace('_', ' ') in text) for key in keywords}

def clean_text(text):
    """Clean and normalize text"""
    if pd.isnull(text):
        return ''
    
    text = str(text).lower()

    text = re.sub(r'[^a-z0-9 ]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_row(row, keywords):
    """Process each row with enhanced features"""
    text = row['catalog_content'] if pd.notnull(row['catalog_content']) else ''
    cleaned_text = clean_text(text)
    

    row['catalog_content_clean'] = cleaned_text
    row['content_length'] = len(cleaned_text)
    row['word_count'] = len(cleaned_text.split())
    row['avg_word_length'] = np.mean([len(w) for w in cleaned_text.split()]) if cleaned_text else 0

    numeric_features = extract_numeric_features(text)
    row['num_count'] = numeric_features['num_count']
    row['avg_number'] = numeric_features['avg_number']
    row['max_number'] = numeric_features['max_number']

    row['weight'] = extract_weight(text)
    row['quantity'] = extract_quantity(text)
    

    flags = extract_flags(text, keywords)
    for k, v in flags.items():
        row[k] = v
    

    row['has_image'] = 1 if pd.notnull(row['image_link']) and row['image_link'] != '' else 0
    row['image_link'] = row['image_link'] if pd.notnull(row['image_link']) else ''
    
    return row

def remove_outliers(df, target_col='price'):
    """Remove price outliers using percentile method"""
    if target_col not in df.columns:
        return df
    
    lower_bound = df[target_col].quantile(OUTLIER_LOWER_PERCENTILE / 100)
    upper_bound = df[target_col].quantile(OUTLIER_UPPER_PERCENTILE / 100)
    
    print(f"Removing outliers: {lower_bound:.2f} < price < {upper_bound:.2f}")
    original_len = len(df)
    df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
    print(f"Removed {original_len - len(df)} outlier samples")
    
    return df

def preprocess_data():
    """Main preprocessing function"""
    print("Loading datasets...")
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    

    print("Cleaning train data...")
    train = train.apply(lambda row: clean_row(row, KEYWORDS), axis=1)
    
    print("Cleaning test data...")
    test = test.apply(lambda row: clean_row(row, KEYWORDS), axis=1)
    

    if REMOVE_OUTLIERS and 'price' in train.columns:
        print("Removing outliers from training data...")
        train = remove_outliers(train, 'price')

    train.to_csv(TRAIN_CLEAN_FILE, index=False)
    test.to_csv(TEST_CLEAN_FILE, index=False)
    
    print(f"✅ Cleaned data saved:")
    print(f"   Train: {TRAIN_CLEAN_FILE} (shape: {train.shape})")
    print(f"   Test: {TEST_CLEAN_FILE} (shape: {test.shape})")

if __name__ == "__main__":
    preprocess_data()
