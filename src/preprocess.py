# src/preprocess.py

import pandas as pd
import os
import re

# Paths
DATASET_DIR = 'dataset'
TRAIN_FILE = os.path.join(DATASET_DIR, 'train.csv')
TEST_FILE = os.path.join(DATASET_DIR, 'test.csv')
TRAIN_CLEAN_FILE = os.path.join(DATASET_DIR, 'train_cleaned.csv')
TEST_CLEAN_FILE = os.path.join(DATASET_DIR, 'test_cleaned.csv')

# Load data
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

# --- Text Cleaning Function ---
def clean_text(text):
    if pd.isnull(text):
        return ''
    # Lowercase, remove non-alphanumerics except spaces
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean catalog_content text
train['catalog_content'] = train['catalog_content'].apply(clean_text)
test['catalog_content'] = test['catalog_content'].apply(clean_text)

# Handle missing image_link as empty string (none in your sample, but robust!)
train['image_link'] = train['image_link'].fillna('')
test['image_link'] = test['image_link'].fillna('')

# EXAMPLE: Extract basic features
# (Add more feature extraction here based on what you learn from EDA)
train['content_length'] = train['catalog_content'].str.len()
test['content_length'] = test['catalog_content'].str.len()

# Optional: Check for any remaining missing values and fill/drop accordingly
train = train.fillna('')   # You may tweak this as needed
test = test.fillna('')

# Save cleaned files
train.to_csv(TRAIN_CLEAN_FILE, index=False)
test.to_csv(TEST_CLEAN_FILE, index=False)

print("Preprocessing complete. Cleaned files saved as 'train_cleaned.csv' and 'test_cleaned.csv' in dataset/")
