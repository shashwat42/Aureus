import pandas as pd
import re
from config import TRAIN_FILE, TEST_FILE, TRAIN_CLEAN_FILE, TEST_CLEAN_FILE, KEYWORDS

def extract_weight(text):
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|kg|oz|ounce|lb|pound)', text.lower())
    return float(match.group(1)) if match else 0.0

def extract_flags(text, keywords):
    text = text.lower()
    return {key: int(key.replace('_',' ') in text) for key in keywords}

def clean_row(row):
    text = row['catalog_content'].lower() if pd.notnull(row['catalog_content']) else ''
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    row['catalog_content'] = text
    row['weight'] = extract_weight(text)
    flags = extract_flags(text, KEYWORDS)
    for k, v in flags.items():
        row[k] = v
    row['content_length'] = len(text)
    return row

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
train = train.apply(clean_row, axis=1)
test = test.apply(clean_row, axis=1)
train['image_link'] = train['image_link'].fillna('')
test['image_link'] = test['image_link'].fillna('')
train = train.fillna(0)
test = test.fillna(0)
train.to_csv(TRAIN_CLEAN_FILE, index=False)
test.to_csv(TEST_CLEAN_FILE, index=False)
print("Preprocessing complete. Engineered features added. Cleaned files saved.")
