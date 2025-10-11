# src/eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
DATASET_DIR = '../dataset/'
OUTPUTS_DIR = '../outputs/'
TRAIN_FILE = os.path.join(DATASET_DIR, 'train.csv')
TEST_FILE = os.path.join(DATASET_DIR, 'test.csv')

if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)

# Load Data
train = pd.read_csv(TRAIN_FILE)
print("Train shape:", train.shape)
test = pd.read_csv(TEST_FILE)
print("Test shape:", test.shape)

# Peek at the data
print("\nTRAIN HEAD:")
print(train.head())
print("\nTEST HEAD:")
print(test.head())

# Info on columns and data types
print("\nTRAIN INFO:")
print(train.info())
print("\nTEST INFO:")
print(test.info())

# Summary statistics (train set)
print("\nTRAIN DESCRIBE:")
print(train.describe(include='all'))

# Missing values report
print("\nTrain missing values:\n", train.isnull().sum())
print("\nTest missing values:\n", test.isnull().sum())

# Distribution of price (train set)
plt.figure(figsize=(8,5))
plt.hist(train['price'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Distribution (Training Set)')
plt.savefig(os.path.join(OUTPUTS_DIR, 'price_distribution.png'))
plt.close()

# Identify Outliers
Q1 = train['price'].quantile(0.25)
Q3 = train['price'].quantile(0.75)
IQR = Q3 - Q1
outliers = train[(train['price'] < Q1 - 1.5 * IQR) | (train['price'] > Q3 + 1.5 * IQR)]
print(f"\nNumber of outliers (price): {len(outliers)}")
outlier_ids = outliers['sample_id'].tolist()

# Save outlier sample_ids
with open(os.path.join(OUTPUTS_DIR, 'train_outlier_sample_ids.txt'), 'w') as f:
    for i in outlier_ids:
        f.write(f"{i}\n")

# Distribution of catalog_content length
train['content_len'] = train['catalog_content'].str.len()
plt.figure(figsize=(8,5))
plt.hist(train['content_len'], bins=50, color='orange', edgecolor='black')
plt.xlabel('Catalog Content Length')
plt.ylabel('Frequency')
plt.title('catalog_content Length Distribution')
plt.savefig(os.path.join(OUTPUTS_DIR, 'content_length_distribution.png'))
plt.close()

# Save simple EDA summary
summary = {
    'train_shape': train.shape,
    'test_shape': test.shape,
    'num_train_missing': train.isnull().sum().to_dict(),
    'num_test_missing': test.isnull().sum().to_dict(),
    'price_min': float(train['price'].min()),
    'price_max': float(train['price'].max()),
    'price_mean': float(train['price'].mean()),
    'num_price_outliers': int(len(outliers))
}
summary_file = os.path.join(OUTPUTS_DIR, 'eda_summary.txt')
with open(summary_file, 'w') as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

print("\nEDA complete. Summary, plots, and outlier IDs saved in ../outputs/")

