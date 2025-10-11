import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from config import TRAIN_FILE, TEST_FILE, OUTPUTS_DIR, KEYWORDS

if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTRAIN HEAD:\n", train.head())
print("\nTEST HEAD:\n", test.head())
print("\nTRAIN INFO:\n"); train.info()
print("\nTEST INFO:\n"); test.info()
print("\nTRAIN DESCRIBE:\n", train.describe(include='all'))
print("\nTrain missing values:\n", train.isnull().sum())
print("\nTest missing values:\n", test.isnull().sum())

plt.figure(figsize=(8,5))
plt.hist(train['price'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Price'); plt.ylabel('Frequency')
plt.title('Price Distribution (Training Set)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'price_distribution.png'))
plt.close()

Q1 = train['price'].quantile(0.25)
Q3 = train['price'].quantile(0.75)
IQR = Q3 - Q1
outliers = train[(train['price'] < Q1 - 1.5 * IQR) | (train['price'] > Q3 + 1.5 * IQR)]
print(f"\nNumber of outliers (price): {len(outliers)}")
outlier_ids = outliers['sample_id'].tolist()
with open(os.path.join(OUTPUTS_DIR, 'train_outlier_sample_ids.txt'), 'w') as f:
    for i in outlier_ids: f.write(f"{i}\n")

if 'catalog_content' in train:
    train['content_length'] = train['catalog_content'].str.len()
    plt.figure(figsize=(8,5))
    plt.hist(train['content_length'], bins=50, color='orange', edgecolor='black')
    plt.xlabel('Catalog Content Length'); plt.ylabel('Frequency')
    plt.title('Catalog Content Length Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'catalog_content_length_distribution.png'))
    plt.close()

for key in KEYWORDS:
    if key in train:
        plt.figure(figsize=(6,4))
        plt.bar(['0','1'], train[key].value_counts().sort_index())
        plt.xlabel(key)
        plt.ylabel('Count')
        plt.title(f'Distribution of {key}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, f'{key}_distribution.png'))
        plt.close()

if 'image_link' in train:
    print(f"Number of train samples with image links: {train['image_link'].notnull().sum()}")

print("\nEDA complete. Plots and outlier IDs saved in outputs/")
