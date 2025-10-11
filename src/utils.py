import os
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import multiprocessing
from functools import partial
from tqdm import tqdm

def download_image(image_link, savefolder):
    if isinstance(image_link, str) and image_link:
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if not os.path.exists(image_save_path):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
            except Exception as ex:
                print(f'Warning: Failed to download - {image_link}\n{ex}')

def download_images(image_links, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    results = []
    download_partial = partial(download_image, savefolder=download_folder)
    with multiprocessing.Pool(8) as pool:
        for result in tqdm(pool.imap_unordered(download_partial, image_links), total=len(image_links)):
            results.append(result)
        pool.close()
        pool.join()

def ensure_download_for_csv(image_link_column, download_folder):
    image_links = image_link_column.dropna().unique()
    download_images(image_links, download_folder)

def safe_image_path(image_url, image_dir):
    fname = Path(image_url).name
    return os.path.join(image_dir, fname)

def evaluate_predictions(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    smape = np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation SMAPE: {smape:.2f}%")
    return mae, smape

def load_features(features_path):
    return np.load(features_path)

def load_model(model_path):
    import joblib
    return joblib.load(model_path)

def save_model(model, model_path):
    import joblib
    joblib.dump(model, model_path)

def save_predictions(ids, preds, out_path):
    df = pd.DataFrame({'sample_id': ids, 'price': preds})
    df.to_csv(out_path, index=False)
