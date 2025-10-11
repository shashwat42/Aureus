# src/utils.py

import re
import os
import pandas as pd
import multiprocessing
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def download_image(image_link, savefolder):
    if(isinstance(image_link, str)):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if(not os.path.exists(image_save_path)):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)    
            except Exception as ex:
                print('Warning: Not able to download - {}\n{}'.format(image_link, ex))
        else:
            return
    return

def download_images(image_links, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    results = []
    download_image_partial = partial(download_image, savefolder=download_folder)
    with multiprocessing.Pool(100) as pool:
        for result in tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)):
            results.append(result)
        pool.close()
        pool.join()

def load_features(features_path):
    return np.load(features_path)

def load_model(model_path):
    return joblib.load(model_path)

def save_model(model, model_path):
    joblib.dump(model, model_path)

def save_predictions(ids, preds, out_path):
    df = pd.DataFrame({'id': ids, 'predicted_price': preds})
    df.to_csv(out_path, index=False)

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"R2: {r2:.3f}")
    return mae, mse, r2
