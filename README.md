# 🏛️ **Aureus – Advanced Product Price Prediction System**  

---

## 🧩 Overview  
**Aureus** is a machine learning pipeline built to **predict product prices** from catalog data by combining text, structured, and statistical features.  
It provides a modular, extensible, and automated workflow covering everything — from **data cleaning**, **feature extraction**, and **exploratory analysis**, to **model training**, **stacking**, and **prediction generation**.

The project focuses on **efficiency**, **scalability**, and **explainable results**, integrating multiple models and analytical visualizations.

---

## ✨ Key Features  

✅ **Automated Data Preprocessing** – Cleans raw product catalogs, extracts key numeric/text attributes, and removes outliers.  
✅ **Feature Engineering Pipeline** – Combines TF-IDF, structured numerical attributes, and engineered indicators.  
✅ **Model Ensemble (Stacking)** – Combines XGBoost, LightGBM, and Random Forest regressors with Ridge as a meta-model.  
✅ **Exploratory Data Analysis (EDA)** – Generates key data distribution plots for better insights.  
✅ **Performance Metrics** – Evaluates using MAE, R², and SMAPE.  
✅ **Prediction Pipeline** – Generates final test predictions (`test_predictions.csv`).  
✅ **Configurable System** – All paths, parameters, and constants are centralized in `config.py`.

---

## 🛠️ Tech Stack  

| Layer | Tools / Libraries |
|-------|-------------------|
| **Language** | Python 3.10+ |
| **Data Processing** | pandas, numpy |
| **Feature Engineering** | scikit-learn (TF-IDF, scaling) |
| **Model Training** | XGBoost, LightGBM, RandomForest, Ridge |
| **Visualization (EDA)** | matplotlib, seaborn |
| **Model Persistence** | joblib |
| **Deep Learning (optional)** | TensorFlow (for image features) |

---

## 📁 Project Structure  

```
📦 Aureus
 ┣ 📂 .vscode
 ┃ ┗ 📜 settings.json
 ┣ 📂 dataset
 ┃ ┣ 📜 train.csv                 # Raw training data
 ┃ ┣ 📜 test.csv                  # Raw test data
 ┃ ┣ 📜 train_cleaned.csv         # Cleaned training data
 ┃ ┣ 📜 test_cleaned.csv          # Cleaned test data
 ┃ ┣ 📜 sample_test.csv           # Sample test input
 ┣ 📂 outputs
 ┃ ┣ 📜 train_features.npy        # Processed training features
 ┃ ┣ 📜 test_features.npy         # Processed test features
 ┃ ┣ 📜 rf_model.joblib           # Trained RandomForest model
 ┃ ┣ 📜 xgb_model.joblib          # Trained XGBoost model
 ┃ ┣ 📜 ensemble_model.joblib     # Final stacking model
 ┃ ┣ 📜 feature_scaler.joblib     # Scaler for normalization
 ┃ ┣ 📜 tfidf_vectorizer.joblib   # TF-IDF vectorizer
 ┃ ┣ 📜 test_predictions.csv      # Final predictions output
 ┃ ┣ 📜 price_distribution.png    # EDA visualization
 ┃ ┣ 📜 content_length_distribution.png
 ┃ ┣ 📜 catalog_content_length_distribution.png
 ┃ ┣ 📜 eda_summary.txt
 ┣ 📂 src
 ┃ ┣ 📜 config.py                 # Configuration and parameters
 ┃ ┣ 📜 eda.py                    # Exploratory data analysis
 ┃ ┣ 📜 preprocess.py             # Data cleaning and feature extraction
 ┃ ┣ 📜 features.py               # Feature generation (TF-IDF + structured)
 ┃ ┣ 📜 models.py                 # Model constructors
 ┃ ┣ 📜 ensemble.py               # Stacking ensemble class
 ┃ ┣ 📜 train.py                  # Model training & validation
 ┃ ┣ 📜 sample_code.py            # Inference / prediction script
 ┃ ┣ 📜 utils.py                  # Helper functions
 ┣ 📜 README.md
 ┣ 📜 requirements.txt
 ┣ 📜 file_structure.csv
 ┣ 📜 .gitignore
 ┣ 📜 .gitattributes
```

---

## ⚙️ Configuration  

All parameters are defined in `src/config.py`, including:  
- File paths for dataset, outputs, models  
- Model hyperparameters (RF, XGBoost, LightGBM)  
- TF-IDF feature extraction settings  
- Random seeds, cross-validation splits, and outlier removal thresholds  

---

## 🚀 Getting Started  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/shashwat42/Aureus.git
cd Aureus
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```
*(Or manually install the following:)*  
```bash
pip install xgboost lightgbm scikit-learn pandas numpy joblib tensorflow tqdm matplotlib seaborn
```

### 3️⃣ Prepare Dataset  
Place your raw CSV files in the `dataset/` directory:
```
dataset/
 ┣ train.csv
 ┣ test.csv
```

### 4️⃣ Run the Full ML Pipeline  

```bash
# Step 1: Clean and preprocess data
python src/preprocess.py

# Step 2: Feature engineering
python src/features.py

# Step 3: Train ensemble models
python src/train.py

# Step 4: Generate predictions
python src/sample_code.py
```

After successful execution, predictions will be saved to:
```
outputs/test_predictions.csv
```

---

## 📈 Model Architecture  

**Stacking Ensemble Framework**

```
 ┌─────────────────────────────┐
 │     Random Forest Model     │
 ├─────────────────────────────┤
 │       XGBoost Model         │
 ├─────────────────────────────┤
 │       LightGBM Model        │
 └──────────────┬──────────────┘
                │
                ▼
      Ridge Regression (Meta-Model)
                │
                ▼
        Final Price Predictions
```

---

## 📊 Evaluation Metrics  

| Metric | Description |
|---------|-------------|
| **MAE** | Mean Absolute Error – average prediction deviation |
| **R² Score** | Measures model’s goodness of fit |
| **SMAPE** | Symmetric Mean Absolute Percentage Error |

Validation metrics are displayed automatically during training.

---

## 🧠 Exploratory Data Analysis (EDA)

- Conducted via `src/eda.py`  
- Generates plots like `price_distribution.png`, `content_length_distribution.png`  
- Summarizes data insights in `outputs/eda_summary.txt`

---

## 🧪 Example Results

| Metric | Value (example) |
|---------|----------------|
| MAE | 12.34 |
| R² | 0.91 |
| SMAPE | 41.6546% |

*(Actual values depend on dataset and parameters.)*

---

## 📜 License  

This project is open-source under the **MIT License**.  
You are free to use, modify, and distribute with attribution.

---

## 🧭 Summary  

Aureus demonstrates a complete **end-to-end ML workflow** for price prediction, integrating:  
- Advanced feature engineering  
- Automated data cleaning  
- Stacking ensemble modeling  
- Streamlined pipeline execution  

> 💡 Use Aureus as a base template for future ML competitions, product catalog analysis, or price intelligence systems.
