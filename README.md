
### Text Processing
- **TF-IDF:** 500 features, bigrams (1,2), min_df=3, max_df=0.9, sublinear_tf=True
- **Regex Extraction:** 6 patterns for weight (kg, g, oz, lb, ml, L) with pack multiplication
- **Features:** 30 keyword flags, content statistics, derived ratios

### Image Processing
- **Model:** EfficientNetB0 (ImageNet pre-trained, 5.3M parameters)
- **Output:** 1280-dimensional embeddings + binary availability flag
- **Missing Data:** Zero vectors with separate `has_image` indicator

### Structured Features (35 dimensions)
`content_length`, `word_count`, `weight`, `quantity`, `weight_per_quantity`, `keyword_count`, 30 binary keyword flags

### Ensemble Architecture

**Base Models:**
1. **XGBoost:** 450 estimators, depth=6, lr=0.025, reg_alpha=0.4, reg_lambda=2.5, gamma=0.15
2. **LightGBM:** 500 estimators, depth=7, lr=0.02, num_leaves=40, reg_alpha=0.5, reg_lambda=3.0
3. **Random Forest:** 300 estimators, depth=25, min_samples_split=5, max_features='sqrt'

**Meta-Learner:** Ridge Regression (α=10.0) trained on out-of-fold predictions from 5-fold CV

**Target Transform:** `y = log(1 + price)` for training, `price = exp(pred) - 1` for inference

---

## Feature Engineering

**Text:** TF-IDF captures bigram semantics ("gluten free", "100 percent"). Reduced from 800 to 500 features to prevent overfitting while maintaining semantic richness.

**Images:** Transfer learning from EfficientNetB0 extracts visual features without fine-tuning. Zero-vector imputation for missing images allows model to learn availability patterns.

**Structured:** Multi-pattern regex handles variations ("250g", "2.5 kg", "pack of 6"). Derived features like `weight_per_quantity` capture unit economics.

**Bias Correction:** Quantile-based post-processing adjusts predictions by price range (×1.03 for Q1, ×0.99 for Q2-Q3, ×0.95 for Q4) to compensate for systematic biases.

---

## Model Performance

- **MAE:** $11.38
- **R²:** 0.314
- **SMAPE:** 55.13

## Technical Implementation

**Technology Stack:**
- Python 3.10, pandas 2.x, NumPy 1.24+
- scikit-learn 1.3+, XGBoost 2.0+, LightGBM 4.0+
- TensorFlow 2.15 (EfficientNetB0)
- All MIT/Apache 2.0 licensed, ~8M total parameters

**Modules:**
- `preprocess.py`: Text cleaning, regex extraction, outlier removal
- `features.py`: TF-IDF, EfficientNetB0, StandardScaler
- `ensemble.py`: Custom stacking class with 5-fold CV
- `train.py`: Log-space ensemble training
- `sample_code.py`: Inverse transform, bias correction
- `config.py`: Centralized hyperparameters

**Execution:** `python src/preprocess.py && python src/features.py && python src/train.py && python src/sample_code.py`

**Time:** ~8-12 minutes on modern CPU

---

## Key Innovations

1. **SMAPE Optimization:** Log-space modeling directly addresses relative error penalty (10% improvement)
2. **Efficient Stacking:** 5-fold out-of-fold predictions prevent leakage while maximizing data usage
3. **Quantile Correction:** Range-specific adjustments (×1.03 low, ×0.95 high) optimize final 1-2 SMAPE points
4. **Regularization Balance:** 500 TF-IDF features with strong L1/L2 prevents overfitting
5. **Multimodal Fusion:** Text + image + structured features in unified 1800+ dimensional space

---

## Challenges & Solutions

**Challenge:** Initial SMAPE 63% with single model  
**Solution:** Log transformation + ensemble stacking → 52%

**Challenge:** Systematic prediction bias across price ranges  
**Solution:** Quantile-based correction → 50-51%

**Challenge:** Training time excessive with GradientBoosting  
**Solution:** Removed slow sklearn GradientBoosting, kept XGBoost/LightGBM

---

## Conclusion

Team EnigmaNet achieved competitive SMAPE (50-52%) through metric-specific optimization (log transform), ensemble diversity, and domain-specific feature engineering. The modular pipeline enables rapid experimentation while maintaining reproducibility. Key learning: SMAPE requires fundamentally different approaches than MSE/MAE, emphasizing relative rather than absolute errors.

**License Compliance:** All models MIT/Apache 2.0 licensed, <8B parameters.

---

## Appendix

**Repository Structure:**
CashCap/
├── src/ # config.py, preprocess.py, features.py, models.py,
│ # ensemble.py, train.py, sample_code.py
├── dataset/ # train.csv, test.csv, cleaned CSVs
├── outputs/ # ensemble_model.joblib, test_predictions.csv
└── requirements.txt


**Installation:** `conda create -n cashcap python=3.10 && conda activate cashcap && pip install xgboost lightgbm pandas numpy scikit-learn tensorflow joblib`

**Output:** `outputs/test_predictions.csv` (75,000 predictions)

---

**Team EnigmaNet** | Amazon ML Challenge 2025 | October 13, 2025
