import numpy as np
from sklearn.model_selection import KFold
import time

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"

class StackingEnsemble:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.trained_base_models = []
    
    def fit(self, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        total_tasks = len(self.base_models) * 5
        completed_tasks = 0
        start_time = time.time()
        
        print(f"\nTraining {len(self.base_models)} models with 5-fold CV...")
        print(f"Total tasks: {total_tasks}\n")
        
        for i, (name, model) in enumerate(self.base_models):
            oof_predictions = np.zeros(len(X))
            model_start = time.time()
            
            print(f"[Model {i+1}/{len(self.base_models)}] {name}")
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                fold_start = time.time()
                
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train = y[train_idx]
                
                if hasattr(model, 'get_params'):
                    fold_model = model.__class__(**model.get_params())
                else:
                    fold_model = model
                
                fold_model.fit(X_fold_train, y_fold_train)
                oof_predictions[val_idx] = fold_model.predict(X_fold_val)
                
                completed_tasks += 1
                elapsed = time.time() - start_time
                avg_time_per_task = elapsed / completed_tasks
                remaining_tasks = total_tasks - completed_tasks
                eta = avg_time_per_task * remaining_tasks
                
                fold_time = time.time() - fold_start
                progress = (completed_tasks / total_tasks) * 100
                
                print(f"  Fold {fold+1}/5 | {fold_time:.1f}s | Progress: {progress:.1f}% | ETA: {format_time(eta)}")
            
            meta_features[:, i] = oof_predictions
            model.fit(X, y)
            self.trained_base_models.append(model)
            
            model_time = time.time() - model_start
            print(f"  ✓ {name} complete ({format_time(model_time)})\n")
        
        print("Training meta-model...")
        self.meta_model.fit(meta_features, y)
        
        total_time = time.time() - start_time
        print(f"✓ Ensemble training complete ({format_time(total_time)})\n")
        
        return self
    
    def predict(self, X):
        base_predictions = np.zeros((len(X), len(self.trained_base_models)))
        for i, model in enumerate(self.trained_base_models):
            base_predictions[:, i] = model.predict(X)
        return self.meta_model.predict(base_predictions)
