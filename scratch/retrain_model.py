"""
Retrain XGBoost using the existing combined_dataset.csv in the CURRENT Python environment.
Saves:
  - data/models/xgboost_model.pkl   (the real trained model)
  - data/processed/metrics.json           (updated metrics)
  - All PNG visualizations
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json, joblib, logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger('retrain')

BASE = Path(__file__).resolve().parent.parent

OUT       = BASE / 'data' / 'processed'
MODEL_DIR = BASE / 'data' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load the already-processed dataset ──────────────────────────────────────
logger.info("Loading combined_dataset.csv ...")
df = pd.read_csv(OUT / 'combined_dataset.csv', encoding='latin-1')
logger.info(f"  Shape: {df.shape}")
logger.info(f"  Classes: {df['classification'].value_counts().to_dict()}")

# ── Feature columns (19) ─────────────────────
feature_cols = [
    'semantic_similarity', 'skill_overlap_score', 'experience_relevance_score',
    'final_match_score', 'overlapping_jobs', 'promotion_speed',
    'experience_graduation_gap', 'skill_density', 'achievement_count',
    'generic_phrase_score', 'gap_years', 'keyword_stuffing_score',
    'years_experience', 'num_certifications', 'num_skills',
    'education_level_encoded', 'has_previous_job',
    'skill_experience_alignment', 'ai_plausibility_score'
]

# Fill missing values with median
for col in feature_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    else:
        df[col] = 0.0

# ── Train / test split ───────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)

label_map = {'Authentic': 0, 'Suspicious': 1, 'Potentially Fake': 2}
df['target'] = df['classification'].map(label_map)

X = df[feature_cols].values.astype(float)
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── Grid search for best hyperparameters ─────────────────────────────────────
logger.info("Running grid search ...")
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
}

grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='mlogloss', objective='multi:softprob', num_class=3),
    param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0
)
grid.fit(X_train, y_train)

best_model    = grid.best_estimator_
best_params = grid.best_params_
logger.info(f"Best params: {best_params}")

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred   = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1  = f1_score(y_test, y_pred, average='weighted')
logger.info(f"Test Accuracy: {test_acc:.4f}  |  Weighted F1: {test_f1:.4f}")

report = classification_report(
    y_test, y_pred,
    target_names=['Authentic', 'Suspicious', 'Potentially Fake'],
    output_dict=True
)

# ── Feature importance ───────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

importance_records = importance_df.to_dict('records')

# ── Save PNG visualizations ──────────────────────────────────────────────────
logger.info("Generating visualizations ...")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Authentic', 'Suspicious', 'Potentially Fake'],
            yticklabels=['Authentic', 'Suspicious', 'Potentially Fake'])
plt.title(f'Confusion Matrix  (Accuracy: {test_acc:.3f})')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(OUT / 'confusion_matrix.png'); plt.close()

# Feature importance bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
plt.title('XGBoost Feature Importance (19 features)')
plt.xlabel('Importance'); plt.tight_layout()
plt.savefig(OUT / 'feature_importance.png'); plt.close()

# Also copy PNGs to app/static for serving
import shutil
STATIC = BASE / 'app' / 'static'
for png in ['confusion_matrix.png', 'correlation_matrix.png',
            'feature_distributions.png', 'feature_importance.png',
            'class_distribution.png']:
    src = OUT / png
    if src.exists():
        shutil.copy2(src, STATIC / png)
logger.info("  PNGs synced to app/static/")

# ── Save model .pkl (current-env compatible) ─────────────────────────────────
model_path = MODEL_DIR / 'xgboost_model.pkl'
joblib.dump({
    'model':              best_model,
    'feature_names':      feature_cols,
    'label_map':          label_map,
    'params':             best_params,
    'test_accuracy':      float(test_acc),
    'test_f1':            float(test_f1),
    'feature_importance': importance_records,
}, model_path)
logger.info(f"  Model saved -> {model_path}")

# ── Update metrics.json ───────────────────────────────────────────────────────
metrics = {
    'dataset_shape':       list(df.shape),
    'class_distribution':  df['classification'].value_counts().to_dict(),
    'feature_cols':        feature_cols,
    'new_features_added':  ['years_experience', 'num_certifications', 'num_skills',
                            'education_level_encoded', 'has_previous_job'],
    'best_params':         best_params,
    'test_accuracy':       float(test_acc),
    'test_f1_weighted':    float(test_f1),
    'feature_importance':  importance_records,
    'classification_report': report,
}
with open(OUT / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
logger.info("  metrics.json updated")

logger.info("=" * 60)
logger.info(f"RETRAINING COMPLETE — Accuracy: {test_acc*100:.2f}%  F1: {test_f1*100:.2f}%")
logger.info("=" * 60)
