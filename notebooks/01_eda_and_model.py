"""
Phase 1 & 2: EDA + Decision Tree Model Training
BERT-Based Resume Screening & Authenticity Validation
Trains on resume_dataset_2000_tech.csv with 17 features.
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
import json, os, logging

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger('training')

PROJ = Path(__file__).resolve().parent.parent.parent
SCREENER = PROJ / 'resume-screener'

OUT = SCREENER / 'data' / 'processed'
MODEL_DIR = SCREENER / 'data' / 'models'
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logger.info("=" * 70)
logger.info("PHASE 1: EXPLORATORY DATA ANALYSIS")
logger.info("=" * 70)

logger.info("Loading dataset...")
df = pd.read_csv(PROJ / 'resume_dataset_2000_tech.csv', encoding='latin-1')
logger.info(f"  dataset: {df.shape}")

logger.info("Class Distribution:")
logger.info("\n" + df['classification'].value_counts().to_string())
logger.info("\n" + (df['classification'].value_counts(normalize=True) * 100).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['classification'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'orange', 'red'])
axes[0].set_title('Class Distribution')
axes[0].set_xlabel('Classification')
axes[0].set_ylabel('Count')
for i, v in enumerate(df['classification'].value_counts().values):
    axes[0].text(i, v + 5, str(v), ha='center')

df['risk_level'].value_counts().plot(kind='bar', ax=axes[1], color=['green', 'orange', 'red'])
axes[1].set_title('Risk Level Distribution')
axes[1].set_xlabel('Risk Level')
for i, v in enumerate(df['risk_level'].value_counts().values):
    axes[1].text(i, v + 5, str(v), ha='center')
plt.tight_layout()
plt.savefig(OUT / 'class_distribution.png')
plt.close()
logger.info("  Saved class_distribution.png")

# Feature engineering: create additional features from raw columns
logger.info("Engineering additional features...")

# Count certifications
def count_certs(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    return len([c for c in str(val).split(',') if c.strip()])

# Count skills
def count_skills(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    return len([s for s in str(val).split(',') if s.strip()])

# Encode education level
edu_map = {"Bachelor's": 1, "Master's": 2, "PhD": 3}

df['num_certifications'] = df['certifications'].apply(count_certs)
df['num_skills'] = df['skills'].apply(count_skills)
df['education_level_encoded'] = df['education_level'].map(edu_map).fillna(0).astype(int)
df['has_previous_job'] = df['previous_job_title'].notna().astype(int)

# 17 feature columns for model
feature_cols = [
    'semantic_similarity', 'skill_overlap_score', 'experience_relevance_score',
    'final_match_score', 'overlapping_jobs', 'promotion_speed',
    'experience_graduation_gap', 'skill_density', 'achievement_count',
    'generic_phrase_score', 'gap_years', 'keyword_stuffing_score',
    'years_experience', 'num_certifications', 'num_skills',
    'education_level_encoded', 'has_previous_job'
]

logger.info(f"\nTotal features: {len(feature_cols)}")
logger.info(f"Feature columns: {feature_cols}")

logger.info("\n--- Feature Overview ---")
logger.info(df[feature_cols].describe())

logger.info("\n--- Missing Values ---")
logger.info(df[feature_cols].isnull().sum())

for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())

logger.info("\n--- Class-wise Feature Means ---")
logger.info(df.groupby('classification')[feature_cols].mean().round(4))

plt.figure(figsize=(14, 12))
corr = df[feature_cols + ['risk_level']].copy()
corr['risk_level'] = corr['risk_level'].map({'Low': 0, 'Medium': 1, 'High': 2})
corr_matrix = corr.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix (17 features)')
plt.tight_layout()
plt.savefig(OUT / 'correlation_matrix.png')
plt.close()
logger.info("  Saved correlation_matrix.png")

fig, axes = plt.subplots(5, 4, figsize=(20, 24))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    for cls in ['Authentic', 'Suspicious', 'Potentially Fake']:
        subset = df[df['classification'] == cls][col].dropna()
        sns.kdeplot(subset, label=cls, ax=axes[i], linewidth=2)
    axes[i].set_title(f'{col} by Class')
    axes[i].legend()
for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / 'feature_distributions.png')
plt.close()
logger.info("  Saved feature_distributions.png")

df.to_csv(OUT / 'combined_dataset.csv', index=False)
logger.info(f"\n  Saved combined_dataset.csv ({df.shape})")

logger.info("\n" + "=" * 70)
logger.info("PHASE 2: DECISION TREE MODEL TRAINING")
logger.info("=" * 70)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

label_map = {'Authentic': 0, 'Suspicious': 1, 'Potentially Fake': 2}
df['target'] = df['classification'].map(label_map)

X = df[feature_cols].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logger.info(f"\nTrain set: {X_train.shape}")
logger.info(f"Test set: {X_test.shape}")
logger.info(f"Train distribution: {pd.Series(y_train).value_counts().to_dict()}")

logger.info("\n--- Training Base Decision Tree ---")
base_dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
base_dt.fit(X_train, y_train)
y_pred_base = base_dt.predict(X_test)

base_acc = accuracy_score(y_test, y_pred_base)
base_f1 = f1_score(y_test, y_pred_base, average='weighted')
logger.info(f"  Base DT Accuracy: {base_acc:.4f}")
logger.info(f"  Base DT F1 (weighted): {base_f1:.4f}")
logger.info("\n  Classification Report:")
logger.info(classification_report(y_test, y_pred_base, target_names=['Authentic', 'Suspicious', 'Potentially Fake']))

logger.info("\n--- Grid Search for Best Decision Tree ---")
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

logger.info(f"\n  Best params: {grid.best_params_}")
logger.info(f"  Best CV F1: {grid.best_score_:.4f}")

best_dt = grid.best_estimator_
y_pred = best_dt.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')
logger.info(f"\n  Test Accuracy: {test_acc:.4f}")
logger.info(f"  Test F1 (weighted): {test_f1:.4f}")
logger.info("\n  Classification Report:")
logger.info(classification_report(y_test, y_pred, target_names=['Authentic', 'Suspicious', 'Potentially Fake']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Authentic', 'Suspicious', 'Potentially Fake'],
            yticklabels=['Authentic', 'Suspicious', 'Potentially Fake'])
plt.title(f'Confusion Matrix (Accuracy: {test_acc:.3f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(OUT / 'confusion_matrix.png')
plt.close()
logger.info("  Saved confusion_matrix.png")

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_dt.feature_importances_
}).sort_values('importance', ascending=False)

logger.info("\n--- Feature Importance ---")
logger.info(importance.to_string(index=False))

plt.figure(figsize=(12, 6))
sns.barplot(data=importance, x='importance', y='feature', palette='viridis')
plt.title('Decision Tree Feature Importance (17 features)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(OUT / 'feature_importance.png')
plt.close()
logger.info("  Saved feature_importance.png")

plt.figure(figsize=(24, 16))
plot_tree(best_dt, feature_names=feature_cols,
          class_names=['Authentic', 'Suspicious', 'Potentially Fake'],
          filled=True, rounded=True, fontsize=10, max_depth=4)
plt.title('Decision Tree (max_depth=4 for visualization)')
plt.tight_layout()
plt.savefig(OUT / 'decision_tree.png', dpi=150)
plt.close()
logger.info("  Saved decision_tree.png")

model_path = MODEL_DIR / 'decision_tree_model.pkl'
joblib.dump({
    'model': best_dt,
    'feature_names': feature_cols,
    'label_map': label_map,
    'params': grid.best_params_,
    'test_accuracy': test_acc,
    'test_f1': test_f1,
    'feature_importance': importance.to_dict('records')
}, model_path)
logger.info(f"\n  Model saved to {model_path}")

metrics = {
    'dataset_shape': df.shape,
    'class_distribution': df['classification'].value_counts().to_dict(),
    'feature_cols': feature_cols,
    'new_features_added': ['years_experience', 'num_certifications', 'num_skills',
                           'education_level_encoded', 'has_previous_job'],
    'best_params': grid.best_params_,
    'test_accuracy': float(test_acc),
    'test_f1_weighted': float(test_f1),
    'feature_importance': importance.to_dict('records'),
    'classification_report': classification_report(
        y_test, y_pred, target_names=['Authentic', 'Suspicious', 'Potentially Fake'],
        output_dict=True
    )
}
with open(OUT / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
logger.info("  Saved metrics.json")

logger.info("\n" + "=" * 70)
logger.info("PHASE 1 & 2 COMPLETE!")
logger.info("=" * 70)
