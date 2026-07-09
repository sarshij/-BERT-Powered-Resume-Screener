"""
XGBoost Classifier Wrapper
Loads the trained model from data/models/xgboost_model.pkl and runs predictions.
Integrates SHAP to provide local explainability (top 3 contributing features).
"""
import joblib
import numpy as np
import pandas as pd
import json
import shap
from pathlib import Path

from app.logger import setup_logger

logger = setup_logger(__name__)

# Path to the trained XGBoost model
MODEL_PATH = Path(__file__).parents[2] / 'data' / 'models' / 'xgboost_model.pkl'
METRICS_PATH = Path(__file__).parents[2] / 'data' / 'processed' / 'metrics.json'

_LABEL_MAP = {0: 'Authentic', 1: 'Suspicious', 2: 'Potentially Fake'}

_loaded_model = None       
_feature_names = None      
_model_meta = None         
_explainer = None          # SHAP explainer

def _load_feature_names_from_metrics() -> list:
    try:
        with open(METRICS_PATH, 'r') as f:
            data = json.load(f)
        cols = data.get('feature_cols')
        if cols:
            return cols
    except Exception as e:
        logger.warning(f"Could not load feature names from metrics.json: {e}")

    # Fallback 19 features
    return [
        "semantic_similarity", "skill_overlap_score", "experience_relevance_score",
        "final_match_score", "overlapping_jobs", "promotion_speed",
        "experience_graduation_gap", "skill_density", "achievement_count",
        "generic_phrase_score", "gap_years", "keyword_stuffing_score",
        "years_experience", "num_certifications", "num_skills",
        "education_level_encoded", "has_previous_job",
        "skill_experience_alignment", "ai_plausibility_score"
    ]


def _load_model_from_pkl():
    logger.info(f"Loading trained model from: {MODEL_PATH}")
    payload = joblib.load(MODEL_PATH)
    if isinstance(payload, dict):
        model = payload['model']
        feat_names = payload.get('feature_names', _load_feature_names_from_metrics())
        return model, feat_names, payload
    else:
        feat_names = _load_feature_names_from_metrics()
        return payload, feat_names, {}


def load_model():
    global _loaded_model, _feature_names, _model_meta, _explainer

    if _loaded_model is not None:
        return _loaded_model, _feature_names

    try:
        _loaded_model, _feature_names, _model_meta = _load_model_from_pkl()
        # Initialize SHAP explainer
        # TreeExplainer is fast and exact for XGBoost
        try:
            _explainer = shap.TreeExplainer(_loaded_model)
        except Exception as shap_e:
            logger.error(f"SHAP explainer failed to initialize (known bug with XGBoost 2.0+ JSON parsing): {shap_e}")
            _explainer = None
        return _loaded_model, _feature_names
    except Exception as e:
        logger.error(f"Failed to load XGBoost pkl ({e}). Using fallback.")

    _feature_names = _load_feature_names_from_metrics()
    _loaded_model = _HeuristicFallbackClassifier()
    _model_meta = {}
    return _loaded_model, _feature_names


class _HeuristicFallbackClassifier:
    def _classify(self, row: np.ndarray, cols: list) -> tuple:
        col_idx = {name: i for i, name in enumerate(cols)}
        match   = float(row[col_idx.get("final_match_score", 3)])
        generic = float(row[col_idx.get("generic_phrase_score", 9)])
        stuffing = float(row[col_idx.get("keyword_stuffing_score", 11)])
        density  = float(row[col_idx.get("skill_density", 7)])

        if generic >= 0.60 or match < 0.20:
            cls = 2
            proba = np.array([0.05, 0.10, 0.85])
        elif generic >= 0.40 or stuffing >= 0.60 or match < 0.40:
            cls = 1
            proba = np.array([0.15, 0.70, 0.15])
        else:
            cls = 0
            confidence = min(0.95, 0.60 + match * 0.35 + density * 0.01)
            proba = np.array([confidence, (1 - confidence) * 0.6, (1 - confidence) * 0.4])
        return cls, proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._classify(row, self._cols)[0] for row in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._classify(row, self._cols)[1] for row in X])


def predict(features: dict | list[dict]) -> list[dict]:
    model, cols = load_model()

    if isinstance(features, dict):
        features = [features]

    df = pd.DataFrame(features)
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[cols].fillna(0.0).values.astype(float)

    if isinstance(model, _HeuristicFallbackClassifier):
        model._cols = cols
        logger.info("🟡 \033[1;33m[Prediction Engine: FALLBACK HEURISTIC] -> Using dummy/fallback logic.\033[0m")
    else:
        logger.info("🟢 \033[1;32m[Prediction Engine: REAL XGBOOST] -> Using trained ML model.\033[0m")

    classes = model.predict(X)
    probs   = model.predict_proba(X)
    
    # Calculate SHAP values for explainability
    shap_vals = None
    if _explainer is not None:
        try:
            # Output of TreeExplainer for multi-class is a list of arrays (one for each class) or an array of shape (n_samples, n_features, n_classes) depending on version.
            # We want the explanation for the predicted class.
            shap_values = _explainer.shap_values(X)
            # Handle different SHAP output formats for multiclass
            if isinstance(shap_values, list): 
                shap_vals = shap_values
            elif len(shap_values.shape) == 3: # (n_samples, n_features, n_classes)
                # Convert to list of (n_samples, n_features) for consistency
                shap_vals = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")

    results = []
    for i, cls in enumerate(classes):
        cls_int = int(cls)
        label      = _LABEL_MAP.get(cls_int, 'Unknown')
        confidence = float(probs[i][cls_int])
        
        result = {
            'classification': label,
            'confidence': round(confidence, 4),
        }
        for j, class_label in _LABEL_MAP.items():
            result[f'prob_{class_label}'] = round(float(probs[i][j]), 4)
            
        # Add SHAP explanation (Top 3 features pushing towards this prediction)
        if shap_vals is not None:
            try:
                # Get the SHAP values for the predicted class for this instance
                instance_shap = shap_vals[cls_int][i]
                # Sort features by absolute SHAP value (impact magnitude)
                top_indices = np.argsort(np.abs(instance_shap))[::-1][:3]
                
                explanations = []
                for idx in top_indices:
                    feat_name = cols[idx]
                    shap_val = float(instance_shap[idx])
                    feat_val = float(X[i][idx])
                    explanations.append({
                        'feature': feat_name,
                        'value': round(feat_val, 4),
                        'contribution': round(shap_val, 4)
                    })
                result['top_features'] = explanations
            except Exception as e:
                logger.error(f"Failed to extract SHAP features: {e}")
                
        results.append(result)

    return results


def get_feature_importance() -> list[dict]:
    load_model()
    if _model_meta and 'feature_importance' in _model_meta:
        return _model_meta['feature_importance']
    try:
        with open(METRICS_PATH, 'r') as f:
            data = json.load(f)
        if 'feature_importance' in data:
            return data['feature_importance']
    except Exception:
        pass
    names = _feature_names or _load_feature_names_from_metrics()
    return [{'feature': f, 'importance': 0.0} for f in names]


def get_model_info() -> dict:
    load_model()
    if _model_meta:
        return {
            'feature_names':      _feature_names,
            'params':             _model_meta.get('params', {}),
            'test_accuracy':      _model_meta.get('test_accuracy', 0.0),
            'test_f1':            _model_meta.get('test_f1', 0.0),
            'feature_importance': get_feature_importance(),
            'classes':            list(_LABEL_MAP.values()),
        }
    try:
        with open(METRICS_PATH, 'r') as f:
            data = json.load(f)
        return {
            'feature_names':      data.get('feature_cols', _feature_names),
            'params':             data.get('best_params', {}),
            'test_accuracy':      data.get('test_accuracy', 0.0),
            'test_f1':            data.get('test_f1_weighted', 0.0),
            'feature_importance': data.get('feature_importance', []),
            'classes':            list(_LABEL_MAP.values()),
        }
    except Exception as e:
        return {
            'feature_names':      _feature_names or [],
            'params':             {},
            'test_accuracy':      0.0,
            'test_f1':            0.0,
            'feature_importance': [],
            'classes':            list(_LABEL_MAP.values()),
        }
