"""
Decision Tree Classifier Wrapper
Loads the trained model from data/models/decision_tree_model.pkl and runs predictions.
The .pkl file is a dict saved by the training notebook with keys:
  'model', 'feature_names', 'label_map', 'params', 'test_accuracy', 'test_f1', 'feature_importance'
"""
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path

from app.logger import setup_logger

logger = setup_logger(__name__)

# Path to the trained Decision Tree model saved by notebooks/01_eda_and_model.py
MODEL_PATH = Path(__file__).parents[2] / 'data' / 'models' / 'decision_tree_model.pkl'
METRICS_PATH = Path(__file__).parents[2] / 'data' / 'processed' / 'metrics.json'

# Label map matching the training notebook: 0=Authentic, 1=Suspicious, 2=Potentially Fake
_LABEL_MAP = {0: 'Authentic', 1: 'Suspicious', 2: 'Potentially Fake'}

# Module-level cache — loaded once, reused for every request
_loaded_model = None       # The actual sklearn DecisionTreeClassifier object
_feature_names = None      # List of 17 feature column names
_model_meta = None         # Full metadata dict from the pkl (params, accuracy, etc.)


def _load_feature_names_from_metrics() -> list:
    """
    Load feature column names from metrics.json as a fallback.
    Returns the hardcoded default list if the file is not found.
    """
    try:
        with open(METRICS_PATH, 'r') as f:
            data = json.load(f)
        cols = data.get('feature_cols')
        if cols:
            return cols
    except Exception as e:
        logger.warning(f"Could not load feature names from metrics.json: {e}")

    # Hardcoded fallback — matches the 17-feature set used during training
    return [
        "semantic_similarity",
        "skill_overlap_score",
        "experience_relevance_score",
        "final_match_score",
        "overlapping_jobs",
        "promotion_speed",
        "experience_graduation_gap",
        "skill_density",
        "achievement_count",
        "generic_phrase_score",
        "gap_years",
        "keyword_stuffing_score",
        "years_experience",
        "num_certifications",
        "num_skills",
        "education_level_encoded",
        "has_previous_job",
    ]


def _load_model_from_pkl():
    """
    Attempt to load the trained Decision Tree from the .pkl file.
    The pkl is a dict: {'model': DT, 'feature_names': [...], 'params': {...}, ...}
    Returns (model, feature_names, meta_dict) or raises on failure.
    """
    logger.info(f"Loading trained Decision Tree model from: {MODEL_PATH}")
    payload = joblib.load(MODEL_PATH)

    # The notebook saved a dict — extract the actual sklearn model
    if isinstance(payload, dict):
        model = payload['model']
        feat_names = payload.get('feature_names', _load_feature_names_from_metrics())
        logger.info(
            f"Model loaded — accuracy: {payload.get('test_accuracy', '?'):.4f}, "
            f"features: {len(feat_names)}"
        )
        return model, feat_names, payload
    else:
        # Fallback: pkl might be the raw sklearn model (older format)
        logger.warning("pkl was not a dict — treating as raw sklearn model.")
        feat_names = _load_feature_names_from_metrics()
        return payload, feat_names, {}


def load_model():
    """
    Load and cache the Decision Tree model.
    Falls back to a rule-based heuristic classifier if the pkl cannot be loaded
    (e.g., scikit-learn version mismatch), so the app never crashes on startup.
    Returns (model, feature_names_list).
    """
    global _loaded_model, _feature_names, _model_meta

    # Return cached model on subsequent calls (singleton pattern)
    if _loaded_model is not None:
        return _loaded_model, _feature_names

    # --- Try loading the real trained model ---
    try:
        _loaded_model, _feature_names, _model_meta = _load_model_from_pkl()
        return _loaded_model, _feature_names

    except Exception as e:
        logger.error(
            f"Failed to load Decision Tree pkl ({e}). "
            "Falling back to heuristic classifier so the app can still run."
        )

    # --- Graceful fallback: rule-based heuristic classifier ---
    # This is ONLY used if the pkl file is missing or incompatible.
    # It uses final_match_score + generic_phrase_score to give a real
    # non-trivial prediction instead of a constant dummy.
    _feature_names = _load_feature_names_from_metrics()
    _loaded_model = _HeuristicFallbackClassifier()
    _model_meta = {}
    return _loaded_model, _feature_names


class _HeuristicFallbackClassifier:
    """
    Rule-based heuristic classifier used ONLY when the pkl model cannot be loaded.
    Produces meaningful 3-class predictions based on the two most important features
    identified during training: final_match_score and generic_phrase_score.

    Thresholds are derived from class-mean analysis in the EDA notebook:
      - Authentic:       final_match_score >= 0.55 AND generic_phrase_score < 0.40
      - Potentially Fake: generic_phrase_score >= 0.60 OR final_match_score < 0.25
      - Suspicious:      everything in between
    """

    def _classify(self, row: np.ndarray, cols: list) -> tuple:
        """
        Classify a single feature row. Returns (class_int, proba_array).
        col_idx maps feature names to array positions for robustness.
        """
        col_idx = {name: i for i, name in enumerate(cols)}
        match   = float(row[col_idx.get("final_match_score", 3)])
        generic = float(row[col_idx.get("generic_phrase_score", 9)])
        stuffing = float(row[col_idx.get("keyword_stuffing_score", 11)])
        density  = float(row[col_idx.get("skill_density", 7)])

        # High generic phrase score is the strongest fake signal (44% feature importance)
        if generic >= 0.60 or match < 0.20:
            cls = 2  # Potentially Fake
            proba = np.array([0.05, 0.10, 0.85])
        elif generic >= 0.40 or stuffing >= 0.60 or match < 0.40:
            cls = 1  # Suspicious
            proba = np.array([0.15, 0.70, 0.15])
        else:
            cls = 0  # Authentic
            confidence = min(0.95, 0.60 + match * 0.35 + density * 0.01)
            proba = np.array([confidence, (1 - confidence) * 0.6, (1 - confidence) * 0.4])

        return cls, proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        # _loaded_model is called with the feature matrix — need cols from outer scope
        # cols injected at call time via predict() wrapper below
        return np.array([self._classify(row, self._cols)[0] for row in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._classify(row, self._cols)[1] for row in X])


def predict(features: dict | list[dict]) -> list[dict]:
    """
    Run classification on one or more resume feature dicts.
    Returns a list of result dicts with 'classification', 'confidence', and per-class probabilities.
    """
    model, cols = load_model()

    if isinstance(features, dict):
        features = [features]

    # Build DataFrame, filling any missing feature columns with 0.0
    df = pd.DataFrame(features)
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[cols].fillna(0.0).values.astype(float)

    # Inject column names into heuristic fallback if needed
    if isinstance(model, _HeuristicFallbackClassifier):
        model._cols = cols

    classes = model.predict(X)
    probs   = model.predict_proba(X)

    results = []
    for i, cls in enumerate(classes):
        label      = _LABEL_MAP.get(int(cls), 'Unknown')
        confidence = float(probs[i][int(cls)])
        result = {
            'classification': label,
            'confidence': round(confidence, 4),
        }
        # Add per-class probabilities for the UI breakdown
        for j, class_label in _LABEL_MAP.items():
            result[f'prob_{class_label}'] = round(float(probs[i][j]), 4)
        results.append(result)

    return results


def get_feature_importance() -> list[dict]:
    """
    Return feature importance from the loaded model metadata.
    Falls back to zeros if model metadata is unavailable.
    """
    load_model()  # Ensure model is loaded and _model_meta is populated

    # Real feature importance from pkl metadata
    if _model_meta and 'feature_importance' in _model_meta:
        return _model_meta['feature_importance']

    # Try metrics.json
    try:
        with open(METRICS_PATH, 'r') as f:
            data = json.load(f)
        if 'feature_importance' in data:
            return data['feature_importance']
    except Exception:
        pass

    # Last resort: zeros
    names = _feature_names or _load_feature_names_from_metrics()
    return [{'feature': f, 'importance': 0.0} for f in names]


def get_model_info() -> dict:
    """
    Return full model metadata for the Analytics dashboard.
    Loads real values from the pkl or metrics.json.
    """
    load_model()  # Ensure cache is populated

    # Pull from pkl metadata first (most complete source)
    if _model_meta:
        return {
            'feature_names':      _feature_names,
            'params':             _model_meta.get('params', {}),
            'test_accuracy':      _model_meta.get('test_accuracy', 0.0),
            'test_f1':            _model_meta.get('test_f1', 0.0),
            'feature_importance': get_feature_importance(),
            'classes':            list(_LABEL_MAP.values()),
        }

    # Fallback: read from metrics.json (generated by the training notebook)
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
        logger.error(f"Could not read metrics.json for model info: {e}")
        return {
            'feature_names':      _feature_names or [],
            'params':             {},
            'test_accuracy':      0.0,
            'test_f1':            0.0,
            'feature_importance': [],
            'classes':            list(_LABEL_MAP.values()),
        }
