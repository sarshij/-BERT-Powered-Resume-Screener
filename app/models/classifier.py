"""
Decision Tree Classifier Wrapper
Loads the trained model and runs predictions.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from app.logger import setup_logger

logger = setup_logger(__name__)

MODEL_PATH = Path(__file__).parents[2] / 'data' / 'models' / 'decision_tree_model.pkl'
_LABEL_MAP = {0: 'Authentic', 1: 'Suspicious', 2: 'Potentially Fake'}
_FEATURE_COLS = None

_loaded = None

def load_model():
    global _loaded, _FEATURE_COLS
    if _loaded is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        data = joblib.load(MODEL_PATH)
        _loaded = data['model']
        _FEATURE_COLS = data['feature_names']
        logger.info(f"Model loaded: max_depth={data['params']['max_depth']}, features={len(_FEATURE_COLS)}, accuracy={data['test_accuracy']:.2%}")
    return _loaded, _FEATURE_COLS

def predict(features: dict | list[dict]) -> list[dict]:
    model, cols = load_model()
    if isinstance(features, dict):
        features = [features]
    df = pd.DataFrame(features)
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
    X = df[cols].values
    classes = model.predict(X)
    probs = model.predict_proba(X)
    results = []
    for i, cls in enumerate(classes):
        label = _LABEL_MAP.get(int(cls), 'Unknown')
        confidence = float(probs[i][int(cls)])
        result = {'classification': label, 'confidence': round(confidence, 4)}
        for j, c in enumerate(_LABEL_MAP.values()):
            result[f'prob_{c}'] = round(float(probs[i][j]), 4)
        results.append(result)
    return results

def get_feature_importance():
    data = joblib.load(MODEL_PATH)
    return data.get('feature_importance', [])

def get_model_info():
    data = joblib.load(MODEL_PATH)
    return {
        'feature_names': data['feature_names'],
        'params': data['params'],
        'test_accuracy': data['test_accuracy'],
        'test_f1': data['test_f1'],
        'feature_importance': data['feature_importance'],
        'classes': list(_LABEL_MAP.values())
    }
