"""
SBERT Embedding Pipeline
Generates sentence embeddings for resume text and job descriptions.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from app.logger import setup_logger

logger = setup_logger(__name__)

_MODEL = None

def get_model(model_name: str = 'all-MiniLM-L6-v2'):
    global _MODEL
    if _MODEL is None:
        logger.info(f"Loading SBERT model: {model_name}")
        _MODEL = SentenceTransformer(model_name)
        logger.info("SBERT model loaded")
    return _MODEL

def embed_texts(texts: list[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    if not texts:
        return np.array([])
    model = get_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)
    dot = np.dot(emb1, emb2.T).flatten()
    return float(np.clip(dot[0], 0, 1)) if len(dot) > 0 else 0.0
