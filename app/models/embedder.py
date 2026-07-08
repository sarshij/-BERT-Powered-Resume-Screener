"""
SBERT Embedding Pipeline
Generates sentence embeddings for resume text and job descriptions.
"""
import numpy as np
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer

from app.logger import setup_logger

logger = setup_logger(__name__)

_MODEL = None
_MODEL_NAME = None
_EMBED_DIM = 384  # all-MiniLM-L6-v2 embedding dimension


class _DummySentenceTransformer:
    """Dummy encoder that returns zero vectors of fixed dimension."""
    def __init__(self, dim=_EMBED_DIM):
        self.dim = dim
        self.max_seq_length = 128  # dummy attribute

    def encode(self, sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True, **kwargs):
        # Return zero matrix of shape (len(sentences), self.dim)
        if isinstance(sentences, str):
            sentences = [sentences]
        return np.zeros((len(sentences), self.dim), dtype=np.float32)


def get_model(model_name: str = 'all-MiniLM-L6-v2'):
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL
    logger.info(f"Loading SBERT model: {model_name}")
    try:
        _MODEL = SentenceTransformer(model_name)
        _MODEL_NAME = model_name
        logger.info("SBERT model loaded")
    except Exception as e:
        logger.warning(f"Failed to load SBERT model '{model_name}': {e}. Using dummy zero embeddings.")
        _MODEL = _DummySentenceTransformer()
        _MODEL_NAME = model_name
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
