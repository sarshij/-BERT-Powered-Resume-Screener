"""
Skills taxonomy, job categories, and generic phrases for resume analysis.
Now loads from data/taxonomy.json and includes dynamic embedding fallback.
"""
import json
import os
import numpy as np
from app.logger import setup_logger

logger = setup_logger(__name__)

# Load from JSON
_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'taxonomy.json')
try:
    with open(_data_path, 'r') as f:
        _data = json.load(f)
        SKILL_KEYWORDS = _data.get('SKILL_KEYWORDS', [])
        JOB_CATEGORIES = _data.get('JOB_CATEGORIES', {})
        GENERIC_PHRASES = _data.get('GENERIC_PHRASES', [])
except Exception as e:
    logger.error(f"Failed to load taxonomy.json: {e}")
    SKILL_KEYWORDS = []
    JOB_CATEGORIES = {}
    GENERIC_PHRASES = []

# Dynamic Taxonomy Fallback
_seed_embeddings = None

def _init_seed_embeddings():
    global _seed_embeddings
    if _seed_embeddings is not None:
        return
    try:
        from app.models.embedder import embed_texts
        # Use key categories as seeds
        seed_phrases = ["Programming Language", "Frontend Web Development", "Backend Development", 
                        "Cloud Computing DevOps", "Machine Learning Artificial Intelligence",
                        "Data Engineering", "Database SQL NoSQL", "Software Testing", 
                        "Cybersecurity", "UI UX Design"]
        _seed_embeddings = embed_texts(seed_phrases)
    except Exception as e:
        logger.warning(f"Failed to initialize seed embeddings for dynamic taxonomy: {e}")
        _seed_embeddings = False

def is_dynamic_skill(noun_phrase: str) -> bool:
    """
    Checks if an unknown noun phrase is semantically close to known technical skills
    using SBERT embeddings.
    """
    global _seed_embeddings
    _init_seed_embeddings()
    if not _seed_embeddings:
        return False
        
    try:
        from app.models.embedder import embed_texts, cosine_similarity
        phrase_emb = embed_texts([noun_phrase])[0]
        # Compare to seeds
        max_sim = 0.0
        for seed_emb in _seed_embeddings:
            sim = cosine_similarity(phrase_emb, seed_emb)
            if sim > max_sim:
                max_sim = sim
                
        # If highly similar to our technical seed categories, consider it a skill
        if max_sim > 0.82:
            logger.debug(f"Dynamically classified '{noun_phrase}' as a skill (sim: {max_sim:.2f})")
            return True
    except Exception as e:
        pass
    return False
