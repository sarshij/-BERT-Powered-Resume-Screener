"""
Semantic Similarity Computation
Computes cosine similarity between resume and job description BERT embeddings.
"""
from app.models.embedder import embed_texts, cosine_similarity

def compute_semantic_similarity(resume_text: str, job_description: str) -> float:
    resume_emb = embed_texts([resume_text])
    jd_emb = embed_texts([job_description])
    if resume_emb.size == 0 or jd_emb.size == 0:
        return 0.0
    return round(cosine_similarity(resume_emb[0], jd_emb[0]), 4)
