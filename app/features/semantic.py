"""
Semantic Similarity Computation
Computes cosine similarity between resume and job description BERT embeddings.

FIX 5 (Short JD blending) + FIX 1 (Alias normalization):
    - Both texts are alias-normalized before embedding so "js" → "JavaScript"
      which gives SBERT much richer context.
    - For very short JDs (<15 words after normalization), also blends in
      token-level keyword overlap to compensate for poor embedding quality.
"""
import asyncio
import re
from app.models.embedder import embed_texts, embed_text_async, cosine_similarity
from app.utils.aliases import normalize_skills_text


# Minimum JD word count (after normalization) to trust pure SBERT embedding
_MIN_JD_WORDS_FOR_PURE_EMBEDDING = 15


def _token_overlap_score(resume_text: str, job_description: str) -> float:
    """
    Simple token-level overlap score between resume and JD.
    Used to supplement SBERT when JD is too short for good embeddings.
    """
    stopwords = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
        'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'with',
        'this', 'that', 'from', 'they', 'been', 'will', 'would', 'could',
        'should', 'their', 'about', 'which', 'when', 'make', 'like',
        'than', 'each', 'other', 'some', 'them', 'then', 'into', 'also',
        'just', 'more', 'most', 'very', 'what', 'where', 'who', 'how',
    }

    resume_words = set(
        w for w in re.findall(r'\b[a-z][a-z0-9+#.]+\b', resume_text.lower())
        if w not in stopwords and len(w) >= 2
    )
    jd_words = set(
        w for w in re.findall(r'\b[a-z][a-z0-9+#.]+\b', job_description.lower())
        if w not in stopwords and len(w) >= 2
    )

    if not jd_words:
        return 0.0

    overlap = jd_words & resume_words
    recall = len(overlap) / len(jd_words)
    return round(min(1.0, recall), 4)


# Keeping synchronous version for backwards compatibility with test suite
def compute_semantic_similarity(resume_text: str, job_description: str) -> float:
    if not job_description or not job_description.strip():
        return 0.0
    norm_resume = normalize_skills_text(resume_text)
    norm_jd = normalize_skills_text(job_description)
    resume_emb = embed_texts([norm_resume])
    jd_emb = embed_texts([norm_jd])
    if resume_emb.size == 0 or jd_emb.size == 0:
        return 0.0
    sbert_sim = cosine_similarity(resume_emb[0], jd_emb[0])
    jd_word_count = len(norm_jd.split())
    if jd_word_count >= _MIN_JD_WORDS_FOR_PURE_EMBEDDING:
        return round(sbert_sim, 4)
    token_sim = _token_overlap_score(norm_resume, norm_jd)
    sbert_weight = min(0.85, 0.3 + jd_word_count * 0.04)
    token_weight = 1.0 - sbert_weight
    blended = sbert_weight * sbert_sim + token_weight * token_sim
    return round(min(1.0, blended), 4)


async def compute_semantic_similarity_async(resume_text: str, job_description: str) -> float:
    """
    Async version of semantic similarity using ThreadPoolExecutor and LRU caching (Fix 9 & 14).
    """
    if not job_description or not job_description.strip():
        return 0.0

    # FIX 1: Normalize abbreviations before embedding
    norm_resume = normalize_skills_text(resume_text)
    norm_jd = normalize_skills_text(job_description)

    # Embed asynchronously in parallel
    resume_task = asyncio.create_task(embed_text_async(norm_resume))
    jd_task = asyncio.create_task(embed_text_async(norm_jd))
    
    resume_emb, jd_emb = await asyncio.gather(resume_task, jd_task)

    if resume_emb.size == 0 or jd_emb.size == 0:
        return 0.0

    sbert_sim = cosine_similarity(resume_emb, jd_emb)

    # FIX 5: For very short JDs, blend SBERT with token overlap
    jd_word_count = len(norm_jd.split())
    if jd_word_count >= _MIN_JD_WORDS_FOR_PURE_EMBEDDING:
        return round(sbert_sim, 4)

    # Short JD path: weighted blend
    token_sim = _token_overlap_score(norm_resume, norm_jd)
    sbert_weight = min(0.85, 0.3 + jd_word_count * 0.04)
    token_weight = 1.0 - sbert_weight

    blended = sbert_weight * sbert_sim + token_weight * token_sim
    return round(min(1.0, blended), 4)
