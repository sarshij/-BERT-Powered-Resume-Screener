"""
Skill Overlap Scoring
Extracts skills from text and computes Jaccard similarity with job description skills.

FIX 1 (Abbreviation normalization):
    Both resume and JD text are normalized through normalize_skills_text() before
    skill extraction. This ensures:
        - "js" in JD matches "JavaScript" in resume
        - "k8s" in JD matches "Kubernetes" in resume
        - "ml", "ai", "ts", "postgres", "mongo" etc. all resolve correctly
        - The canonical name is what gets stored, displayed, and compared
"""
import re
from ..utils.nlp import extract_skills_spacy
from ..utils.aliases import normalize_skills_text
from app.utils.taxonomy import SKILL_KEYWORDS


def extract_skills(text: str) -> set[str]:
    """
    Extract skills from text.

    Pipeline:
        1. Normalize abbreviations (js → JavaScript, ml → Machine Learning, etc.)
        2. Try spaCy EntityRuler-based extraction
        3. Fallback to regex matching against SKILL_KEYWORDS if spaCy yields nothing

    Args:
        text: Raw resume or job description text.

    Returns:
        Set of canonical skill names (as they appear in SKILL_KEYWORDS).
    """
    # Step 1: Normalize abbreviations FIRST so spaCy/regex sees canonical names
    normalized = normalize_skills_text(text)

    # Step 2: Try spaCy-based extraction on normalized text
    spacy_skills = extract_skills_spacy(normalized)
    if spacy_skills:
        return spacy_skills

    # Step 3: Fallback — regex match normalized text against taxonomy
    found = set()
    text_lower = normalized.lower()
    for skill in SKILL_KEYWORDS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def compute_skill_overlap(resume_text: str, job_description: str) -> float:
    """
    Compute Jaccard similarity between skills in resume and job description.
    Both texts are normalized before extraction so abbreviations match correctly.

    Returns:
        Float in [0.0, 1.0] — 1.0 means identical skill sets.
    """
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)
    if not resume_skills or not jd_skills:
        return 0.0
    intersection = resume_skills & jd_skills
    union = resume_skills | jd_skills
    return round(len(intersection) / len(union), 4)


def get_matched_skills(resume_text: str, job_description: str) -> dict:
    """
    Get detailed skill match breakdown between resume and job description.
    Both texts are normalized before extraction so abbreviations are resolved.

    Returns:
        Dict with keys: matched, missing, extra, match_count, missing_count, extra_count
        All skill names are canonical (e.g., "JavaScript" not "js").
    """
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)
    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)
    return {
        'matched': matched,
        'missing': missing,
        'extra': extra,
        'match_count': len(matched),
        'missing_count': len(missing),
        'extra_count': len(extra)
    }
