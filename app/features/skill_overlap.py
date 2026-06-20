"""
Skill Overlap Scoring
Extracts skills from text and computes Jaccard similarity with job description skills.
"""
import re
from app.utils.taxonomy import SKILL_KEYWORDS

def extract_skills(text: str) -> set[str]:
    found = set()
    text_lower = text.lower()
    for skill in SKILL_KEYWORDS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)
    return found

def compute_skill_overlap(resume_text: str, job_description: str) -> float:
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)
    if not resume_skills or not jd_skills:
        return 0.0
    intersection = resume_skills & jd_skills
    union = resume_skills | jd_skills
    return round(len(intersection) / len(union), 4)

def get_matched_skills(resume_text: str, job_description: str) -> dict:
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
