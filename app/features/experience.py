"""
Experience Relevance Scoring
Parses job titles from resume and scores relevance to target role.
"""
from app.utils.taxonomy import JOB_CATEGORIES

def score_experience_relevance(resume_text: str, target_job_title: str | None = None) -> float:
    resume_lower = resume_text.lower()
    target_lower = target_job_title.lower() if target_job_title else ''
    target_words = set(target_lower.split())
    relevant_categories = set()
    for category, roles in JOB_CATEGORIES.items():
        for role in roles:
            role_words = set(role.lower().split())
            overlap = len(role_words & target_words)
            total = len(role_words | target_words)
            if total > 0 and overlap / total > 0.25:
                relevant_categories.add(category)
                break
    if relevant_categories:
        mention_score = sum(1 for cat in relevant_categories if cat.lower() in resume_lower)
        return round(min(1.0, mention_score / max(len(relevant_categories), 1) + 0.3), 4)
    job_role_match = sum(1 for word in target_words if word in resume_lower)
    if job_role_match > 0:
        return round(min(1.0, job_role_match / max(len(target_words) * 0.5, 1)), 4)
    experience_keywords = ['year of experience', 'years of experience', 'yr experience',
                           'worked as', 'worked at', 'employment history', 'professional experience']
    mention_count = sum(1 for kw in experience_keywords if kw in resume_lower)
    return round(min(1.0, mention_count / 3), 4)
