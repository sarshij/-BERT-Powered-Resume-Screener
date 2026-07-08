"""
Experience Relevance Scoring
Implements the proposed formula: 
    Experience relevance = min(1.0, candidate_years / required_years) * domain_relevance_factor
where:
    candidate_years: years of experience extracted from resume
    required_years: years of experience required by the job (extracted from target_job_title/job_description)
    domain_relevance_factor: proportion of relevant job categories mentioned in the resume
"""
import re 
from .experience_extraction import extract_years_experience 
from ..utils.taxonomy import JOB_CATEGORIES 
from ..utils.nlp import get_nlp 

def _extract_required_years(text: str) -> float: 
    """ 
    Extract a single number representing required years of experience from a string. 
    Returns the first reasonable integer found (0-50), or 0.0 if none. 
    """ 
    if not text: 
        return 0.0 
    # Find all integers in the string 
    numbers = re.findall(r'\b(\d+)\b', text) 
    for num in numbers: 
        n = int(num) 
        if 0 <= n <= 50: 
            return float(n) 
    return 0.0 

def score_experience_relevance(resume_text: str, target_job_title: str | None = None) -> float: 
    # 1. Get candidate years from resume (using the existing extraction function which tries spaCy first) 
    candidate_years = extract_years_experience(resume_text) 
    # 2. Get required years from target_job_title (or job_description) 
    required_years = _extract_required_years(target_job_title or "") 

    # 3. Compute ratio: min(1.0, candidate_years / required_years) if required_years > 0 else 0.0 
    if required_years > 0: 
        ratio = min(1.0, candidate_years / required_years) 
    else: 
        ratio = 0.0 

    # 4. Compute domain_relevance_factor using spaCy noun chunks and entities 
    # Determine relevant categories based on target_job_title (same as before) 
    relevant = set() 
    if target_job_title: 
        targ = (target_job_title or "").lower() 
        targ_words = set(targ.split()) 
        for category, roles in JOB_CATEGORIES.items(): 
            for role in roles: 
                role_words = set(role.lower().split()) 
                if len(role_words & targ_words) > 0: 
                    relevant.add(category) 
                    break 

    # If no relevant categories, factor = 0.5 
    if not relevant: 
        factor = 0.5 
    else: 
        # Process resume with spaCy to get noun chunks and entities 
        nlp = get_nlp() 
        mentioned_texts = set() 
        if nlp is not None and resume_text.strip(): 
            doc = nlp(resume_text) 
            for chunk in doc.noun_chunks: 
                mentioned_texts.add(chunk.text.lower()) 
            for ent in doc.ents: 
                mentioned_texts.add(ent.text.lower()) 
        # Count how many relevant categories have at least one role mentioned 
        mentioned_count = 0 
        for category in relevant: 
            found = False 
            for role in JOB_CATEGORIES[category]: 
                role_lower = role.lower() 
                for m in mentioned_texts: 
                    if role_lower in m: 
                        found = True 
                        break 
                if found: 
                    break 
            if found: 
                mentioned_count += 1 
        factor = mentioned_count / len(relevant) 

    # 5. Final score 
    score = ratio * factor 
    return round(min(1.0, score), 4) 
