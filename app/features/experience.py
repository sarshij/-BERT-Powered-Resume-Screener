"""
Experience Relevance Scoring
Implements the proposed formula: 
    Experience relevance = min(1.0, candidate_years / required_years) * domain_relevance_factor
where:
    candidate_years: years of experience extracted from resume
    required_years: years of experience required by the job (extracted from target_job_title/job_description)
    domain_relevance_factor: proportion of relevant job categories mentioned in the resume

BUG FIXES Applied:
  - BUG 4: When required_years is 0 (no explicit requirement), use a sensible
           baseline ratio instead of returning 0. This was causing experience
           relevance to always show 0% for typical job titles.
  - BUG 7: _extract_required_years now only matches numbers near "year/experience"
           keywords, preventing false positives like "React 18" → 18 years.
"""
import re 
from .experience_extraction import extract_years_experience 
from ..utils.taxonomy import JOB_CATEGORIES 
from ..utils.nlp import get_nlp 


def _extract_required_years(text: str) -> float: 
    """ 
    Extract required years of experience from a job title or description.
    Only matches numbers that appear near 'year/years/yrs/experience' keywords
    to avoid false positives like "React 18" being interpreted as 18 years.
    Returns the first reasonable match (1-50), or 0.0 if none found.
    """ 
    if not text: 
        return 0.0 

    # BUG 7 FIX: Only match numbers explicitly tied to experience/years keywords
    # Pattern 1: "5+ years of experience", "3 years experience", "2 yrs exp"
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp|work)',
        r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:in|of|working)',
        r'(?:minimum|min|at\s+least|over)\s+(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:relevant|professional|hands-on|practical)',
        r'experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            n = int(match.group(1))
            if 1 <= n <= 50:
                return float(n)
    
    return 0.0 


def score_experience_relevance(resume_text: str, target_job_title: str | None = None) -> float: 
    """
    Compute experience relevance score combining years ratio and domain match.
    
    BUG 4 FIX: When no explicit years requirement is found in the JD/title,
    we now use a sensible baseline ratio based on the candidate's actual
    experience, instead of returning 0. A candidate with real experience
    should get credit for it even when the JD doesn't specify "X years required".
    """
    # 1. Get candidate years from resume
    candidate_years = extract_years_experience(resume_text) 

    # 2. Get required years from target_job_title (or job_description) 
    required_years = _extract_required_years(target_job_title or "") 

    # 3. Compute experience ratio
    if required_years > 0: 
        # Explicit requirement found — use the proposed formula
        ratio = min(1.0, candidate_years / required_years) 
    else:
        # BUG 4 FIX: No explicit years requirement in JD.
        # Use a gentle baseline — any experience is better than none.
        # Scale: 0 years → 0.3 (fresh grad baseline), 2+ years → 0.6+, 5+ years → ~0.85
        if candidate_years >= 5:
            ratio = min(1.0, 0.7 + candidate_years * 0.03)
        elif candidate_years >= 1:
            ratio = min(0.85, 0.4 + candidate_years * 0.1)
        elif candidate_years > 0:
            ratio = 0.35
        else:
            # No experience detected at all — give a small baseline for fresh grads
            ratio = 0.2

    # 4. Compute domain_relevance_factor using spaCy noun chunks and entities
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

    # If no relevant categories matched, use a moderate factor
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

        # Also check raw text for role keywords (more robust than just NER)
        resume_lower = resume_text.lower()
        
        # Count how many relevant categories have at least one role mentioned 
        mentioned_count = 0 
        for category in relevant: 
            found = False 
            for role in JOB_CATEGORIES[category]: 
                role_lower = role.lower() 
                # Check both NER extractions and direct text search
                if role_lower in resume_lower:
                    found = True
                    break
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
