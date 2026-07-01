"""
Validation Feature Extraction
Detects keyword stuffing, generic phrases, gaps, inconsistencies.
17 features total: 12 original + 5 new (years_experience, num_certifications,
num_skills, education_level_encoded, has_previous_job).
"""
import re
from app.utils.taxonomy import GENERIC_PHRASES

def compute_keyword_stuffing_score(resume_text: str, job_description: str) -> float:
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()
    jd_words = set(re.findall(r'\b[a-z]+\b', jd_lower))
    resume_words = re.findall(r'\b[a-z]+\b', resume_lower)
    if not resume_words or not jd_words:
        return 0.0
    word_freq = {}
    for w in resume_words:
        word_freq[w] = word_freq.get(w, 0) + 1
    jd_word_count = sum(1 for w in resume_words if w in jd_words)
    total_words = len(resume_words)
    ratio = jd_word_count / total_words if total_words > 0 else 0
    return round(min(1.0, ratio * 2.5), 4)

def compute_generic_phrase_score(resume_text: str) -> float:
    resume_lower = resume_text.lower()
    matches = 0
    for phrase in GENERIC_PHRASES:
        if phrase in resume_lower:
            matches += 1
    total_words = len(resume_text.split())
    if total_words == 0:
        return 0.0
    return round(min(1.0, matches / max(total_words * 0.01, 1)), 4)

def detect_gaps(resume_text: str) -> dict:
    date_pattern = r'(19|20)\d{2}'
    years = sorted(set(int(y) for y in re.findall(date_pattern, resume_text)))
    gaps = []
    if len(years) > 1:
        for i in range(1, len(years)):
            gap = years[i] - years[i - 1]
            if gap > 2:
                gaps.append({'from': years[i - 1], 'to': years[i], 'years': gap})
    return {'gap_count': len(gaps), 'gaps': gaps, 'total_gap_years': sum(g['years'] for g in gaps)}

def compute_skill_density(resume_text: str, years_experience: float = None) -> float:
    from app.features.skill_overlap import extract_skills
    skills = extract_skills(resume_text)
    skill_count = len(skills)
    if years_experience and years_experience > 0:
        density = skill_count / years_experience
    else:
        total_words = len(resume_text.split())
        density = skill_count / max(total_words * 0.001, 1)
    return round(min(density, 20), 4)

def count_achievements(resume_text: str) -> int:
    patterns = [
        r'\b\d+%\b', r'\b\d+x\b', r'\$\s*\d+[kKmMbB]?\b',
        r'increased\b', r'reduced\b', r'improved\b', r'generated\b',
        r'led\b', r'managed\b', r'created\b', r'developed\b',
        r'implemented\b', r'achieved\b', r'delivered\b'
    ]
    count = 0
    text_lower = resume_text.lower()
    for pattern in patterns:
        count += len(re.findall(pattern, text_lower))
    return min(count, 50)

def compute_experience_graduation_gap(years_experience: float, graduation_year: float, current_year: float = 2026) -> float:
    if graduation_year <= 0:
        return 0.0
    years_since_grad = current_year - graduation_year
    gap = years_since_grad - years_experience
    return round(gap, 2)

def compute_promotion_speed(resume_text: str) -> float:
    title_keywords = ['promoted', 'promotion', 'advanced to', 'elevated to',
                      'senior', 'lead', 'head', 'manager', 'director', 'chief']
    text_lower = resume_text.lower()
    titles = sum(1 for kw in title_keywords if kw in text_lower)
    years = re.findall(r'(19|20)\d{2}', resume_text)
    unique_years = len(set(years))
    if unique_years > 1 and titles > 0:
        return round(titles / unique_years, 4)
    return 0.0

def detect_overlapping_jobs(resume_text: str) -> int:
    date_ranges = re.findall(r'(19|20)\d{2}\s*[-–]\s*(?:present|current|(19|20)\d{2})', resume_text, re.IGNORECASE)
    return len(date_ranges) if len(date_ranges) > 2 else 0

def count_certifications(resume_text: str) -> int:
    cert_keywords = [
        r'certified', r'certification', r'certificate', r'license',
        r'\bAWS\b', r'\bAzure\b', r'\bGCP\b', r'\bCISSP\b', r'\bCISA\b',
        r'\bCISM\b', r'\bPMP\b', r'\bOSCP\b', r'\bCEH\b', r'\bCKA\b',
        r'\bCKAD\b', r'\bSCM\b', r'\bPSM\b', r'\bCSPO\b', r'\bCompTIA\b',
        r'\bISO\b', r'ITIL', r'TOGAF', r'COBIT', r'CPA', r'CFA',
        r'Google.*(?:Cloud|Professional|Data|ML)',
        r'Microsoft.*(?:Certified|Azure|DP|AZ|MS)',
        r'AWS.*(?:Certified|Solutions|Developer|DevOps|Machine Learning)',
        r'(?:Oracle|Java|MongoDB|Snowflake|Databricks).*(?:Certified|Associate|Developer|Professional)',
        r'HashiCorp|Terraform|Docker|Kubernetes',
        r'Scrum Master|SAFe|Lean Six Sigma|Six Sigma',
    ]
    text_lower = resume_text.lower()
    count = sum(1 for pat in cert_keywords if re.search(pat, text_lower, re.IGNORECASE))
    return min(count, 30)

def extract_education_level(resume_text: str) -> int:
    text_lower = resume_text.lower()
    if re.search(r'\bphd\b|\bdoctorate\b|\bdoctor of\b', text_lower):
        return 3
    if re.search(r"\bmaster[''\u2019]?s\b|\bma\b|\bms\b|\bm\.?\s*sc\.?\b|\bmba\b", text_lower):
        return 2
    if re.search(r"\bbachelor[''\u2019]?s\b|\bba\b|\bbs\b|\bb\.?\s*sc\.?\b|\bb\.?\s*tech\b|\bb\.?\s*eng\b", text_lower):
        return 1
    if re.search(r'\b(?:diploma|associate|high school|higher secondary|10\+2)\b', text_lower):
        return 0
    return 1

def has_previous_job(resume_text: str) -> int:
    text_lower = resume_text.lower()
    job_indicators = [
        r'previously worked as', r'previous.*role', r'former',
        r'(?:previously|formerly|before that|prior to).*(?:was|were|worked|served)',
    ]
    for pat in job_indicators:
        if re.search(pat, text_lower):
            return 1
    job_blocks = re.findall(
        r'((?:19|20)\d{2}\s*[-–]\s*(?:present|current|(?:19|20)\d{2})).*?\n',
        resume_text
    )
    if len(set(job_blocks)) >= 2:
        return 1
    return 0

def compute_all_validation_features(resume_text: str, job_description: str,
                                   **kwargs) -> dict:
    gaps = detect_gaps(resume_text)
    years_exp = kwargs.get('years_experience', 0)
    return {
        'semantic_similarity': kwargs.get('semantic_similarity', 0.0),
        'skill_overlap_score': kwargs.get('skill_overlap_score', 0.0),
        'experience_relevance_score': kwargs.get('experience_relevance_score', 0.0),
        'final_match_score': kwargs.get('final_match_score', 0.0),
        'overlapping_jobs': detect_overlapping_jobs(resume_text),
        'promotion_speed': compute_promotion_speed(resume_text),
        'experience_graduation_gap': compute_experience_graduation_gap(
            years_exp, kwargs.get('graduation_year', 0)),
        'skill_density': compute_skill_density(resume_text, years_exp),
        'achievement_count': count_achievements(resume_text),
        'generic_phrase_score': compute_generic_phrase_score(resume_text),
        'gap_years': gaps['total_gap_years'],
        'keyword_stuffing_score': compute_keyword_stuffing_score(resume_text, job_description),
        'years_experience': float(years_exp),
        'num_certifications': float(count_certifications(resume_text)),
        'num_skills': float(len(kwargs.get('extracted_skills', []))),
        'education_level_encoded': float(extract_education_level(resume_text)),
        'has_previous_job': float(has_previous_job(resume_text)),
    }
