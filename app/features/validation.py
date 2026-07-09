"""
Validation Feature Extraction
Detects keyword stuffing, generic phrases, gaps, inconsistencies.
17 features total: 12 original + 5 new (years_experience, num_certifications,
num_skills, education_level_encoded, has_previous_job).

BUG FIXES Applied:
  - BUG 8:  compute_keyword_stuffing_score now filters out stopwords before
            computing the stuffing ratio, preventing inflation from common words.
  - BUG 9:  has_previous_job now detects multiple date-range job blocks more
            robustly (doesn't depend on newline right after date range).
  - BUG 10: detect_overlapping_jobs now checks for actual date range overlaps
            instead of just counting total date ranges.
  - BUG 11: compute_skill_density uses consistent normalization.
"""
import re
from app.utils.taxonomy import GENERIC_PHRASES
from datetime import datetime

# ── Common English stopwords to filter from keyword stuffing ──
_STOPWORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'not',
    'no', 'nor', 'so', 'if', 'than', 'that', 'this', 'these', 'those',
    'it', 'its', 'he', 'she', 'we', 'they', 'i', 'me', 'my', 'your',
    'our', 'their', 'you', 'him', 'her', 'us', 'them', 'who', 'whom',
    'which', 'what', 'where', 'when', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'any',
    'only', 'own', 'same', 'also', 'about', 'up', 'out', 'into',
    'over', 'after', 'before', 'between', 'under', 'during', 'through',
    'above', 'below', 'then', 'once', 'here', 'there', 'just', 'very',
    'too', 'well', 'back', 'now', 'still', 'even', 'new', 'first',
    'last', 'long', 'great', 'little', 'right', 'old', 'big', 'high',
    'small', 'large', 'next', 'early', 'young', 'important', 'public',
    'able', 'etc', 'per', 'via', 'one', 'two', 'three', 'four', 'five',
})


def compute_keyword_stuffing_score(resume_text: str, job_description: str) -> float:
    """
    Compute how much the resume is 'stuffed' with JD keywords.
    BUG 8 FIX: Filters out stopwords before computing ratio to prevent
    common words from inflating the score artificially.
    """
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    # Extract meaningful words (exclude stopwords and very short words)
    jd_words = set(
        w for w in re.findall(r'\b[a-z]{3,}\b', jd_lower)
        if w not in _STOPWORDS
    )
    resume_words = [
        w for w in re.findall(r'\b[a-z]{3,}\b', resume_lower)
        if w not in _STOPWORDS
    ]

    if not resume_words or not jd_words:
        return 0.0

    # Count how many resume words match JD keywords
    jd_word_count = sum(1 for w in resume_words if w in jd_words)
    total_words = len(resume_words)
    ratio = jd_word_count / total_words if total_words > 0 else 0

    # Also check for excessive repetition of specific JD keywords
    word_freq = {}
    for w in resume_words:
        if w in jd_words:
            word_freq[w] = word_freq.get(w, 0) + 1
    
    # High repetition of JD words is a strong stuffing signal
    max_repeat = max(word_freq.values()) if word_freq else 0
    repeat_penalty = min(0.3, max_repeat * 0.02) if max_repeat > 10 else 0

    return round(min(1.0, ratio * 2.0 + repeat_penalty), 4)


def compute_generic_phrase_score(resume_text: str) -> float:
    """
    Score how many generic/cliché phrases the resume contains.
    High scores indicate possible fake or low-quality resumes.
    FIX 13: Uses regex word boundaries to prevent substring false positives
    (e.g., 'motivate' won't match 'motivated').
    """
    resume_lower = resume_text.lower()
    matches = 0
    for phrase in GENERIC_PHRASES:
        pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
        if re.search(pattern, resume_lower):
            matches += 1
    total_words = len(resume_text.split())
    if total_words == 0:
        return 0.0
    return round(min(1.0, matches / max(total_words * 0.01, 1)), 4)


def detect_gaps(resume_text: str) -> dict:
    """
    Detect unexplained employment gaps in the resume.
    Returns gap count, details, and total gap years.
    """
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
    """
    Compute skill density — ratio of skills to experience or resume length.
    BUG 11 FIX: Consistent normalization scale regardless of whether
    years_experience is available or not.
    """
    from app.features.skill_overlap import extract_skills
    skills = extract_skills(resume_text)
    skill_count = len(skills)

    if years_experience and years_experience > 0:
        # Skills per year of experience
        density = skill_count / years_experience
    else:
        # For fresh grads or when years unknown, use word-count-based density
        # Normalize to similar scale as years-based density
        total_words = len(resume_text.split())
        if total_words > 0:
            # Typical resume is 300-800 words, typical experience is 2-10 years
            # So normalize word count to an equivalent "years" scale
            equivalent_years = max(1, total_words / 150)
            density = skill_count / equivalent_years
        else:
            density = 0.0

    return round(min(density, 20), 4)


def count_achievements(resume_text: str) -> int:
    """
    Count quantifiable achievements and action verbs in the resume.
    Higher counts indicate more concrete, evidence-based experience.
    """
    patterns = [
        r'\b\d+%\b', r'\b\d+x\b', r'\$\s*\d+[kKmMbB]?\b',
        r'increased\b', r'reduced\b', r'improved\b', r'generated\b',
        r'led\b', r'managed\b', r'created\b', r'developed\b',
        r'implemented\b', r'achieved\b', r'delivered\b',
        r'optimized\b', r'automated\b', r'launched\b', r'designed\b',
        r'built\b', r'deployed\b', r'migrated\b', r'refactored\b',
        r'scaled\b', r'streamlined\b', r'integrated\b',
    ]
    count = 0
    text_lower = resume_text.lower()
    for pattern in patterns:
        count += len(re.findall(pattern, text_lower))
    return min(count, 50)


def compute_experience_graduation_gap(years_experience: float, graduation_year: float) -> float:
    """
    Compute the gap between years since graduation and claimed experience.
    Large gaps may indicate fabricated experience or education dates.
    """
    if graduation_year <= 0:
        return 0.0
    current_year = datetime.now().year
    years_since_grad = current_year - graduation_year
    gap = years_since_grad - years_experience
    return round(gap, 2)


def compute_promotion_speed(resume_text: str) -> float:
    """
    Estimate how quickly the candidate progressed through job titles.
    Unusually fast promotions may indicate exaggeration.
    """
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
    """
    Detect potentially overlapping job date ranges.
    BUG 10 FIX: Now actually checks if date ranges overlap with each other,
    instead of just counting total date ranges and using arbitrary threshold.
    """
    current_year = datetime.now().year

    # Extract date ranges like "2019 - 2022", "2020 – present"
    range_pattern = r'((?:19|20)\d{2})\s*[-–—]\s*(present|current|(?:19|20)\d{2})'
    matches = re.findall(range_pattern, resume_text, re.IGNORECASE)

    if len(matches) < 2:
        return 0

    # Parse date ranges into (start, end) tuples
    ranges = []
    for start_str, end_str in matches:
        start = int(start_str)
        end_lower = end_str.lower()
        if end_lower in ('present', 'current'):
            end = current_year
        else:
            end = int(end_str)
        if start <= end and 1950 <= start <= current_year:
            ranges.append((start, end))

    # Sort by start year and check for overlaps
    ranges.sort()
    overlap_count = 0
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            # Two ranges overlap if one starts before the other ends
            if ranges[j][0] < ranges[i][1]:
                overlap_count += 1

    return overlap_count


def count_certifications(resume_text: str) -> int:
    """
    Count professional certifications mentioned in the resume.
    """
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
    """
    Extract the highest education level from the resume.
    Returns: 0=Diploma/HighSchool, 1=Bachelor, 2=Master, 3=PhD
    """
    text_lower = resume_text.lower()
    if re.search(r'\bphd\b|\bdoctorate\b|\bdoctor of\b', text_lower):
        return 3
    if re.search(r"\bmaster[''\\u2019]?s\b|\bma\b|\bms\b|\bm\.?\s*sc\.?\b|\bmba\b", text_lower):
        return 2
    if re.search(r"\bbachelor[''\\u2019]?s\b|\bba\b|\bbs\b|\bb\.?\s*sc\.?\b|\bb\.?\s*tech\b|\bb\.?\s*eng\b|\bbca\b|\bbba\b|\bb\.?\s*com\b|\bb\.?\s*ed\b", text_lower):
        return 1
    if re.search(r'\b(?:diploma|associate|high school|higher secondary|10\+2|slc|see)\b', text_lower):
        return 0
    return 1


def has_previous_job(resume_text: str) -> int:
    """
    Detect whether the resume mentions previous employment.
    BUG 9 FIX: Improved detection to not depend on newline character after
    date ranges (PDF extraction often strips newlines). Now uses multiple
    strategies: explicit phrases, date range blocks, and company/role patterns.
    """
    text_lower = resume_text.lower()

    # Strategy 1: Explicit past-tense job indicators
    job_indicators = [
        r'previously worked',
        r'previous.*role',
        r'former\s+\w+',
        r'(?:previously|formerly|before that|prior to).*(?:was|were|worked|served)',
        r'(?:worked|served|employed)\s+(?:at|for|with|in)',
        r'experience\s+at\b',
    ]
    for pat in job_indicators:
        if re.search(pat, text_lower):
            return 1

    # Strategy 2: Count distinct date-range blocks (more robust regex)
    # Match patterns like "2019 - 2022", "Jan 2020 – Present", etc.
    date_range_pattern = r'(?:19|20)\d{2}\s*[-–—]\s*(?:present|current|(?:19|20)\d{2})'
    date_ranges = re.findall(date_range_pattern, resume_text, re.IGNORECASE)
    if len(set(date_ranges)) >= 2:
        return 1

    # Strategy 3: Multiple job title keywords found in text
    title_patterns = [
        r'\b(?:software|senior|junior|lead|principal|staff)\s+(?:engineer|developer|analyst|designer|architect|consultant)\b',
        r'\b(?:intern|internship|trainee|apprentice)\b',
        r'\b(?:manager|director|coordinator|specialist|executive|officer)\b',
    ]
    title_count = 0
    for pat in title_patterns:
        if re.search(pat, text_lower):
            title_count += 1
    if title_count >= 2:
        return 1

    # Strategy 4: Check for multiple company name indicators
    company_indicators = re.findall(
        r'\b(?:pvt|ltd|inc|corp|llc|limited|private|company|technologies|solutions|services|group|enterprise)\b',
        text_lower
    )
    if len(company_indicators) >= 2:
        return 1

    return 0


def compute_skill_experience_alignment(resume_text: str, extracted_skills: list) -> float:
    """
    Check if claimed skills are supported by corresponding project work and employment history.
    It does this by checking if a skill appears in the same sentence as an action verb.
    """
    if not extracted_skills:
        return 0.0
        
    text_lower = resume_text.lower()
    # Action verbs typically found in experience/project sections
    action_verbs = ['developed', 'built', 'created', 'led', 'managed', 'implemented', 
                    'designed', 'optimized', 'integrated', 'achieved', 'delivered', 
                    'architected', 'spearheaded', 'orchestrated', 'engineered']
                    
    # Tokenize into sentences or bullet points
    sentences = re.split(r'[.!?\n•\t-]', text_lower)
    # Filter out empty sentences and trim whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    supported_skills = 0
    for skill in extracted_skills:
        skill_lower = skill.lower()
        # Look for the skill in a sentence that also contains an action verb
        for sentence in sentences:
            if re.search(r'\b' + re.escape(skill_lower) + r'\b', sentence):
                if any(re.search(r'\b' + verb + r'\b', sentence) for verb in action_verbs):
                    supported_skills += 1
                    break
                    
    return round(supported_skills / len(extracted_skills), 4)


def compute_all_validation_features(resume_text: str, job_description: str,
                                   **kwargs) -> dict:
    """
    Compute all 18 validation features used as input to the XGBoost classifier.
    Returns a dict with all feature names matching the training columns exactly.
    """
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
        'skill_experience_alignment': compute_skill_experience_alignment(resume_text, kwargs.get('extracted_skills', [])),
        'ai_plausibility_score': kwargs.get('ai_plausibility_score', 0.5),
    }
