'''
Simple spaCy-based NER helpers for resume analysis.
Falls back to rule-based methods if model not available.
'''
import os
import re
from datetime import datetime

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span

from app.logger import setup_logger

logger = setup_logger(__name__)

# Load spaCy model (small English) - disable unnecessary components for speed
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load('en_core_web_sm')
            logger.info('Loaded spaCy model en_core_web_sm')
        except OSError:
            logger.warning('spaCy model en_core_web_sm not found. Install via: python -m spacy download en_core_web_sm')
            _nlp = False
    return _nlp if _nlp is not False else None

# We'll add an EntityRuler for custom entities (SKILL, EDUCATION, JOB_TITLE)
# We'll create a function to get the nlp pipeline with the ruler added.
def get_nlp_with_ruler():
    nlp = get_nlp()
    if nlp is None:
        return None
    # Check if ruler already added
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        # Patterns will be added lazily to avoid slowing down load
        ruler._patterns_added = False
    return nlp

def _ensure_patterns(nlp):
    """Add patterns to the entity ruler if not already added."""
    ruler = nlp.get_pipe("entity_ruler")
    if getattr(ruler, '_patterns_added', False):
        return
    patterns = []
    # Skill patterns
    from app.utils.taxonomy import SKILL_KEYWORDS
    for skill in SKILL_KEYWORDS:
        patterns.append({"label": "SKILL", "pattern": skill.lower()})
    # Education patterns
    education_keywords = [
        "bachelor", "master", "phd", "doctorate", "associate", "diploma",
        "b.tech", "m.tech", "b.sc", "m.sc", "b.a", "m.a", "mba",
        "b.e", "m.e", "b.s", "m.s", "b.eng", "m.eng",
        "high school", "secondary school", "highschool", "secondaryschool",
        "college", "university", "institute", "school",
    ]
    for kw in education_keywords:
        patterns.append({"label": "EDUCATION", "pattern": kw})
    # Job title patterns
    from app.utils.taxonomy import JOB_CATEGORIES
    for category, titles in JOB_CATEGORIES.items():
        for title in titles:
            patterns.append({"label": "JOB_TITLE", "pattern": title.lower()})
    # Add patterns
    ruler.add_patterns(patterns)
    ruler._patterns_added = True

def extract_skills_spacy(text: str) -> set[str]:
    """
    Extract skills using spaCy entity recognition (SKILL entities).
    Returns a set of skill strings (as defined in SKILL_KEYWORDS).
    """
    nlp = get_nlp_with_ruler()
    if nlp is None:
        return set()
    _ensure_patterns(nlp)
    doc = nlp(text.lower())
    from app.utils.taxonomy import SKILL_KEYWORDS
    lower_to_original = {k.lower(): k for k in SKILL_KEYWORDS}
    skills = set()
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            key = ent.text.lower()
            if key in lower_to_original:
                skills.add(lower_to_original[key])
    return skills

def extract_education_spacy(text: str) -> set[str]:
    """
    Extract education entities (EDUCATION).
    Returns a set of education phrases (lowercased).
    """
    nlp = get_nlp_with_ruler()
    if nlp is None:
        return set()
    _ensure_patterns(nlp)
    doc = nlp(text.lower())
    edu = set()
    for ent in doc.ents:
        if ent.label_ == "EDUCATION":
            edu.add(ent.text)
    return edu

def extract_job_titles_spacy(text: str) -> set[str]:
    """
    Extract job title entities (JOB_TITLE).
    Returns a set of job title phrases (lowercased).
    """
    nlp = get_nlp_with_ruler()
    if nlp is None:
        return set()
    _ensure_patterns(nlp)
    doc = nlp(text.lower())
    titles = set()
    for ent in doc.ents:
        if ent.label_ == "JOB_TITLE":
            titles.add(ent.text)
    return titles

# Keep the existing date extraction functions (they use the base NER model's DATE entities)
def _is_range_separator(text: str) -> bool:
    return ('-' in text) or ('\u2013' in text) or ('\u2014' in text) or (' to ' in text.lower())

def extract_years_experience_spacy(text: str) -> float:
    """
    Extract years of experience using spaCy DATE entities and simple heuristics.
    Returns years as float.
    """
    nlp = get_nlp()
    if nlp is None:
        return 0.0
    doc = nlp(text)
    intervals = []  # list of (start_year, end_year)
    present_indicators = {'present', 'current', 'now'}
    current_year = datetime.now().year
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            txt = ent.text.strip()
            nums = re.findall(r'\b(?:19|20)\d{2}\b', txt)
            if len(nums) >= 2 and _is_range_separator(txt):
                start = int(nums[0])
                end = int(nums[1])
                if start < end:
                    intervals.append((start, end))
            elif len(nums) == 1:
                year = int(nums[0])
                lower_txt = txt.lower()
                if any(word in lower_txt for word in present_indicators):
                    if year <= current_year:
                        intervals.append((year, current_year))
                # else ignore single year (likely a date like graduation)
    if not intervals:
        return 0.0
    # Merge intervals
    intervals.sort(key=lambda x: x[0])
    merged = []
    if len(intervals) > 0:
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end:  # overlapping or adjacent
                if end > cur_end:
                    cur_end = end
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))
    total_years = 0
    for start, end in merged:
        total_years += (end - start)
    return min(total_years, 50.0)

def extract_graduation_year_spacy(text: str) -> int:
    """
    Extract likely graduation year from DATE entities.
    Returns year as int, or current year - 4 as fallback.
    """
    nlp = get_nlp()
    if nlp is None:
        return datetime.now().year - 4
    doc = nlp(text)
    years = []
    current_year = datetime.now().year
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            for y in re.findall(r'\b(?:19|20)\d{2}\b', ent.text):
                yi = int(y)
                if 1950 <= yi <= current_year:
                    years.append(yi)
    if years:
        # Prefer recent plausible graduation year
        return max(years)
    return current_year - 4
