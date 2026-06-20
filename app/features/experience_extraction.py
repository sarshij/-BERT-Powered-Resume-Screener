"""
Extract years of experience and graduation year from resume text.
Replaces previously hardcoded values.
"""
import re
from datetime import datetime

CURRENT_YEAR = datetime.now().year

def extract_years_experience(resume_text: str) -> float:
    text = resume_text.lower()
    ranges = []

    for match in re.finditer(
        r'\b(\d{4})\s*[-–to]+\s*(present|current|now|\d{4})\b',
        text, re.IGNORECASE
    ):
        start = int(match.group(1))
        end_str = match.group(2).lower()
        if end_str in ('present', 'current', 'now'):
            end = CURRENT_YEAR
        else:
            end = int(end_str)
        if start < end <= CURRENT_YEAR + 5 and 1950 <= start <= CURRENT_YEAR:
            ranges.append((start, end))

    if ranges:
        ranges.sort()
        merged = [ranges[0]]
        for start, end in ranges[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        total_years = sum(end - start for start, end in merged)
    else:
        total_years = 0.0

    explicit = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?experience', text)
    if explicit:
        explicit_val = float(explicit.group(1))
        if total_years > 0:
            total_years = (total_years + explicit_val) / 2
        else:
            total_years = explicit_val

    return round(min(total_years, 50), 1)


def extract_graduation_year(resume_text: str) -> int:
    text = resume_text.lower()

    edu_section = re.split(
        r'\b(?:education|academic\s+background|academic\s+qualifications'
        r'|educational\s+qualification)\b',
        text, flags=re.IGNORECASE
    )

    if len(edu_section) > 1:
        search_text = edu_section[1]
    else:
        edu_keywords = [
            r'\b(?:b\.?[sa]\.?|bachelor|master|ph\.?d|doctorate|mba'
            r'|associate|diploma|b\.?tech|m\.?tech|b\.?sc|m\.?sc)\b',
            r'\b(?:university|college|institute|school)\b',
            r'\b(?:graduated|graduation|degree)\b',
        ]
        search_text = text
        for kw in edu_keywords:
            m = re.search(kw, text)
            if m:
                start = max(0, m.start() - 200)
                end = min(len(text), m.end() + 200)
                search_text = text[start:end]
                break

    years = sorted(set(
        int(m.group()) for m in re.finditer(r'\b(?:19|20)\d{2}\b', search_text)
        if 1950 <= int(m.group()) <= CURRENT_YEAR
    ))

    if not years:
        years = sorted(set(
            int(m.group()) for m in re.finditer(r'\b(?:19|20)\d{2}\b', text)
            if 1950 <= int(m.group()) <= CURRENT_YEAR
        ))

    if not years:
        return CURRENT_YEAR - 4

    candidates = [y for y in years if 1990 <= y <= CURRENT_YEAR]
    if not candidates:
        return CURRENT_YEAR - 4

    recent = sorted(candidates, reverse=True)[:3]
    experience_years = extract_years_experience(text)

    for y in recent:
        years_since = CURRENT_YEAR - y
        if 1 <= years_since <= 50:
            if experience_years > 0:
                implied_grad = CURRENT_YEAR - experience_years
                if abs(y - implied_grad) <= 5:
                    return y
            return y

    return recent[0]
