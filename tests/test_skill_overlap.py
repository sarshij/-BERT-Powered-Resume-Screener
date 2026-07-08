import pytest
from app.features.skill_overlap import extract_skills, compute_skill_overlap, get_matched_skills

def test_extract_skills_basic():
    text = "I have experience with Python and Java."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "Java" in skills
    # Ensure we don't get false positives
    assert "Cobol" not in skills

def test_extract_skills_empty():
    text = "No relevant skills here."
    assert extract_skills(text) == set()

def test_compute_skill_overlap():
    resume = "Expert in Python, Java, and SQL."
    job = "Looking for Python and JavaScript."
    # Expected: intersection = {Python}, union = {Python, Java, SQL, JavaScript} -> 1/4 = 0.25
    score = compute_skill_overlap(resume, job)
    assert abs(score - 0.25) < 0.001

def get_matched_skills_test():
    resume = "I know Python, Java, and C++."
    job = "Java and PythonScript required."
    result = get_matched_skills(resume, job)
    assert result["matched"] == ["Java"]
    assert result["missing"] == ["PythonScript"]
    assert result["extra"] == ["C++", "Python"]
