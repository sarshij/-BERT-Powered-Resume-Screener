from datetime import datetime
from app.features.experience_extraction import extract_years_experience, extract_graduation_year

def test_extract_years_explicit():
    text = "I have 5 years of experience."
    assert extract_years_experience(text) == 5.0

def test_extract_years_range():
    text = "Worked from 2018 to 2020."
    assert abs(extract_years_experience(text) - 2.0) < 0.1

def test_extract_years_overlap():
    text = "Worked 2018-2020 and 2019-2021."
    # overlapping ranges: 2018-2021 => 3 years
    res = extract_years_experience(text)
    assert 2.5 <= res <= 3.5

def test_extract_years_none():
    text = "No experience mentioned."
    assert extract_years_experience(text) == 0.0

def test_graduation_year_basic():
    text = "Bachelor of Science, 2015"
    assert extract_graduation_year(text) == 2015

def test_graduation_year_multiple():
    text = "Born 1990, graduated 2012, PhD 2018"
    # Should pick most recent plausible graduation year (2018) given experience logic later
    # but our function will return 2018 as it's in candidates and not filtered by experience yet
    assert extract_graduation_year(text) == 2018

def test_graduation_year_fallback():
    text = "No date"
    expected = datetime.now().year - 4
    assert extract_graduation_year(text) == expected

def test_score_experience_relevance_zero_required_years():
    resume = "I have 5 years of experience."
    target_job_title = "Manager"  # no numbers -> required_years = 0
    from app.features.experience import score_experience_relevance
    score = score_experience_relevance(resume, target_job_title)
    assert score == 0.0

def test_score_experience_relevance_sufficient_years_full_domain():
    resume = "I have 10 years of experience. Worked as a Software Engineer and Data Scientist and Data Analyst."
    target_job_title = "Data Scientist requires 5 years of experience"
    from app.features.experience import score_experience_relevance
    score = score_experience_relevance(resume, target_job_title)
    # candidate_years = 10, required_years = 5 -> ratio = 1.0
    # relevant categories: Engineering, AI_ML, Data (3)
    # all three mentioned -> factor = 1.0
    # score = 1.0
    assert abs(score - 1.0) < 1e-4

def test_score_experience_relevance_insufficient_years_full_domain():
    resume = "I have 3 years of experience. Worked as a Software Engineer and Data Scientist and Data Analyst."
    target_job_title = "Data Scientist requires 5 years of experience"
    from app.features.experience import score_experience_relevance
    score = score_experience_relevance(resume, target_job_title)
    # candidate_years = 3, required_years = 5 -> ratio = 3/5 = 0.6
    # factor = 1.0
    # score = 0.6
    assert abs(score - 0.6) < 1e-4

def test_score_experience_relevance_sufficient_years_no_domain():
    resume = "I have 10 years of experience."
    target_job_title = "Data Scientist requires 5 years of experience"
    from app.features.experience import score_experience_relevance
    score = score_experience_relevance(resume, target_job_title)
    # ratio = 1.0
    # relevant categories are Engineering, AI_ML, Data (from target_job_title)
    # but none of the role names appear in resume -> factor = 0.0
    # score = 0.0
    assert abs(score - 0.0) < 1e-4

def test_score_experience_relevance_sufficient_years_partial_domain():
    resume = "I have 10 years of experience. Worked as a Software Engineer and Data Analyst."
    target_job_title = "Data Scientist requires 5 years of experience"
    from app.features.experience import score_experience_relevance
    score = score_experience_relevance(resume, target_job_title)
    # ratio = 1.0
    # relevant categories: Engineering, AI_ML, Data (3)
    # mentioned: Software Engineer (Engineering), Data Analyst (Data) -> 2 out of 3 -> factor = 2/3 ˜ 0.666...
    # score = 0.666...
    assert abs(score - 2/3) < 1e-4
