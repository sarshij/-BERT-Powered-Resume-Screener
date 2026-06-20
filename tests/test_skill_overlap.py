from app.features.skill_overlap import extract_skills, compute_skill_overlap, get_matched_skills


class TestExtractSkills:

    def test_detects_known_skills(self, sample_resume):
        skills = extract_skills(sample_resume)
        assert 'Python' in skills
        assert 'Java' in skills
        assert 'Docker' in skills
        assert 'Kubernetes' in skills
        assert 'AWS' in skills
        assert 'CI/CD' in skills
        assert 'Agile' in skills

    def test_no_skills_in_empty_text(self):
        assert extract_skills("") == set()

    def test_no_skills_in_generic_text(self, generic_resume):
        skills = extract_skills(generic_resume)
        assert 'Python' not in skills


class TestComputeSkillOverlap:

    def test_partial_overlap(self, sample_resume, sample_jd):
        score = compute_skill_overlap(sample_resume, sample_jd)
        assert 0 < score < 1.0

    def test_no_overlap(self):
        score = compute_skill_overlap("I like cooking", "software engineer python")
        assert score == 0.0

    def test_identical_skills(self):
        text = "Python Java AWS"
        score = compute_skill_overlap(text, text)
        assert score == 1.0

    def test_empty_text_returns_zero(self):
        assert compute_skill_overlap("", "python") == 0.0


class TestGetMatchedSkills:

    def test_returns_all_keys(self, sample_resume, sample_jd):
        result = get_matched_skills(sample_resume, sample_jd)
        assert set(result.keys()) == {'matched', 'missing', 'extra', 'match_count', 'missing_count', 'extra_count'}

    def test_matched_skills_count(self, sample_resume, sample_jd):
        result = get_matched_skills(sample_resume, sample_jd)
        assert result['match_count'] >= 3
        assert 'Python' in result['matched']
        assert 'AWS' in result['matched']
