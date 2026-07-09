from app.features.validation import (
    compute_keyword_stuffing_score,
    compute_generic_phrase_score,
    detect_gaps,
    compute_skill_density,
    count_achievements,
    compute_experience_graduation_gap,
    compute_promotion_speed,
    detect_overlapping_jobs,
    compute_all_validation_features,
    count_certifications,
    extract_education_level,
    has_previous_job,
)


class TestKeywordStuffing:

    def test_high_repetition_of_jd_words(self):
        resume = "python python python python python java java java java aws aws aws"
        jd = "Looking for Python, Java and AWS developer"
        score = compute_keyword_stuffing_score(resume, jd)
        assert score > 0.5

    def test_no_matching_words(self):
        resume = "managing financial portfolios and risk assessment"
        jd = "software engineer with python and cloud"
        score = compute_keyword_stuffing_score(resume, jd)
        assert 0 <= score <= 1.0

    def test_empty_jd_returns_zero(self):
        score = compute_keyword_stuffing_score("python developer", "")
        assert score == 0.0

    def test_empty_resume_returns_zero(self):
        score = compute_keyword_stuffing_score("", "python developer")
        assert score == 0.0

    def test_returns_between_zero_and_one(self):
        score = compute_keyword_stuffing_score("python java aws docker kubernetes", "python java aws")
        assert 0.0 <= score <= 1.0


class TestGenericPhrase:

    def test_detects_common_buzzwords(self, generic_resume):
        score = compute_generic_phrase_score(generic_resume)
        assert score > 0.5

    def test_clean_resume_low_score(self, sample_resume):
        score = compute_generic_phrase_score(sample_resume)
        assert score <= 1.0

    def test_empty_text_returns_zero(self):
        score = compute_generic_phrase_score("")
        assert score == 0.0


class TestDetectGaps:

    def test_no_gaps_when_continuous(self, sample_resume):
        gaps = detect_gaps(sample_resume)
        assert gaps['gap_count'] == 0

    def test_detects_large_gap(self):
        text = "Job A 2010 - 2012\n Job B 2018 - 2020"
        gaps = detect_gaps(text)
        assert gaps['gap_count'] >= 0
        assert gaps['total_gap_years'] >= 0

    def test_empty_text_no_gaps(self):
        gaps = detect_gaps("")
        assert gaps['gap_count'] == 0
        assert gaps['total_gap_years'] == 0


class TestSkillDensity:

    def test_returns_positive_value(self, sample_resume):
        density = compute_skill_density(sample_resume, years_experience=5)
        assert density > 0

    def test_high_density_indicates_stuffing(self):
        text = "Python Java AWS Docker Kubernetes " * 20
        density = compute_skill_density(text, years_experience=1)
        assert density >= 5

    def test_empty_text_returns_zero(self):
        density = compute_skill_density("", years_experience=5)
        assert density == 0.0

    def test_zero_experience_uses_word_count_fallback(self):
        text = "Python Java AWS"
        density = compute_skill_density(text, years_experience=0)
        assert density >= 0


class TestCountAchievements:

    def test_detects_percentages(self, sample_resume):
        count = count_achievements(sample_resume)
        assert count >= 3

    def test_no_achievements_in_generic_text(self, generic_resume):
        count = count_achievements(generic_resume)
        assert count == 0

    def test_empty_text(self):
        assert count_achievements("") == 0


class TestGraduationGap:

    def test_reasonable_gap(self):
        gap = compute_experience_graduation_gap(5, 2018)
        assert gap == 3.0

    def test_inflated_experience_creates_large_gap(self):
        gap = compute_experience_graduation_gap(15, 2018)
        assert gap < 0

    def test_zero_graduation_year(self):
        gap = compute_experience_graduation_gap(5, 0)
        assert gap == 0.0


class TestPromotionSpeed:

    def test_has_promotions(self, sample_resume):
        speed = compute_promotion_speed(sample_resume)
        assert speed >= 0

    def test_no_promotion_keywords(self, generic_resume):
        speed = compute_promotion_speed(generic_resume)
        assert speed == 0.0

    def test_empty_text(self):
        assert compute_promotion_speed("") == 0.0


class TestOverlappingJobs:

    def test_no_overlap_in_normal_resume(self, sample_resume):
        assert detect_overlapping_jobs(sample_resume) >= 0

    def test_empty_text(self):
        assert detect_overlapping_jobs("") == 0


class TestCountCertifications:

    def test_detects_certification_keywords(self):
        text = "Certified AWS Solutions Architect, CISSP, PMP certification"
        assert count_certifications(text) >= 3

    def test_no_certifications_returns_zero(self):
        text = "python developer with experience"
        assert count_certifications(text) == 0

    def test_empty_text_returns_zero(self):
        assert count_certifications("") == 0


class TestExtractEducationLevel:

    def test_phd_detected(self):
        assert extract_education_level("PhD in Computer Science") == 3
        assert extract_education_level("Doctorate in Physics") == 3

    def test_masters_detected(self):
        assert extract_education_level("Master's in Data Science") == 2
        assert extract_education_level("MBA from Harvard") == 2
        assert extract_education_level("MS in Robotics") == 2

    def test_bachelors_detected(self):
        assert extract_education_level("Bachelor's in Engineering") == 1
        assert extract_education_level("BS in Computer Science") == 1
        assert extract_education_level("B.Tech in IT") == 1

    def test_empty_text_defaults_to_bachelor(self):
        assert extract_education_level("") == 1


class TestHasPreviousJob:

    def test_detects_previous_role(self):
        text = "Previously worked as a Software Engineer before becoming a Senior Engineer"
        assert has_previous_job(text) == 1

    def test_no_previous_job(self):
        text = "First job as Software Engineer 2020 - Present"
        assert has_previous_job(text) == 0

    def test_empty_text_returns_zero(self):
        assert has_previous_job("") == 0


class TestComputeAllFeatures:

    def test_returns_all_19_keys(self, sample_resume, sample_jd):
        features = compute_all_validation_features(
            sample_resume, sample_jd,
            semantic_similarity=0.5,
            skill_overlap_score=0.3,
            experience_relevance_score=0.7,
            final_match_score=0.5,
            years_experience=5,
            graduation_year=2018,
        )
        expected_keys = {
            'semantic_similarity', 'skill_overlap_score', 'experience_relevance_score',
            'final_match_score', 'overlapping_jobs', 'promotion_speed',
            'experience_graduation_gap', 'skill_density', 'achievement_count',
            'generic_phrase_score', 'gap_years', 'keyword_stuffing_score',
            'years_experience', 'num_certifications', 'num_skills',
            'education_level_encoded', 'has_previous_job',
            'skill_experience_alignment', 'ai_plausibility_score'
        }
        assert set(features.keys()) == expected_keys

    def test_defaults_when_not_provided(self):
        features = compute_all_validation_features("", "")
        assert features['skill_density'] == 0.0
