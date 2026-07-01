import pytest
import numpy as np
from app.models.classifier import load_model, predict, get_feature_importance, get_model_info


class TestLoadModel:

    def test_model_loads_successfully(self):
        model, cols = load_model()
        assert model is not None
        assert len(cols) == 17

    def test_feature_names_are_correct(self):
        _, cols = load_model()
        expected = [
            'semantic_similarity', 'skill_overlap_score', 'experience_relevance_score',
            'final_match_score', 'overlapping_jobs', 'promotion_speed',
            'experience_graduation_gap', 'skill_density', 'achievement_count',
            'generic_phrase_score', 'gap_years', 'keyword_stuffing_score',
            'years_experience', 'num_certifications', 'num_skills',
            'education_level_encoded', 'has_previous_job'
        ]
        assert cols == expected


class TestPredict:

    def test_single_dict_returns_list(self, sample_resume, sample_jd):
        from app.features.validation import compute_all_validation_features
        features = compute_all_validation_features(
            sample_resume, sample_jd,
            semantic_similarity=0.5, skill_overlap_score=0.3,
            experience_relevance_score=0.7, final_match_score=0.5,
            years_experience=5, graduation_year=2018,
        )
        result = predict(features)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_result_contains_expected_keys(self, sample_resume, sample_jd):
        from app.features.validation import compute_all_validation_features
        features = compute_all_validation_features(
            sample_resume, sample_jd,
            semantic_similarity=0.5, skill_overlap_score=0.3,
            experience_relevance_score=0.7, final_match_score=0.5,
            years_experience=5, graduation_year=2018,
        )
        result = predict(features)[0]
        assert 'classification' in result
        assert 'confidence' in result
        assert 'prob_Authentic' in result
        assert 'prob_Suspicious' in result
        assert 'prob_Potentially Fake' in result

    def test_classification_is_valid(self, sample_resume, sample_jd):
        from app.features.validation import compute_all_validation_features
        features = compute_all_validation_features(
            sample_resume, sample_jd,
            semantic_similarity=0.5, skill_overlap_score=0.3,
            experience_relevance_score=0.7, final_match_score=0.5,
            years_experience=5, graduation_year=2018,
        )
        result = predict(features)[0]
        assert result['classification'] in ['Authentic', 'Suspicious', 'Potentially Fake']

    def test_confidence_between_zero_and_one(self, sample_resume, sample_jd):
        from app.features.validation import compute_all_validation_features
        features = compute_all_validation_features(
            sample_resume, sample_jd,
            semantic_similarity=0.5, skill_overlap_score=0.3,
            experience_relevance_score=0.7, final_match_score=0.5,
            years_experience=5, graduation_year=2018,
        )
        result = predict(features)[0]
        assert 0 <= result['confidence'] <= 1

    def test_missing_features_filled_with_zero(self):
        result = predict({'semantic_similarity': 0.5})[0]
        assert result['classification'] in ['Authentic', 'Suspicious', 'Potentially Fake']


class TestModelInfo:

    def test_returns_all_keys(self):
        info = get_model_info()
        assert 'feature_names' in info
        assert 'params' in info
        assert 'test_accuracy' in info
        assert 'test_f1' in info
        assert 'feature_importance' in info
        assert 'classes' in info

    def test_accuracy_is_high(self):
        info = get_model_info()
        assert info['test_accuracy'] >= 0.9

    def test_f1_is_high(self):
        info = get_model_info()
        assert info['test_f1'] >= 0.9

    def test_feature_importance_has_17_entries(self):
        info = get_model_info()
        assert len(info['feature_importance']) == 17

    def test_feature_importance_sums_to_one(self):
        info = get_model_info()
        total = sum(f['importance'] for f in info['feature_importance'])
        assert abs(total - 1.0) < 0.01
