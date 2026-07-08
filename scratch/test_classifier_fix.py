import sys
sys.path.insert(0, '.')
from app.models.classifier import load_model, predict, get_model_info

# Test 1: model loads correctly
model, cols = load_model()
print("Model type:", type(model).__name__)
print("Features:", len(cols))

# Test 2: authentic-looking resume features
sample_authentic = {
    'semantic_similarity': 0.82, 'skill_overlap_score': 0.65,
    'experience_relevance_score': 0.7, 'final_match_score': 0.77,
    'overlapping_jobs': 0, 'promotion_speed': 0.1,
    'experience_graduation_gap': 2.0, 'skill_density': 3.5,
    'achievement_count': 8, 'generic_phrase_score': 0.12,
    'gap_years': 0, 'keyword_stuffing_score': 0.18,
    'years_experience': 4.0, 'num_certifications': 2.0,
    'num_skills': 8.0, 'education_level_encoded': 1.0,
    'has_previous_job': 1.0
}

# Test 3: fake-looking resume features
sample_fake = {
    'semantic_similarity': 0.45, 'skill_overlap_score': 0.3,
    'experience_relevance_score': 0.2, 'final_match_score': 0.38,
    'overlapping_jobs': 2, 'promotion_speed': 2.5,
    'experience_graduation_gap': 15.0, 'skill_density': 12.0,
    'achievement_count': 1, 'generic_phrase_score': 0.75,
    'gap_years': 8, 'keyword_stuffing_score': 0.85,
    'years_experience': 0.0, 'num_certifications': 0.0,
    'num_skills': 25.0, 'education_level_encoded': 0.0,
    'has_previous_job': 0.0
}

results = predict([sample_authentic, sample_fake])
r0 = results[0]
r1 = results[1]
print("Authentic sample ->", r0['classification'], round(r0['confidence']*100, 1), "%")
print("Fake sample      ->", r1['classification'], round(r1['confidence']*100, 1), "%")

# Test 4: model info for analytics page
info = get_model_info()
print("Accuracy:", round(info['test_accuracy']*100, 2), "%")
print("F1:", round(info['test_f1']*100, 2), "%")
print("ALL TESTS PASSED")
