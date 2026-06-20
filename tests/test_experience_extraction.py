from app.features.experience_extraction import extract_years_experience, extract_graduation_year


class TestExtractYearsExperience:

    def test_range_based_extraction(self, sample_resume):
        years = extract_years_experience(sample_resume)
        assert 5 <= years <= 15

    def test_finds_explicit_mention(self):
        text = "Experienced professional with 10 years of experience in software development."
        years = extract_years_experience(text)
        assert years >= 9

    def test_no_experience_info_returns_zero(self, empty_resume):
        years = extract_years_experience(empty_resume)
        assert years == 0.0

    def test_no_experience_in_generic_text(self, generic_resume):
        years = extract_years_experience(generic_resume)
        assert years == 0.0

    def test_single_date_range(self):
        text = "Engineer at Company 2020 - 2024"
        years = extract_years_experience(text)
        assert years == 4.0

    def test_overlapping_ranges_merged(self):
        text = "Job A 2018 - 2022\n Job B 2020 - 2024"
        years = extract_years_experience(text)
        assert years == 6.0

    def test_present_to_current_year(self):
        text = "Current Job 2020 - Present"
        from datetime import datetime
        expected = datetime.now().year - 2020
        years = extract_years_experience(text)
        assert years == expected


class TestExtractGraduationYear:

    def test_finds_graduation_in_education_section(self, sample_resume):
        year = extract_graduation_year(sample_resume)
        assert year == 2018

    def test_no_education_info_returns_default(self, generic_resume):
        year = extract_graduation_year(generic_resume)
        from datetime import datetime
        assert datetime.now().year - 6 <= year <= datetime.now().year - 4

    def test_empty_text_returns_default(self, empty_resume):
        from datetime import datetime
        year = extract_graduation_year(empty_resume)
        assert year == datetime.now().year - 4

    def test_finds_year_with_keyword_proximity(self):
        text = "Graduated in 2016 with a Bachelor of Science in Computer Science from Stanford University."
        year = extract_graduation_year(text)
        assert year == 2016
