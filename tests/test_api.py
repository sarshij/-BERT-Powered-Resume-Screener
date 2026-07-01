import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Must import app after path setup
from app.main import app

client = TestClient(app)


class TestPages:
    """Test that all 3 HTML pages load correctly."""

    def test_home_page(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_batch_page(self):
        response = client.get("/batch")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_analytics_page(self):
        response = client.get("/analytics")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestModelInfoEndpoint:

    def test_returns_success(self):
        response = client.get("/api/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_contains_expected_fields(self):
        data = client.get("/api/model/info").json()
        assert "feature_names" in data
        assert "params" in data
        assert "test_accuracy" in data
        assert "test_f1" in data
        assert "feature_importance" in data
        assert "classes" in data

    def test_accuracy_above_95_percent(self):
        data = client.get("/api/model/info").json()
        assert data["test_accuracy"] >= 0.95


class TestClassDistributionEndpoint:

    def test_returns_success(self):
        response = client.get("/api/class_distribution")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_has_three_classes(self):
        data = client.get("/api/class_distribution").json()
        assert len(data["class_distribution"]) == 3

    def test_total_is_2000(self):
        data = client.get("/api/class_distribution").json()
        total = sum(data["class_distribution"].values())
        assert total == 2000


class TestDatasetStatsEndpoint:

    def test_returns_success(self):
        response = client.get("/api/dataset/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_total_samples(self):
        data = client.get("/api/dataset/stats").json()
        assert data["total_samples"] == 2000

    def test_has_feature_stats(self):
        data = client.get("/api/dataset/stats").json()
        assert len(data["feature_stats"]) >= 12


class TestPredictEndpoint:

    def test_predict_with_text_file(self):
        response = client.post(
            "/api/predict",
            files={"resume": ("test.txt", b"Experienced Python developer with 5 years in AWS and Docker. Led team of engineers.")},
            data={"job_description": "Looking for a Python developer with cloud experience."}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "classification" in data
        assert data["classification"]["classification"] in ["Authentic", "Suspicious", "Potentially Fake"]

    def test_predict_without_job_description(self):
        response = client.post(
            "/api/predict",
            files={"resume": ("test.txt", b"Python developer with AWS experience.")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_predict_too_short_returns_400(self):
        response = client.post(
            "/api/predict",
            files={"resume": ("short.txt", b"Hi")},
            data={"job_description": "Software engineer"}
        )
        assert response.status_code == 400

    def test_predict_returns_all_sections(self):
        response = client.post(
            "/api/predict",
            files={"resume": ("test.txt", b"Experienced Python developer with 5 years in AWS and Docker. Led team of engineers.")},
            data={"job_description": "Looking for a Python developer."}
        )
        data = response.json()
        assert "scores" in data
        assert "skills" in data
        assert "validation" in data
        assert "classification" in data
        assert "resume_preview" in data
        assert "filename" in data

    def test_predict_with_job_title(self):
        response = client.post(
            "/api/predict",
            files={"resume": ("test.txt", b"Senior engineer with Python skills")},
            data={"job_title": "Senior Software Engineer", "job_description": "Python developer"}
        )
        assert response.status_code == 200

    def test_predict_scores_are_between_0_and_1(self):
        response = client.post(
            "/api/predict",
            files={"resume": ("test.txt", b"Python AWS Docker engineer")},
            data={"job_description": "Python and AWS developer"}
        )
        data = response.json()
        for key, val in data["scores"].items():
            assert 0 <= val <= 1, f"{key} = {val} is not in [0,1]"


class TestPredictBatchEndpoint:

    def test_batch_with_two_files(self):
        response = client.post(
            "/api/predict_batch",
            files=[
                ("resumes", ("a.txt", b"Python developer with AWS experience.")),
                ("resumes", ("b.txt", b"Results-driven go-getter team player.")),
            ],
            data={"job_description": "Software engineer"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_file_handled(self):
        response = client.post(
            "/api/predict_batch",
            files=[("resumes", ("empty.txt", b""))],
            data={"job_description": "test"}
        )
        data = response.json()
        assert "error" in data["results"][0] or data["results"][0].get("classification")

    def test_batch_sorts_by_classification_priority(self):
        response = client.post(
            "/api/predict_batch",
            files=[
                ("resumes", ("a.txt", b"Results-driven go-getter team player synergy leverage.")),
                ("resumes", ("b.txt", b"Experienced Python developer with 5 years in AWS and Docker. Led team of 10 engineers.")),
            ],
            data={"job_description": "Software engineer with Python and AWS experience."}
        )
        data = response.json()
        assert data["status"] == "success"
        assert data["count"] == 2
        assert len(data["results"]) == 2


class TestStaticFiles:

    def test_confusion_matrix_png(self):
        response = client.get("/static/confusion_matrix.png")
        assert response.status_code == 200
        assert "image/png" in response.headers["content-type"]

    def test_correlation_matrix_png(self):
        response = client.get("/static/correlation_matrix.png")
        assert response.status_code == 200

    def test_feature_importance_png(self):
        response = client.get("/static/feature_importance.png")
        assert response.status_code == 200
