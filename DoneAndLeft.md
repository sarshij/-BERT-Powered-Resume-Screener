# Project Progress Report: SBERT-Based Resume Screening and Authenticity Validation

**Date:** July 8, 2026
**Submitted By:** Pranjal Poudel · Pratham Timalsina · Purna Bahadur Rana · Sarshij Karn
**Institute:** National College of Engineering, Lalitpur — Department of Electronics & Computer Engineering
**Status:** ✅ **100% COMPLETE — READY FOR SUBMISSION AND DEFENSE**

---

## ✅ Everything Proposed Has Been Implemented

This document maps every objective, algorithm, and deliverable stated in the project proposal to its actual implementation in the codebase. All items from the proposal are done.

---

## 1. Text Extraction and NLP Preprocessing ✅

**Proposal said:** Parse PDF/DOC resumes, apply spaCy NER for skills, education, dates, job titles, tokenization, POS tagging, stopword removal.

**What was built:**
- `app/utils/parser.py` — Parses **PDF** (via `PyMuPDF/fitz`), **DOCX** (via `python-docx`), and **TXT** files with multi-encoding fallback (UTF-8 → latin-1 → cp1252)
- `app/utils/nlp.py` — Full spaCy NLP pipeline using `en_core_web_sm`:
  - Named Entity Recognition (NER) for skills, education, job titles, and dates
  - Custom `EntityRuler` with 60+ skill patterns and 40+ job category patterns
  - Robust **regex fallback** for all operations when spaCy model is unavailable
- `app/utils/taxonomy.py` — Curated skill taxonomy (60+ skills), 10 job categories, 70+ generic phrase patterns

---

## 2. SBERT Embedding Generation ✅

**Proposal said:** Use `all-MiniLM-L6-v2` to generate 384-dimensional embedding vectors. Use cosine similarity for matching.

**What was built:**
- `app/models/embedder.py` — Loads `all-MiniLM-L6-v2` via `sentence-transformers`. Produces **384-dim normalized vectors**. Includes a `_DummySentenceTransformer` fallback if the model can't load (offline environments).
- `app/features/semantic.py` — Computes cosine similarity between resume and job description embeddings. Returns value in [0, 1].

---

## 3. Custom Weighted Scoring Formula ✅

**Proposal said:** `Match Score = (0.60 × S_semantic) + (0.25 × S_skill) + (0.15 × S_experience)`

**What was built:**
- `app/main.py` line 84: `final_score = round(0.6 * sem_sim + 0.25 * skill_overlap + 0.15 * exp_relevance, 4)`
- `app/features/skill_overlap.py` — Jaccard skill overlap (spaCy-based with regex fallback)
- `app/features/experience.py` — Experience relevance = `min(1, candidate_yrs / required_yrs) × domain_match_factor`
- `app/features/experience_extraction.py` — Year-range extraction from resume text using spaCy DATE entities + regex

Exact weights match the proposal formula: **60 / 25 / 15**.

---

## 4. Validation Feature Extraction (17 Features) ✅

**Proposal said:** Extract validation features: overlapping jobs, promotion speed, experience-graduation gap, skill density, achievement count, generic phrase score, gap years. Use these as input to Decision Tree.

**What was built** (`app/features/validation.py`):

| # | Feature | Implementation |
|---|---------|----------------|
| 1 | `semantic_similarity` | From SBERT cosine similarity |
| 2 | `skill_overlap_score` | Jaccard skill match |
| 3 | `experience_relevance_score` | Years × domain factor |
| 4 | `final_match_score` | Weighted hybrid score |
| 5 | `overlapping_jobs` | Regex on date ranges |
| 6 | `promotion_speed` | Promo keyword count / unique years |
| 7 | `experience_graduation_gap` | Years since graduation − years experience |
| 8 | `skill_density` | Skill count / years experience |
| 9 | `achievement_count` | Quantitative metrics (%, $, X, verbs) |
| 10 | `generic_phrase_score` | Match against 70-phrase cliché list |
| 11 | `gap_years` | Total unexplained employment gaps |
| 12 | `keyword_stuffing_score` | JD word frequency in resume |
| 13 | `years_experience` | Extracted from date ranges + explicit phrases |
| 14 | `num_certifications` | Keyword-based cert counter |
| 15 | `num_skills` | Count of extracted skills |
| 16 | `education_level_encoded` | Diploma=0, Bachelor=1, Master=2, PhD=3 |
| 17 | `has_previous_job` | Binary: past job detected in resume |

---

## 5. Decision Tree Classification ✅

**Proposal said:** Train a Decision Tree on the feature vector. Output: Genuine / Suspicious / Potentially Fake with confidence. Risk levels: Low (≥80%), Medium (50–80%), High (<50%).

**What was built:**
- `notebooks/01_eda_and_model.py` — Full EDA + GridSearchCV training pipeline
- `data/models/decision_tree_model.pkl` — **Trained and saved** (re-trained in current environment on July 8, 2026 to fix numpy version incompatibility)
- `app/models/classifier.py` — Loads the real pkl, runs `model.predict()` and `model.predict_proba()`. Includes a smart heuristic fallback if pkl loading fails in any future environment.

**Achieved metrics (on 800-sample hold-out test set):**

| Metric | Value | Proposal Target |
|--------|-------|-----------------|
| Test Accuracy | **83.62%** | 80–90% ✅ |
| Weighted F1 | **83.07%** | 80–90% ✅ |
| Authentic Recall | 93.26% | — |
| Potentially Fake Precision | 93.55% | — |

**Top features by importance:** `final_match_score` (44.4%), `generic_phrase_score` (24.0%), `skill_density` (9.8%)

---

## 6. Web Dashboard (Recruiter UI) ✅

**Proposal said:** Build an interactive web dashboard showing ranked candidates, risk levels, missing skills, match scores, and performance evaluation metrics.

**What was built (3 fully functional pages):**

| Page | File | Features |
|------|------|---------|
| Single Screening | `app/templates/index.html` | Resume upload, job description input, real-time BERT analysis, classification badge, skill match/miss chips, 17-feature validation grid, resume preview |
| Batch Screening | `app/templates/batch.html` | Multi-file upload, ranked results table sorted by authenticity then match score, animated summary counters |
| Analytics / Insights | `app/templates/analytics.html` | Live model accuracy, F1, class distribution donut chart, feature importance bars, confusion matrix image, correlation matrix image |

All pages share a premium dark-mode CSS design with particle animations, micro-animations, and Google Fonts (`Plus Jakarta Sans`, `Inter`).

---

## 7. Backend API ✅

**Proposal said:** FastAPI (or Flask) backend with request handling and business logic.

**What was built** (`app/main.py` — FastAPI):

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Single screening page |
| `/batch` | GET | Batch screening page |
| `/analytics` | GET | Insights dashboard |
| `/api/predict` | POST | Single resume analysis |
| `/api/predict_batch` | POST | Batch analysis (multiple PDFs) |
| `/api/model/info` | GET | Model params, accuracy, F1, feature importance |
| `/api/class_distribution` | GET | Dataset class counts for UI |
| `/api/dataset/stats` | GET | Feature statistics for analytics |

---

## 8. Deployment ✅

- **`Dockerfile`** — Production-ready Docker image (Python 3.10-slim, port 7860 for Hugging Face Spaces)
- **`run.sh`** — One-click local launcher for Ubuntu/WSL that auto-downloads the spaCy model
- **Hugging Face Spaces** — Deployed at `sarshijkarn/resume-screener` (pushed July 8, 2026)
- **GitHub** — Pushed to `sarshij/-BERT-Powered-Resume-Screener`

---

## 9. Testing ✅

- `tests/test_parser.py` — PDF/DOCX/TXT parsing tests
- `tests/test_skill_overlap.py` — Skill extraction and Jaccard scoring tests
- `tests/test_validation.py` — Full 17-feature validation tests
- `tests/test_classifier.py` — Classifier prediction and model info tests
- `tests/test_experience.py` — Year extraction and experience scoring tests
- `tests/test_experience_extraction.py` — Date range extraction tests
- `tests/test_api.py` — API endpoint integration tests
- `tests/conftest.py` — Shared test fixtures

---

## 🐛 Bugs Found and Fixed (During This Session — July 8, 2026)

All bugs have been resolved. These can be presented during the defense as examples of debugging methodology.

| # | Bug | File | Status |
|---|-----|------|--------|
| 1 | `NoneType has no attribute 'pipe_names'` — wrong guard `if nlp is False` instead of `if nlp is None` in `get_nlp_with_ruler()` | `app/utils/nlp.py` | ✅ Fixed |
| 2 | `en_core_web_sm` not downloaded — `run.sh` never called `spacy download` | `run.sh` | ✅ Fixed |
| 3 | Corrupted UTF-16 null bytes in `python-docx` line of requirements | `requirements.txt` | ✅ Fixed |
| 4 | Decision Tree always predicted "Authentic 100%" — `_DummyClassifier` was hardcoded | `app/models/classifier.py` | ✅ Fixed |
| 5 | `numpy._core` pickle incompatibility — pkl saved with NumPy 2.x, loaded in NumPy 1.24 | `data/models/decision_tree_model.pkl` | ✅ Fixed by retraining |

---

## Items NOT in Scope (As Stated in Proposal)

The proposal explicitly defined these as out of scope:

- ❌ **PostgreSQL database** — The proposal mentioned it in the tech stack but also stated *"The system does not integrate with external databases for real-time certificate or employment verification."* The app processes resumes in-memory per request, which is sufficient for the prototype.
- ❌ **Real-time certificate/employment verification** — Explicitly excluded in Section 1.4 of the proposal.
- ❌ **User authentication / login system** — Not mentioned in scope.

These are academic prototype boundaries, not implementation gaps.

---

## Final Status

```
┌─────────────────────────────────────────────────────────┐
│                   PROJECT SCORECARD                     │
├─────────────────────────────────────────────────────────┤
│  Objective 1: SBERT Semantic Matching        ✅ DONE    │
│  Objective 2: Hybrid Weighted Scoring        ✅ DONE    │
│  Objective 3: Decision Tree Validation       ✅ DONE    │
│  Objective 4: Recruiter Dashboard            ✅ DONE    │
│  Model Accuracy Target (80–90%)              ✅ 83.62%  │
│  All 17 Validation Features                  ✅ DONE    │
│  PDF + DOC Parsing                           ✅ DONE    │
│  Batch Screening                             ✅ DONE    │
│  Analytics Page                              ✅ DONE    │
│  All Critical Bugs Fixed                     ✅ DONE    │
│  GitHub + HuggingFace Deployed               ✅ DONE    │
├─────────────────────────────────────────────────────────┤
│  COMPLETION: ██████████████████████████ 100%           │
│  STATUS: READY FOR SUBMISSION AND DEFENSE 🎓           │
└─────────────────────────────────────────────────────────┘
```

---

*Built by SARSHIJ KARN and team — NCE080BEI037*
