# BERT-Based Resume Screening & Authenticity Validation

## Project Overview
Automated resume screening system using Sentence-BERT embeddings for semantic understanding and Decision Tree classification for authenticity validation. Classifies resumes as **Authentic**, **Suspicious**, or **Potentially Fake**.

## Features
- **Single Resume Screening** — Upload a PDF/TXT resume + job description → instant analysis
- **Batch Screening** — Upload multiple resumes → ranked table sorted by authenticity + match score
- **BERT Semantic Similarity** — SBERT embeddings compare resume content to job description
- **Skill Overlap Analysis** — Matched, missing, and extra skills visualization
- **Validation Feature Extraction** — 12 features including keyword stuffing, generic phrase detection, gap analysis, skill density
- **Decision Tree Classifier** — 97% accuracy, trained on 3000 labeled resumes
- **Analytics Dashboard** — Model metrics, feature importance, dataset statistics

## Tech Stack
- **Python 3.10+** — pandas, numpy, scikit-learn, sentence-transformers, PyMuPDF
- **FastAPI** — REST API backend
- **HTML/CSS/JS** — Frontend dashboard (no framework needed)
- **SBERT** (`all-MiniLM-L6-v2`) — Lightweight sentence embeddings (384-dim)

## Model Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | **97.0%** |
| Weighted F1 | **97.0%** |
| Authentic F1 | 0.97 |
| Suspicious F1 | 0.95 |
| Potentially Fake F1 | 1.00 |
| Best Params | max_depth=7, entropy, min_samples_leaf=5 |

**Top Features:** skill_overlap_score (46%), generic_phrase_score (23%), skill_density (17%)

## Project Structure
```
resume-screener/
├── app/
│   ├── main.py                 # FastAPI server (routes, endpoints)
│   ├── models/
│   │   ├── embedder.py         # SBERT embedding pipeline
│   │   └── classifier.py       # Decision Tree wrapper
│   ├── features/
│   │   ├── semantic.py         # BERT similarity computation
│   │   ├── skill_overlap.py    # Skill extraction & Jaccard scoring
│   │   ├── experience.py       # Experience relevance scoring
│   │   └── validation.py       # 12 validation features
│   ├── utils/
│   │   ├── parser.py           # PDF/TXT text extraction
│   │   └── taxonomy.py         # Skills list, job categories, generic phrases
│   ├── static/                 # Images for analytics dashboard
│   └── templates/              # HTML pages (index.html, batch.html, analytics.html)
├── data/
│   ├── processed/              # Cleaned dataset, metrics, visualizations
│   └── models/                 # Trained decision_tree_model.pkl
├── notebooks/
│   ├── 01_eda_and_model.py     # EDA + model training script
├── requirements.txt
├── run.sh
└── README.md
```

## Quick Start
```bash
cd resume-screener
./run.sh
# Open http://localhost:8000
```

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Single resume upload page |
| GET | `/batch` | Batch resume upload page |
| GET | `/analytics` | Model & dataset analytics |
| POST | `/api/predict` | Single resume analysis |
| POST | `/api/predict_batch` | Batch resume screening |
| GET | `/api/model/info` | Model metadata & feature importance |
| GET | `/api/class_distribution` | Dataset class distribution |
| GET | `/api/dataset/stats` | Dataset feature statistics |

---
**Built by SARSHIJ KARN**
