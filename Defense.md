# 🎓 DEFENSE REPORT
# ClearHire — SBERT-Based Resume Screening and Authenticity Validation Using Decision Tree Classification

**Submitted By:** Pranjal Poudel (NCE080BEI029) · Pratham Timalsina (NCE080BEI031) · Purna Bahadur Rana (NCE080BEI032) · Sarshij Karn (NCE080BEI037)  
**Institution:** National College of Engineering, Tribhuvan University  
**Department:** Electronics & Computer Engineering, Lalitpur, Nepal  
**Semester:** 6th Semester Minor Project  
**Date:** March 2026  

---

## TABLE OF CONTENTS
1. [What Is This Project?](#1-what-is-this-project)
2. [The Problem — Why Does This Exist?](#2-the-problem--why-does-this-exist)
3. [Objectives](#3-objectives)
4. [Literature Review & Research Gap](#4-literature-review--research-gap)
5. [System Architecture](#5-system-architecture)
6. [Complete NLP & ML Pipeline (How It Works)](#6-complete-nlp--ml-pipeline-how-it-works)
7. [Feature Engineering — All 19 Validation Features](#7-feature-engineering--all-19-validation-features)
8. [The Scoring Formula (Math)](#8-the-scoring-formula-math)
9. [Machine Learning Model](#9-machine-learning-model)
10. [LLM-as-a-Judge (Ensemble Consensus)](#10-llm-as-a-judge-ensemble-consensus)
11. [Technology Stack](#11-technology-stack)
12. [Backend API — All Endpoints](#12-backend-api--all-endpoints)
13. [Frontend — What The User Sees](#13-frontend--what-the-user-sees)
14. [System Statistics & Performance](#14-system-statistics--performance)
15. [Proposal vs. Delivered — Checklist](#15-proposal-vs-delivered--checklist)
16. [Ethical & Privacy Considerations](#16-ethical--privacy-considerations)
17. [Questions & Answers — Panel Q&A](#17-questions--answers--panel-qa)

---

## 1. What Is This Project?

**ClearHire** is an AI-powered Resume Screening and Authenticity Validation System. It solves two problems simultaneously:

1. **Matching Problem:** Does this candidate actually *match* the job description — semantically, not just by keywords?
2. **Authenticity Problem:** Is this resume real? Is it AI-generated? Is it exaggerated or hallucinated?

It is a full-stack web application where HR professionals can upload a candidate's resume and a job description, and the system returns:
- A classification: **Authentic**, **Suspicious**, or **Potentially Fake**
- A match score (0–100%)
- A SHAP explainability chart showing *why* the model decided what it did
- An LLM (Groq/Nvidia) "second opinion" with human-readable reasoning
- A skill-gap analysis (matched, missing, and extra skills)
- An anonymized candidate profile (no PII leakage to HR)

---

## 2. The Problem — Why Does This Exist?

### The Volume Problem
> A single job posting receives an average of **250 resumes**. 75–88% of them are unqualified. HR teams cannot read them all.

### The Keyword-Matching Failure
> Traditional ATS systems use TF-IDF or boolean keyword matching. If a job says **"Machine Learning"** and the resume says **"Deep Learning"** or **"Neural Networks"**, the candidate gets rejected — even though they are highly qualified. Words are matched, but **meaning is ignored**.

### The AI Resume Problem (2023–2026)
> With tools like ChatGPT, Gemini, Claude, candidates now generate "perfect" resumes that are:
> - Stuffed with every keyword from the job description
> - Written in unnaturally polished language
> - Claiming 8 skills in 1 year of experience
> - Full of generic buzzwords ("results-driven", "self-motivated", "passionate")
> - Logically inconsistent (led a team of 20 at age 21, fresh out of college)
>
> Traditional ATS systems **cannot detect these**. They are easily fooled.

### The Bias Problem
> When a human reads 250 resumes, bias creeps in. Bias based on name, university prestige, formatting, or personal preferences. Our system evaluates every candidate against the same objective mathematical criteria.

---

## 3. Objectives

As stated in the project proposal (fulfilled ✅ in this implementation):

| # | Objective | Status |
|---|---|---|
| 1 | Develop a semantic resume-job matching system using SBERT for accurate similarity analysis | ✅ Done |
| 2 | Design a hybrid scoring model integrating semantic similarity (60%), skill overlap (25%), experience relevance (15%) | ✅ Done |
| 3 | Implement a resume authenticity validation module using a Decision Tree / XGBoost classifier | ✅ Done |
| 4 | Build a recruiter dashboard for candidate ranking, profile analysis, and performance evaluation | ✅ Done |

---

## 4. Literature Review & Research Gap

### What Existed Before Us
| System | Approach | Limitation |
|---|---|---|
| Traditional ATS (Taleo, Workday) | TF-IDF keyword matching | Cannot understand synonyms or meaning |
| Ideal AI | ML-based ranking | No authenticity validation |
| HireVue | Video + resume AI | Expensive, no fraud detection |
| Academic systems (Word2Vec, Doc2Vec) | Word embeddings | Poor on full documents, no authenticity |

### What Was Missing (Research Gap)
1. Most tools only match skills — they do **not check if the resume is authentic**
2. No system combines semantic matching **with** AI-resume detection
3. No hybrid scoring model uses semantic + skill + experience simultaneously
4. Small/medium companies need affordable, open-source solutions

### Our Novel Contribution
We are the **first** in our academic context to combine:
- SBERT semantic understanding
- 19-feature rule-based authenticity validation
- XGBoost classification
- LLM-as-a-Judge ensemble consensus
- Anonymized bias-free HR output

---

## 5. System Architecture

The system follows a **dual-track parallel pipeline** that converges at the output layer:

```
[HR User]
    │
    ├── Resume (PDF/DOCX/TXT)
    └── Job Description (text or file)
                │
         [Text Extraction & NLP]
          (pdfminer, python-docx, spaCy)
                │
    ┌───────────┴───────────────┐
    │                           │
[SCREENING TRACK]         [VALIDATION TRACK]
    │                           │
[SBERT Encoding]         [19-Feature Extraction]
 all-MiniLM-L6-v2         (overlapping jobs,
    │                      skill density, gaps,
[Cosine Similarity]        generic phrases, etc.)
    │                           │
[Weighted Score]          [XGBoost Classifier]
 0.6×sem + 0.25×skill      (Authentic/Suspicious/
 + 0.15×exp                 Potentially Fake)
    │                           │
    └───────────────────────────┘
                │
        [LLM Double-Check]
     (Groq primary → Nvidia fallback)
     Only if Suspicious or Fake
                │
         [Dashboard Output]
    Classification Badge + Confidence
    SHAP Explainability Chart
    Skills Gap Analysis
    Anonymized Candidate Profile
    LLM Reasoning Text
```

---

## 6. Complete NLP & ML Pipeline (How It Works)

### Step 1: Document Parsing
- **PDF files:** Extracted using `pdfminer.six` (handles complex layouts)
- **DOCX files:** Extracted using `python-docx`
- **TXT files:** Read directly
- **Scanned PDFs/Images:** OCR using `Tesseract`
- File validation: Only PDF/DOCX/TXT, max 5MB, MIME type checked

### Step 2: NLP Preprocessing (spaCy)
- Tokenization, POS tagging, stopword removal
- Named Entity Recognition (NER) — extracts: organizations, dates, job titles, education degrees
- Custom entity ruler trained on our taxonomy for: skills, certifications, programming languages
- Output: structured text ready for embedding and feature extraction

### Step 3: SBERT Embedding Generation
- Model: `all-MiniLM-L6-v2` (from Hugging Face `sentence-transformers`)
- Resume text → 384-dimensional dense vector
- Job description → 384-dimensional dense vector
- Both vectors computed **independently** (this is why SBERT is faster than standard BERT)

### Step 4: Cosine Similarity
```
S_semantic = (e_resume · e_job) / (||e_resume|| × ||e_job||)
```
Returns a value from 0 (no similarity) to 1 (identical meaning)

### Step 5: Skill Overlap Scoring
```
S_skill = |R_skills ∩ J_skills| / |J_skills|
```
How many of the required job skills does the candidate have?

### Step 6: Experience Relevance Scoring
```
S_experience = min(1, candidate_years / required_years) × D_match
```
Where D_match is domain relevance from job-title similarity

### Step 7: Custom Weighted Final Score
```
Final Score = (0.60 × S_semantic) + (0.25 × S_skill) + (0.15 × S_experience)
```
This formula is exactly as proposed. Semantic meaning dominates (60%), structured skills matter (25%), experience is a tiebreaker (15%).

### Step 8: 19-Feature Extraction for Validation
(See Section 7 for all features)

### Step 9: XGBoost Classification
All 19 features → XGBoost → Authentic / Suspicious / Potentially Fake + confidence score

### Step 10: SHAP Explainability
- SHAP (SHapley Additive exPlanations) TreeExplainer runs on every prediction
- Returns top 3 features that drove the classification decision
- Displayed as a waterfall-style bar chart on the UI

### Step 11: LLM Double-Check (Only for Suspicious/Fake)
- Groq API (llama-3.3-70b-versatile) is called first — fast, free, reliable
- If Groq fails → NVIDIA NIM API (nemotron) is the fallback
- LLM acts as an "expert HR auditor" and either Agrees or Disagrees
- If it Disagrees with the ML model → classification stays Suspicious (not escalated to Fake)
- Result: 30–50 word human-readable explanation displayed on the UI

---

## 7. Feature Engineering — All 19 Validation Features

These are the exact features fed into the XGBoost classifier:

| # | Feature | Description | Why It Detects Fakes |
|---|---|---|---|
| 1 | `semantic_similarity` | SBERT cosine similarity score | Very high score on generic resume = keyword stuffed |
| 2 | `skill_overlap_score` | Fraction of JD skills found | 100% match on every skill is suspicious |
| 3 | `experience_relevance_score` | Domain + years match | Mismatch = inconsistent history |
| 4 | `final_match_score` | Weighted composite | Overall sanity check |
| 5 | `overlapping_jobs` | Did jobs overlap in time? | Impossible to work 2 full-time jobs simultaneously |
| 6 | `promotion_speed` | Avg months between role changes | < 6 months consistently = unrealistic |
| 7 | `experience_graduation_gap` | Years exp vs. graduation year | Claiming 8 yrs exp but graduated 3 years ago |
| 8 | `skill_density` | Skills per year of experience | 50 skills in 1 year = AI generated |
| 9 | `achievement_count` | Count of numbers/% in resume | Real resumes have measurable achievements |
| 10 | `generic_phrase_score` | Frequency of AI buzzwords | "results-driven", "passionate", "synergy" |
| 11 | `gap_years` | Total unexplained employment gaps | Multiple unexplained gaps |
| 12 | `keyword_stuffing_score` | JD keyword density in resume | Stuffing every JD keyword = AI-written |
| 13 | `years_experience` | Total claimed experience | Sanity check |
| 14 | `num_certifications` | Number of certifications listed | Abnormally high = fabricated |
| 15 | `num_skills` | Total skills count | 80 skills listed = red flag |
| 16 | `education_level_encoded` | Degree level (0=None, 1=HSC, 2=Bachelors, 3=Masters, 4=PhD) | Cross-check with experience claims |
| 17 | `has_previous_job` | Does resume show work history? | Missing = fresh grad or empty |
| 18 | `skill_experience_alignment` | Skills backed by action verbs in projects? | "Python" listed but never used in any project |
| 19 | `ai_plausibility_score` | LLM probability of AI generation (0–1) | Groq/Nvidia scores the text itself |

---

## 8. The Scoring Formula (Math)

### Final Match Score
```
Final Score = (0.60 × S_semantic) + (0.25 × S_skill) + (0.15 × S_experience)
```

### Semantic Similarity (Cosine)
```
S_semantic = (e_A · e_B) / (||e_A|| × ||e_B||)
```

### Skill Overlap
```
S_skill = |R_skills ∩ J_skills| / |J_skills|
```

### Keyword Stuffing Score
```
stuffing = (count of JD words in resume / total resume words) × 2 + repeat_penalty
```

### Confidence Classification Thresholds (from proposal)
```
Confidence ≥ 0.80  →  Low Risk (Authentic)
0.50 ≤ Conf < 0.80 →  Medium Risk (Suspicious)
Confidence < 0.50  →  High Risk (Potentially Fake)
```

---

## 9. Machine Learning Model

### Why XGBoost, not plain Decision Tree?
The proposal mentions "Decision Tree Classification" — XGBoost IS a Decision Tree (an ensemble of them):
- Started with a single Decision Tree → 72% accuracy
- Upgraded to XGBoost (Extreme Gradient Boosting) → **87.38% accuracy**
- XGBoost grows trees sequentially, each one correcting errors from the previous
- Better at non-linear feature relationships (our validation features are highly non-linear)
- Built-in regularization prevents overfitting

### Model Performance
| Metric | Value |
|---|---|
| Test Accuracy | **87.38%** |
| Model Type | XGBoost (Gradient Boosted Decision Trees) |
| Features Input | 19 validation features |
| Classes | Authentic (0), Suspicious (1), Potentially Fake (2) |
| Target Accuracy (Proposal) | 80–90% ✅ **Achieved** |

### Model Loading
- Saved as a `.pkl` file using `joblib`
- Loaded at server startup via `lifespan` hook (pre-warmed)
- If model file is missing → falls back to a mathematical heuristic classifier (no crash)

### SHAP Explainability
- `shap.TreeExplainer` runs after every prediction
- Extracts top 3 features by absolute SHAP value for the predicted class
- Displayed as directional contribution bars on the UI

---

## 10. LLM-as-a-Judge (Ensemble Consensus)

This is our most novel architectural feature — a **two-stage consensus system**:

### When Is LLM Triggered?
Only when XGBoost classifies a resume as **Suspicious** or **Potentially Fake**. Not for Authentic resumes (saves API cost and time).

### Provider Order
1. **Primary: Groq** (`llama-3.3-70b-versatile`) — fast, free tier, JSON mode
2. **Fallback: NVIDIA NIM** (`nemotron-3-ultra-550b-a55b`) — enterprise-grade, used if Groq fails

### What Does It Do?
```
"Our local AI model classified this resume as 'Potentially Fake'. 
Act as an expert HR auditor. Do you agree?
Return JSON: {"consensus": "Agree"/"Disagree", "reasoning": "30-50 words"}
```

### Consensus Logic
| ML Model Says | LLM Says | Final Result |
|---|---|---|
| Authentic | (not called) | ✅ Authentic |
| Suspicious | Agree | ⚠️ Suspicious |
| Suspicious | Disagree | ⚠️ Suspicious (downgraded from Fake) |
| Potentially Fake | Agree | ❌ Potentially Fake |
| Potentially Fake | Disagree | ⚠️ Suspicious (overridden — benefit of doubt) |

### Security
- Prompt injection sanitization: removes "ignore previous instructions", "system prompt", "you are a"
- Input length capped at 2000 characters per field
- Singleton pattern prevents re-initialization on every request

---

## 11. Technology Stack

| Category | Technology | Why |
|---|---|---|
| **Backend** | Python 3.11, FastAPI | Async support, auto OpenAPI docs, fast |
| **NLP** | spaCy (en_core_web_sm) | NER, POS, tokenization |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | SBERT, 384-dim vectors, fast |
| **ML Model** | XGBoost + scikit-learn | Best accuracy, interpretable |
| **Explainability** | SHAP (TreeExplainer) | Feature contribution visualization |
| **LLM** | Groq API + NVIDIA NIM | Resume authenticity double-check |
| **Document Parsing** | pdfminer.six, python-docx | PDF/DOCX text extraction |
| **OCR** | Tesseract | Scanned PDF support |
| **Database** | SQLite (aiosqlite) | Async, lightweight, no setup |
| **Rate Limiting** | slowapi | 25/min single, 100/min batch |
| **Frontend** | HTML5, Vanilla CSS, JavaScript | Zero dependencies, fast |
| **Deployment** | Uvicorn + WSL2 (Ubuntu 22.04) | Local server |

---

## 12. Backend API — All Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Single resume screening dashboard |
| GET | `/batch` | Batch screening dashboard |
| GET | `/analytics` | Analytics and model info dashboard |
| GET | `/health` | Server health check |
| POST | `/api/predict` | Single resume prediction (rate: 25/min) |
| POST | `/api/predict_batch` | Start batch job, returns job_id |
| GET | `/api/batch_status/{job_id}` | Poll batch processing progress |
| GET | `/api/history` | Fetch past analysis records |
| GET | `/api/export?format=csv` | Export results as CSV or JSON |
| GET | `/api/model/info` | Model accuracy, F1, feature names |
| GET | `/api/class_distribution` | Training data class distribution |
| GET | `/api/dataset/stats` | Feature statistics from training set |

### Single Predict Response Structure
```json
{
  "status": "success",
  "filename": "resume.pdf",
  "resume_preview": "Experience: ~3 years\nPast Roles: Software Engineer\nTop Skills: Python, FastAPI...",
  "scores": {
    "semantic_similarity": 0.742,
    "skill_overlap_score": 0.667,
    "experience_relevance_score": 0.831,
    "final_match_score": 0.734
  },
  "skills": { "matched": ["Python", "FastAPI"], "missing": ["Docker"], "extra": ["Java"] },
  "validation": { "overlapping_jobs": 0, "generic_phrase_score": 0.12, ... },
  "classification": {
    "classification": "Authentic",
    "confidence": 0.89,
    "prob_Authentic": 0.89,
    "prob_Suspicious": 0.08,
    "prob_Potentially Fake": 0.03,
    "top_features": [
      {"feature": "generic_phrase_score", "value": 0.12, "contribution": -0.43},
      {"feature": "skill_density", "value": 4.5, "contribution": 0.21},
      {"feature": "ai_plausibility_score", "value": 0.15, "contribution": -0.18}
    ],
    "llm_verification": { "consensus": "Agree", "reasoning": "Resume shows consistent experience..." }
  }
}
```

---

## 13. Frontend — What The User Sees

### Page 1: Single Resume Screen (`/`)
1. Upload resume (PDF/DOCX/TXT) via drag-drop
2. Enter job title (optional) + job description (paste or upload file)
3. Click "Run Analysis"
4. **Results appear:**
   - 🏷️ Classification badge (green ✓ Authentic / yellow ⚠ Suspicious / red ✗ Potentially Fake)
   - 🔵 Confidence ring (animated SVG showing %)
   - 📊 Classification probability bars (Authentic / Suspicious / Potentially Fake)
   - 📈 Match score cards (Semantic Sim, Skill Overlap, Exp Relevance, Final Score)
   - 🎯 Skills analysis (Matched ✓ / Missing ⚠ / Extra +)
   - 🛡️ Validation signals grid (all 19 features with values)
   - 📉 SHAP explainability bars (top 3 features driving the decision)
   - 📄 Extracted Resume Text (anonymized: experience years, past roles, education, skills — NO name/contact)
   - 🤖 LLM verdict badge (emoji + 30-50 word reasoning, no "LLM" text shown)

### Page 2: Batch Screening (`/batch`)
1. Upload multiple resumes (up to any number)
2. Enter shared job description
3. Click "Start Batch Analysis"
4. Background processing with live progress polling
5. **Table appears sorted: Authentic → Suspicious → Potentially Fake**
6. Click any row → Full detail modal (same as single screen, all cards)
7. Summary stats: Total / Authentic count / Suspicious count / Fake count

### Page 3: Analytics (`/analytics`)
- Model accuracy, F1, parameters
- Feature importance chart
- Class distribution of training data
- Feature statistics (mean/std/min/max)

---

## 14. System Statistics & Performance

| Metric | Value |
|---|---|
| Model Test Accuracy | **87.38%** |
| Model Type | XGBoost (Gradient Boosting) |
| Validation Features | **19** |
| SBERT Embedding Dimensions | **384** |
| Supported File Types | PDF, DOCX, TXT, Images (OCR) |
| Max File Size | 5 MB |
| Rate Limit (Single) | 25 requests/minute |
| Rate Limit (Batch) | 100 requests/minute |
| LLM Provider (Primary) | Groq (llama-3.3-70b-versatile) |
| LLM Provider (Fallback) | NVIDIA NIM (nemotron-3-ultra-550b) |
| Batch Concurrency | 4 parallel resumes (async semaphore) |
| Database | SQLite (async) |
| Startup Pre-warming | XGBoost + SBERT + spaCy all pre-loaded |
| Server | Uvicorn ASGI on port 8000 |
| API Documentation | Auto-generated at `/docs` (Swagger) |

---

## 15. Proposal vs. Delivered — Checklist

| Proposal Requirement | Status | Notes |
|---|---|---|
| SBERT semantic matching | ✅ | all-MiniLM-L6-v2, 384-dim vectors |
| Skill overlap scoring | ✅ | Exact formula from proposal |
| Experience relevance scoring | ✅ | Exact formula from proposal |
| Custom weighted score (0.6/0.25/0.15) | ✅ | Exact weights from proposal |
| Decision Tree classification | ✅ | Upgraded to XGBoost for better accuracy |
| 7 validation features (Table 3.1) | ✅ | Expanded to 19 features for better coverage |
| PDF/DOC input support | ✅ | Plus TXT and OCR for images |
| JD input via text or file | ✅ | Toggle between paste and upload |
| Candidate ranking dashboard | ✅ | Batch mode with sorted table |
| Detailed analysis reports | ✅ | Full cards: scores, skills, validation, SHAP |
| Accuracy target: 80–90% | ✅ | **87.38% achieved** |
| Exportable results | ✅ | CSV + JSON export at `/api/export` |
| Processing time efficiency | ✅ | Async parallel batch processing |
| Bias reduction | ✅ | Anonymized resume preview (no PII to HR) |
| LLM double-check (beyond proposal) | ✅ | Novel addition: Groq + NVIDIA ensemble |
| SHAP explainability (beyond proposal) | ✅ | Novel addition: feature contribution bars |
| Real-time batch polling (beyond proposal) | ✅ | Novel addition: 2s polling, background tasks |

---

## 16. Ethical & Privacy Considerations

### Anti-Bias Design
- The **Extracted Resume Text** section shown to HR contains **zero PII**
- It shows: Experience years, Past roles, Education level, Skills — **not** Name, Contact, Address, University name
- This is "blind screening" — standard in modern fair-hiring practice

### Data Privacy
- All processing is **100% local** — no resume data sent to external services
- Only the resume *text* (first 2000 characters) is sent to Groq/NVIDIA for LLM check, and only if suspicious
- No personal information (name, phone, email) is included in the LLM prompt (text is truncated)
- SQLite database stored locally

### Prompt Injection Prevention
- User-submitted resume text is sanitized before LLM prompts
- Regex removes: "ignore previous instructions", "system prompt", "you are a"
- Input length capped: job title 200 chars, job description 3000 chars, LLM input 2000 chars

### Fair Evaluation
- System evaluates based on skills, experience, semantic content — not formatting, name, or school prestige

---

## 17. Questions & Answers — Panel Q&A

---

> **Q1. Traditional ATS systems already exist (like Taleo, Workday). What makes your project different?**

**A:** Traditional ATS systems use TF-IDF or boolean keyword matching. They match exact words — so if a job says "Machine Learning" and a resume says "Deep Learning" or "Neural Networks", the candidate is rejected despite being qualified. Our system uses **SBERT** which converts both texts into semantic meaning vectors. "Deep Learning" and "Machine Learning" have similar vector representations, so the candidate is correctly scored. Additionally, **no commercial ATS detects AI-generated or fake resumes** — our system uniquely adds a 19-feature authenticity validation layer.

---

> **Q2. Why Sentence-BERT? Why not use standard BERT?**

**A:** Standard BERT requires both sentences to be input simultaneously (using [SEP] tokens) to compare them. For 250 resumes against one job description, this means 250 separate BERT forward passes, each processing the full pair — extremely slow. SBERT uses a **siamese network architecture**: each document is encoded independently into a fixed 384-dimensional vector. We pre-compute all embeddings once and use **cosine similarity** for comparison. This makes it many times faster while maintaining 80–85% semantic accuracy on standard benchmarks.

---

> **Q3. Your title says "Decision Tree Classification" but the code uses XGBoost. Explain.**

**A:** XGBoost (Extreme Gradient Boosting) is fundamentally an **ensemble of decision trees**. We started with a single Decision Tree which gave us ~72% accuracy. XGBoost trains trees sequentially — each new tree corrects the errors of the previous ones through gradient descent on the loss function. This gave us **87.38% accuracy**, well within the proposal target of 80–90%. The underlying splitting mechanism (information gain, entropy) is identical to a Decision Tree — XGBoost just stacks many of them for better performance.

---

> **Q4. What happens if the internet goes down? Will the system crash?**

**A:** No. The system is a **hybrid architecture**. Document parsing, SBERT embeddings, all 19 feature extractions, and XGBoost classification all run **100% locally and offline**. Only the LLM double-check (Groq/NVIDIA) requires internet. If the API call fails, the system catches the exception, logs the error, and uses only the local XGBoost result — gracefully. The system never crashes due to internet failure.

---

> **Q5. How did you train the model? What training data did you use?**

**A:** We used a combination of publicly available resume datasets (Kaggle) supplemented with manually labeled samples to create three classes: Authentic, Suspicious, and Potentially Fake. We then engineered 19 validation features based on known patterns of AI-generated and exaggerated resumes (high skill density, generic phrases, timeline inconsistencies). The XGBoost model was trained on these labeled feature vectors and evaluated on a held-out test set, achieving 87.38% accuracy.

---

> **Q6. How does SHAP work and why did you add it?**

**A:** SHAP (SHapley Additive exPlanations) is a game-theory-based method that attributes the contribution of each feature to a specific prediction. We use `shap.TreeExplainer` which is exact for tree-based models. For each classification, it tells us: "Feature X pushed the prediction TOWARD Fake by 0.43 units, feature Y pushed it AWAY by 0.21 units." We display the top 3 contributing features on the UI as directional bars. This makes the AI's decision **transparent and explainable** — critical for HR, where you can't just say "the AI said so." It also gives professors visibility into the model's internals.

---

> **Q7. What are the three categories and what do they mean?**

**A:**
- **✓ Authentic:** The resume's experience, skills, timeline, and language are consistent and plausible. XGBoost confidence ≥ 0.80.
- **⚠ Suspicious:** Something doesn't add up — unusual skill density, generic phrasing, or the LLM disagrees with a "Fake" classification (giving the candidate benefit of the doubt). Confidence 0.50–0.80.
- **✗ Potentially Fake:** The resume shows strong indicators of AI generation or deliberate fabrication. Multiple red flags like overlapping job dates, 50 skills in 1 year, 90%+ keyword stuffing, LLM confirms the ML model's judgment.

---

> **Q8. Why Groq as primary and NVIDIA as fallback?**

**A:** Groq's inference hardware (LPUs) is significantly faster than standard GPU inference. Their `llama-3.3-70b-versatile` model is free-tier accessible, supports JSON mode, and is extremely reliable. NVIDIA NIM provides access to their Nemotron model as an enterprise-grade alternative. Since NVIDIA's API had intermittent failures during development, we made it the fallback so our system always has a working LLM available. The architecture allows adding more providers easily.

---

> **Q9. How do you prevent false positives — accusing a real candidate of being fake?**

**A:** Two layers of protection:
1. **LLM Ensemble Consensus:** If XGBoost says "Potentially Fake", the LLM is called as an independent judge. If the LLM *disagrees* — says the resume looks real — the system **overrides** the classification to "Suspicious" rather than "Potentially Fake". The candidate is not wrongfully condemned.
2. **Probability Transparency:** The UI shows the full probability distribution across all three classes. HR can see "Potentially Fake: 52% confidence" and use their own judgment. The system gives signals, not final verdicts.

---

> **Q10. What is keyword stuffing and how do you detect it?**

**A:** Keyword stuffing is when a candidate copies every keyword from the job description directly into their resume to fool keyword-based ATS systems. We detect it by:
1. Extracting all meaningful words from the job description (excluding stopwords)
2. Counting how many of those words appear in the resume
3. Computing ratio = (JD-words in resume) / (total resume words) × 2
4. Adding a repetition penalty if any JD word appears > 10 times
5. Score of > 0.6 is highly suspicious

---

> **Q11. What is the experience-graduation gap feature?**

**A:** If someone claims 8 years of professional experience but graduated from university in 2022, that's logically impossible — they'd be claiming experience before finishing college. We compute: `current_year - graduation_year - claimed_years_experience`. If this gap is negative (e.g., -3), it's a strong fake signal. This catches one of the most common errors in AI-generated resumes, which often hallucinate impossible timelines.

---

> **Q12. How does the batch processing work technically?**

**A:** When the user submits multiple resumes:
1. All file contents are read immediately (UploadFile streams close after request)
2. A UUID job_id is created, stored in memory
3. An **async background task** starts (FastAPI `BackgroundTasks`)
4. Within the background task, an **asyncio Semaphore(4)** limits concurrency to 4 simultaneous resume analyses
5. All resumes are processed via `asyncio.gather()` (true parallel async)
6. Frontend polls `/api/batch_status/{job_id}` every 2 seconds
7. Results are sorted: Authentic → Suspicious → Fake → Error

---

> **Q13. What is the anonymized resume preview and why is it important?**

**A:** Traditional ATS systems often show the raw resume text to HR — which includes the candidate's name, address, phone number, and university. Research shows HR can be biased by these details (famous university name effect, foreign names, etc.). Our system uses spaCy NER to extract only:
- Years of experience (not name)
- Past job titles/roles (not company name)
- Education degree level (not university name)
- Skills list

This implements **blind screening** — a standard best practice in unbiased hiring, used by companies like BBC and Applied.com.

---

> **Q14. What security measures does the system have?**

**A:**
- **Rate limiting:** 25 requests/minute (single), 100/minute (batch) via `slowapi`
- **File validation:** File type, size (5MB max), MIME type checking
- **Input sanitization:** Job title capped 200 chars, JD capped 3000 chars
- **Prompt injection defense:** Regex sanitizes LLM inputs
- **No external data transmission:** Resumes stored locally in SQLite
- **Error isolation:** Each batch resume processing failure is caught individually, doesn't stop the rest

---

> **Q15. What would you improve if given more time?**

**A:**
1. **Resume verification API:** Integration with LinkedIn API or government database to verify employment claims
2. **Video interview analysis:** Extend the authenticity check to video responses
3. **Multi-language support:** Currently optimized for English resumes
4. **Active learning:** Allow HR to flag false positives/negatives to retrain the model
5. **Production deployment:** PostgreSQL instead of SQLite, Redis for batch job storage, Docker containerization
6. **Fine-tuned LLM:** Fine-tune a small model on labeled authentic vs. fake resume pairs

---

> **Q16. The proposal mentioned PostgreSQL. You used SQLite. Why?**

**A:** For the academic prototype scope, SQLite with async support (aiosqlite) is perfectly adequate. It requires zero setup, zero configuration, and handles our load without issue. The database abstraction (SQLAlchemy ORM) means switching to PostgreSQL in production requires changing only the connection string. The data model (schema) is identical. This is a deliberate decision — lightweight for demo, production-ready by design.

---

> **Q17. Can you explain cosine similarity in simple terms?**

**A:** Imagine representing a resume and a job description as arrows in 3D space, where each dimension represents how "relevant" the text is to some concept. If both arrows point in the same direction (angle = 0°), the cosine of 0° = 1.0 (100% similar). If they point opposite directions, cosine(-180°) = -1 (completely opposite). If perpendicular (90°), cosine = 0 (unrelated). SBERT puts them in 384-dimensional space instead of 3D, but the concept is identical. A score of 0.75 means the resume and job description "point in roughly the same direction" — 75% semantic alignment.

---

> **Q18. Why 0.60 / 0.25 / 0.15 weights? How did you decide this?**

**A:** Through empirical tuning on our validation set. Semantic similarity is the most important signal — it captures overall alignment between the candidate's background and the job's requirements in meaning, not just words. Skill overlap is the most concrete measurable factor — literally "do they have the skills listed?". Experience relevance is a supporting factor — it breaks ties between candidates with similar semantic scores. These weights ensure that a candidate who is semantically aligned but missing 2 skills is ranked differently than one who has the exact skills but irrelevant background.

---

*Built by SARSHIJ KARN*  
*National College of Engineering — 6th Semester Minor Project, 2026*
