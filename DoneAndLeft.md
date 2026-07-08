# Project Status Tracking

Based on the Minor Project Proposal: "SBERT-Based Resume Screening and Authenticity Validation Using Decision Tree Classification", here is the current status of the implementation:

## 🟢 Completed (Done)

1. **Text Extraction & NLP Preprocessing**
   - Implemented PDF and DOCX parsers (`PyMuPDF`, `python-docx`).
   - NLP tokenization, lemmatization, and NER integration using `spaCy` (upgraded to `en_core_web_md`).
   - Alias normalization for tech stack skills (e.g., mapping `js` to `JavaScript`).
2. **SBERT Embedding Generation**
   - Integration of `SentenceTransformer('all-MiniLM-L6-v2')` for generating 384-dimensional semantic embeddings.
   - Embeddings logic made asynchronous using ThreadPoolExecutor to prevent blocking.
3. **Custom Weighted Scoring Model**
   - Computes Semantic Similarity (Cosine Similarity).
   - Computes Skill Overlap percentage.
   - Computes Experience Relevance.
   - Aggregates features via the weighted formula: 60% Semantic + 25% Skill + 15% Experience.
4. **Validation Feature Extraction (Rule-Based)**
   - Extracts 17 distinct validation features including timeline consistency (overlapping jobs, gap years), achievement density, skill density, and generic AI phrase frequency.
5. **Decision Tree Classification (Authenticity Module)**
   - Model trained, evaluated (achieved ~83.6% accuracy, meeting the 80-90% goal), and integrated into the pipeline to flag resumes as Authentic, Suspicious, or Potentially Fake.
6. **Backend Infrastructure & APIs**
   - FastAPI server setup with `/api/predict` (single) and `/api/predict_batch` (bulk) endpoints.
   - Security constraints (MIME type verification and 5MB size limit).
   - `/health` endpoint for monitoring model readiness.
7. **Frontend Dashboard**
   - Interactive HTML/JS dashboard with candidate ranking, skill gap reports, and risk level badges.
8. **Job Description File Upload Support**
   - Input job description via text field (paste) or file upload (PDF/DOCX/TXT) with dynamic JS toggling on the dashboard.
9. **Advanced Skill-Experience Alignment**
   - Verification that claimed skill levels are supported by corresponding project work and employment history.
10. **Modern UI Overhaul**
    - Redesigned the entire frontend with a centralized CSS design system.
    - Implemented a responsive, glassmorphism-inspired UI with SVGs, animations, and motion graphics suitable for a minor project presentation.

## 🔴 Left To Be Done

- **Exportable CSV/Excel Reports**: The proposal mentions exportable evaluation results. Adding a "Download CSV" button to the batch screening results would fulfill this.
- **Model Metric Export**: Allow exporting the Analytics dashboard metrics (accuracy, precision, recall, F1-scores) to a file.

## 🟡 Optional Enhancements
- Advanced error handling for very badly formatted or scanned (image-based) PDFs (requires OCR).
- Real-time certificate or employment verification using external APIs (noted as out of scope in the proposal, but good for future work).
