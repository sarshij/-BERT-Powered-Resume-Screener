"""
FastAPI Backend for Resume Screening & Authenticity Validation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import json, os, io, time
import pandas as pd
import numpy as np

from app.logger import setup_logger
from app.utils.parser import parse_resume
from app.utils.file_validator import validate_upload
from app.features.semantic import compute_semantic_similarity, compute_semantic_similarity_async
from app.features.skill_overlap import compute_skill_overlap, get_matched_skills, extract_skills
from app.features.experience import score_experience_relevance
from app.utils.nlp import extract_education_spacy, extract_job_titles_spacy
from app.features.experience_extraction import extract_years_experience, extract_graduation_year
from app.features.validation import compute_all_validation_features
from app.models.classifier import predict, get_model_info
from app.models.database import init_db

logger = setup_logger(__name__)
BASE = Path(__file__).resolve().parent.parent

# Set up Rate Limiter
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Resume Screener API starting up...")
    
    # ── Database Initialization ─────────────────────────────────────────────
    await init_db()
    
    # ── Fix 10: Startup model pre-warming ────────────────────────────────────
    # Load model
    from app.models.classifier import load_model
    load_model()
    # Pre-warm spaCy pipeline
    from app.utils.nlp import get_nlp_with_ruler, _ensure_patterns
    nlp = get_nlp_with_ruler()
    if nlp:
        _ensure_patterns(nlp)
        logger.info("spaCy pipeline pre-warmed successfully")
    # Pre-warm SBERT model
    from app.models.embedder import get_model as get_sbert_model
    get_sbert_model()
    logger.info("SBERT model pre-warmed successfully")
    
    yield
    logger.info("Resume Screener API shutting down.")

app = FastAPI(title="Resume Screener API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

templates = Jinja2Templates(directory=str(BASE / 'app' / 'templates'))
app.mount("/static", StaticFiles(directory=str(BASE / 'app' / 'static')), name="static")

class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration:.2f}s)")
        return response

app.add_middleware(RequestLogMiddleware)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    return templates.TemplateResponse(request=request, name="batch.html", context={"request": request})

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    return templates.TemplateResponse(request=request, name="analytics.html", context={"request": request})

@app.get("/health")
async def health():
    """
    Fix 8: Health endpoint for monitoring.
    """
    from app.models.classifier import _loaded_model
    return {
        "status": "ok", 
        "version": "2.0",
        "model_loaded": _loaded_model is not None
    }

@app.post("/api/predict")
@limiter.limit("25/minute")
async def predict_single(
    request: Request,
    resume: UploadFile = File(...),
    job_title: str = Form(""),
    job_description: str = Form(""),
    job_description_file: UploadFile = File(None)
):
    try:
        # Parse JD from file if provided
        if job_description_file and job_description_file.filename:
            jd_bytes = await job_description_file.read()
            if jd_bytes:
                validate_upload(jd_bytes, job_description_file.filename)
                job_description = parse_resume(jd_bytes, job_description_file.filename)

        # ── Fix 6: Input length sanitization ─────────────────────────────────
        job_title = job_title.strip()[:200]
        job_description = job_description.strip()[:3000]
        
        if not job_description:
            raise HTTPException(400, "Job description must be provided via text or file upload")

        resume_bytes = await resume.read()
        
        # ── Fix 2 & 3: File type and size validation ─────────────────────────
        validate_upload(resume_bytes, resume.filename or "resume.pdf")

        resume_text = parse_resume(resume_bytes, resume.filename or "resume.pdf")
        if not resume_text or len(resume_text.strip()) < 20:
            logger.warning(f"Too short or empty: {resume.filename} ({len(resume_text or '')} chars)")
            raise HTTPException(400, "Could not extract enough text from resume")

        logger.info(f"🚀 \033[1;35mPredicting: {resume.filename} | JD: {len(job_description)} chars\033[0m")
        years_exp = extract_years_experience(resume_text)
        grad_year = extract_graduation_year(resume_text)
        logger.debug(f"Extracted: {years_exp} years exp, graduation year {grad_year}")
        sem_sim = await compute_semantic_similarity_async(resume_text, job_description)
        skill_overlap = compute_skill_overlap(resume_text, job_description)
        exp_relevance = score_experience_relevance(resume_text, job_title or job_description)
        final_score = round(0.6 * sem_sim + 0.25 * skill_overlap + 0.15 * exp_relevance, 4)
        extracted_skills = list(extract_skills(resume_text))
        validation = compute_all_validation_features(
            resume_text, job_description,
            semantic_similarity=sem_sim,
            skill_overlap_score=skill_overlap,
            experience_relevance_score=exp_relevance,
            final_match_score=final_score,
            years_experience=years_exp,
            graduation_year=grad_year,
            extracted_skills=extracted_skills
        )
        classification = predict([validation])[0]
        
        # Double check with LLM if Suspicious or Fake
        current_class = classification.get('classification', 'Unknown')
        if current_class in ['Suspicious', 'Potentially Fake']:
            from app.models.llm_detector import get_llm_detector
            detector = get_llm_detector()
            verification = detector.verify_prediction(resume_text, job_description, current_class)
            if verification:
                classification['llm_verification'] = verification
                if verification.get('consensus') == 'Disagree':
                    classification['classification'] = 'Suspicious'

        skill_details = get_matched_skills(resume_text, job_description)

        # Generate a cleaner, anonymized summary without PII
        edu_list = list(extract_education_spacy(resume_text))
        edu_str = ", ".join([e.title() for e in edu_list[:3]]) + ("..." if len(edu_list) > 3 else "")
        if not edu_str: edu_str = "Not detected"

        job_list = list(extract_job_titles_spacy(resume_text))
        job_str = ", ".join([j.title() for j in job_list[:3]]) + ("..." if len(job_list) > 3 else "")
        if not job_str: job_str = "Not detected"

        skill_str = ", ".join(extracted_skills[:10]) + ("..." if len(extracted_skills) > 10 else "")
        
        summary_preview = (
            f"Experience: ~{years_exp} years\n"
            f"Past Roles: {job_str}\n"
            f"Education: {edu_str} (Class of {grad_year})\n\n"
            f"Top Skills: {skill_str}"
        )

        result_data = {
            "status": "success",
            "filename": resume.filename,
            "resume_preview": summary_preview,
            "scores": {
                "semantic_similarity": sem_sim,
                "skill_overlap_score": skill_overlap,
                "experience_relevance_score": exp_relevance,
                "final_match_score": final_score
            },
            "skills": skill_details,
            "validation": validation,
            "classification": classification
        }
        
        # Async save to DB
        try:
            from app.models.database import async_session, ResumeAnalysis, JobDescription
            async with async_session() as session:
                # Basic saving logic without parsing job title/desc IDs for now (can expand later)
                db_analysis = ResumeAnalysis(
                    filename=resume.filename,
                    final_match_score=final_score,
                    ai_plausibility_score=validation.get('ai_plausibility_score', 0.5),
                    classification=classification['classification'],
                    full_results=result_data
                )
                session.add(db_analysis)
                await session.commit()
        except Exception as db_e:
            logger.error(f"Failed to save to database: {db_e}")

        logger.info(f"✅ \033[1;32mResult: {resume.filename} -> {classification['classification']} ({classification['confidence']*100:.1f}%)\033[0m")
        return result_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Predict failed for {resume.filename}: {e}")
        raise HTTPException(500, str(e))

# ── Fix 15: Batch endpoint parallelism & Polling ───────────────────────────────
# Dictionary to store batch job statuses in memory (for production, use Redis/DB)
batch_jobs = {}

@app.post("/api/predict_batch")
@limiter.limit("100/minute")
async def predict_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    resumes: list[UploadFile] = File(...),
    job_title: str = Form(""),
    job_description: str = Form(""),
    job_description_file: UploadFile = File(None)
):
    import asyncio
    import uuid
    
    # Parse JD from file if provided
    if job_description_file and job_description_file.filename:
        jd_bytes = await job_description_file.read()
        if jd_bytes:
            validate_upload(jd_bytes, job_description_file.filename)
            job_description = parse_resume(jd_bytes, job_description_file.filename)

    job_title = job_title.strip()[:200]
    job_description = job_description.strip()[:3000]
    
    if not job_description:
        raise HTTPException(400, "Job description must be provided via text or file upload")

    job_id = str(uuid.uuid4())
    batch_jobs[job_id] = {
        "status": "processing",
        "total": len(resumes),
        "completed": 0,
        "results": [],
        "errors": []
    }
    
    # We must read the file contents now because UploadFile stream closes after the request ends
    resume_data = []
    for resume in resumes:
        try:
            content = await resume.read()
            resume_data.append((resume.filename, content))
        except Exception as e:
            batch_jobs[job_id]["errors"].append(f"Failed to read {resume.filename}: {e}")
            batch_jobs[job_id]["completed"] += 1

    logger.info(f"Batch predict started: {len(resume_data)} files, Job ID: {job_id}")
    
    # Function to process in background
    async def process_batch_background(job_id: str, resume_data: list, job_title: str, job_description: str):
        semaphore = asyncio.Semaphore(4) # Limit concurrency
        
        async def process_single(filename: str, resume_bytes: bytes) -> dict:
            async with semaphore:
                try:
                    try:
                        validate_upload(resume_bytes, filename or "resume.pdf")
                    except HTTPException as ve:
                        return {"filename": filename, "error": ve.detail}
                        
                    resume_text = parse_resume(resume_bytes, filename or "resume.pdf")
                    if not resume_text or len(resume_text.strip()) < 20:
                        return {"filename": filename, "error": "Could not extract text"}
                        
                    years_exp = extract_years_experience(resume_text)
                    grad_year = extract_graduation_year(resume_text)
                    sem_sim = await compute_semantic_similarity_async(resume_text, job_description)
                    skill_overlap = compute_skill_overlap(resume_text, job_description)
                    exp_relevance = score_experience_relevance(resume_text, job_title or job_description)
                    final_score = round(0.6 * sem_sim + 0.25 * skill_overlap + 0.15 * exp_relevance, 4)
                    extracted_skills = list(extract_skills(resume_text))
                    validation = compute_all_validation_features(
                        resume_text, job_description,
                        semantic_similarity=sem_sim,
                        skill_overlap_score=skill_overlap,
                        experience_relevance_score=exp_relevance,
                        final_match_score=final_score,
                        years_experience=years_exp,
                        graduation_year=grad_year,
                        extracted_skills=extracted_skills
                    )
                    classification = predict([validation])[0]
                    
                    # Double check with LLM if Suspicious or Fake
                    current_class = classification.get('classification', 'Unknown')
                    if current_class in ['Suspicious', 'Potentially Fake']:
                        from app.models.llm_detector import get_llm_detector
                        detector = get_llm_detector()
                        verification = detector.verify_prediction(resume_text, job_description, current_class)
                        if verification:
                            classification['llm_verification'] = verification
                            if verification.get('consensus') == 'Disagree':
                                classification['classification'] = 'Suspicious'
                    
                    skill_details = get_matched_skills(resume_text, job_description)
                    # Generate a cleaner, anonymized summary without PII
                    edu_list = list(extract_education_spacy(resume_text))
                    edu_str = ", ".join([e.title() for e in edu_list[:3]]) + ("..." if len(edu_list) > 3 else "")
                    if not edu_str: edu_str = "Not detected"

                    job_list = list(extract_job_titles_spacy(resume_text))
                    job_str = ", ".join([j.title() for j in job_list[:3]]) + ("..." if len(job_list) > 3 else "")
                    if not job_str: job_str = "Not detected"

                    skill_str = ", ".join(extracted_skills[:10]) + ("..." if len(extracted_skills) > 10 else "")
                    
                    summary_preview = (
                        f"Experience: ~{years_exp} years\n"
                        f"Past Roles: {job_str}\n"
                        f"Education: {edu_str} (Class of {grad_year})\n\n"
                        f"Top Skills: {skill_str}"
                    )
                    
                    return {
                        "status": "success",
                        "filename": filename,
                        "resume_preview": summary_preview,
                        "scores": {
                            "semantic_similarity": sem_sim,
                            "skill_overlap_score": skill_overlap,
                            "experience_relevance_score": exp_relevance,
                            "final_match_score": final_score
                        },
                        "skills": skill_details,
                        "validation": validation,
                        "classification": classification["classification"],
                        "classification_details": classification,
                        "confidence": classification["confidence"],
                        "final_match_score": final_score,
                        "semantic_similarity": sem_sim,
                        "skill_overlap_score": skill_overlap
                    }
                except Exception as e:
                    return {"filename": filename, "error": str(e)}
                finally:
                    batch_jobs[job_id]["completed"] += 1

        tasks = [process_single(fname, content) for fname, content in resume_data]
        results = await asyncio.gather(*tasks)
        
        sorted_results = sorted(results, key=lambda x: (
            0 if x.get("classification") == "Authentic"
            else 1 if x.get("classification") == "Suspicious"
            else 2 if x.get("classification") == "Potentially Fake"
            else 3,
            -x.get("final_match_score", 0)
        ))
        batch_jobs[job_id]["results"] = sorted_results
        batch_jobs[job_id]["status"] = "completed"
        logger.info(f"Batch {job_id} complete")

    if background_tasks:
        background_tasks.add_task(process_batch_background, job_id, resume_data, job_title, job_description)
    else:
        # Fallback if background_tasks is somehow None (e.g. testing)
        asyncio.create_task(process_batch_background(job_id, resume_data, job_title, job_description))

    return {"status": "processing", "job_id": job_id, "message": "Batch processing started."}

@app.get("/api/batch_status/{job_id}")
async def batch_status(job_id: str):
    if job_id not in batch_jobs:
        raise HTTPException(404, "Job ID not found")
    
    job = batch_jobs[job_id]
    return {
        "status": job["status"],
        "total": job["total"],
        "completed": job["completed"],
        "progress": round((job["completed"] / job["total"]) * 100) if job["total"] > 0 else 0,
        "results": job["results"] if job["status"] == "completed" else []
    }

@app.get("/api/history")
async def get_history(limit: int = 50, offset: int = 0):
    try:
        from app.models.database import async_session, ResumeAnalysis
        from sqlalchemy import select
        async with async_session() as session:
            stmt = select(ResumeAnalysis).order_by(ResumeAnalysis.created_at.desc()).limit(limit).offset(offset)
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            history = []
            for r in records:
                history.append({
                    "id": r.id,
                    "filename": r.filename,
                    "candidate_name": r.candidate_name,
                    "classification": r.classification,
                    "final_match_score": r.final_match_score,
                    "ai_plausibility_score": r.ai_plausibility_score,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                })
            return {"status": "success", "history": history}
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(500, "Could not fetch history")

@app.get("/api/export")
async def export_data(format: str = 'csv'):
    try:
        from app.models.database import async_session, ResumeAnalysis
        from sqlalchemy import select
        async with async_session() as session:
            stmt = select(ResumeAnalysis).order_by(ResumeAnalysis.created_at.desc())
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            data = []
            for r in records:
                data.append({
                    "id": r.id,
                    "filename": r.filename,
                    "classification": r.classification,
                    "final_match_score": r.final_match_score,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                })
                
            if format == 'json':
                return JSONResponse(content={"status": "success", "data": data})
            else:
                # Basic CSV generation
                import csv, io
                from fastapi.responses import StreamingResponse
                
                stream = io.StringIO()
                writer = csv.DictWriter(stream, fieldnames=["id", "filename", "classification", "final_match_score", "created_at"])
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
                    
                response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
                response.headers["Content-Disposition"] = "attachment; filename=resume_analysis_export.csv"
                return response
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        raise HTTPException(500, "Could not export data")

@app.get("/api/model/info")
async def model_info():
    try:
        info = get_model_info()
        return {"status": "success", **info}
    except Exception as e:
        logger.error(f"Model info failed: {e}")
        raise HTTPException(500, str(e))

@app.get("/api/class_distribution")
async def class_distribution():
    df_path = BASE / 'data' / 'processed' / 'combined_dataset.csv'
    if not df_path.exists():
        raise HTTPException(404, "Dataset not found")
    df = pd.read_csv(df_path)
    dist = df['classification'].value_counts().to_dict()
    risk = df['risk_level'].value_counts().to_dict()
    return {"status": "success", "class_distribution": dist, "risk_distribution": risk}

@app.get("/api/dataset/stats")
async def dataset_stats():
    df_path = BASE / 'data' / 'processed' / 'combined_dataset.csv'
    if not df_path.exists():
        raise HTTPException(404, "Dataset not found")
    df = pd.read_csv(df_path)
    feature_cols = [
        'semantic_similarity', 'skill_overlap_score', 'experience_relevance_score',
        'final_match_score', 'skill_density', 'achievement_count',
        'generic_phrase_score', 'gap_years', 'keyword_stuffing_score',
        'years_experience', 'num_certifications', 'num_skills',
        'education_level_encoded', 'has_previous_job',
    ]
    stats = {}
    for col in feature_cols:
        if col in df.columns:
            stats[col] = {
                'mean': round(float(df[col].mean()), 4),
                'std': round(float(df[col].std()), 4),
                'min': round(float(df[col].min()), 4),
                'max': round(float(df[col].max()), 4)
            }
    return {"status": "success", "total_samples": len(df), "feature_stats": stats}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
