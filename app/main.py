"""
FastAPI Backend for Resume Screening & Authenticity Validation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
import json, os, io, time
import pandas as pd
import numpy as np

from app.logger import setup_logger
from app.utils.parser import parse_resume
from app.features.semantic import compute_semantic_similarity
from app.features.skill_overlap import compute_skill_overlap, get_matched_skills, extract_skills
from app.features.experience import score_experience_relevance
from app.features.experience_extraction import extract_years_experience, extract_graduation_year
from app.features.validation import compute_all_validation_features
from app.models.classifier import predict, get_model_info

logger = setup_logger(__name__)
BASE = Path(__file__).resolve().parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Resume Screener API starting up...")
    yield
    logger.info("Resume Screener API shutting down.")

app = FastAPI(title="Resume Screener API", version="1.0.0", lifespan=lifespan)

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

@app.post("/api/predict")
async def predict_single(
    resume: UploadFile = File(...),
    job_title: str = Form(""),
    job_description: str = Form("")
):
    try:
        resume_bytes = await resume.read()
        resume_text = parse_resume(resume_bytes, resume.filename or "resume.pdf")
        if not resume_text or len(resume_text.strip()) < 20:
            logger.warning(f"Too short or empty: {resume.filename} ({len(resume_text or '')} chars)")
            raise HTTPException(400, "Could not extract enough text from resume")

        logger.info(f"Predicting: {resume.filename} | JD: {len(job_description)} chars")
        years_exp = extract_years_experience(resume_text)
        grad_year = extract_graduation_year(resume_text)
        logger.debug(f"Extracted: {years_exp} years exp, graduation year {grad_year}")
        sem_sim = compute_semantic_similarity(resume_text, job_description)
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
        skill_details = get_matched_skills(resume_text, job_description)

        logger.info(f"Result: {resume.filename} -> {classification['classification']} ({classification['confidence']*100:.1f}%)")
        return {
            "status": "success",
            "filename": resume.filename,
            "resume_preview": resume_text[:500] + ("..." if len(resume_text) > 500 else ""),
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Predict failed for {resume.filename}: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/predict_batch")
async def predict_batch(
    resumes: list[UploadFile] = File(...),
    job_title: str = Form(""),
    job_description: str = Form("")
):
    logger.info(f"Batch predict: {len(resumes)} files")
    results = []
    for resume in resumes:
        try:
            resume_bytes = await resume.read()
            resume_text = parse_resume(resume_bytes, resume.filename or "resume.pdf")
            if not resume_text or len(resume_text.strip()) < 20:
                logger.warning(f"Batch: {resume.filename} — too short")
                results.append({"filename": resume.filename, "error": "Could not extract text"})
                continue
            years_exp = extract_years_experience(resume_text)
            grad_year = extract_graduation_year(resume_text)
            sem_sim = compute_semantic_similarity(resume_text, job_description)
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
            results.append({
                "filename": resume.filename,
                "classification": classification["classification"],
                "confidence": classification["confidence"],
                "final_match_score": final_score,
                "semantic_similarity": sem_sim,
                "skill_overlap_score": skill_overlap
            })
            logger.debug(f"Batch: {resume.filename} -> {classification['classification']}")
        except Exception as e:
            logger.error(f"Batch failed for {resume.filename}: {e}")
            results.append({"filename": resume.filename, "error": str(e)})
    sorted_results = sorted(results, key=lambda x: (
        0 if x.get("classification") == "Authentic"
        else 1 if x.get("classification") == "Suspicious"
        else 2 if x.get("classification") == "Potentially Fake"
        else 3,
        -x.get("final_match_score", 0)
    ))
    logger.info(f"Batch complete: {len(results)} results ({sum(1 for r in results if 'error' not in r)} ok)")
    return {"status": "success", "count": len(results), "results": sorted_results}

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
