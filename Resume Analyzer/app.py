"""
app.py — AI Resume Analyzer API + Frontend

Run with:
    uvicorn app:app --reload

Endpoints:
    GET  /          → Serves the frontend UI
    GET  /health    → Liveness check
    POST /analyze   → Multi-dimensional resume analysis
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .model import ResumeAnalyzer
    from .utils import extract_text_from_pdf
except ImportError:
    from model import ResumeAnalyzer
    from utils import extract_text_from_pdf


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Resume Analyzer",
    version="4.0.0",
    description=(
        "Upload a PDF resume and paste a job description. "
        "Returns a multi-dimensional analysis with an overall score (0-100), "
        "letter grade, dimension scores, missing keywords, and suggestions."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_analyzer = ResumeAnalyzer(
    max_features=8000,
    top_keywords=10,
    min_keyword_len=3,
)

_MAX_FILE_SIZE_MB = 10
_MAX_FILE_SIZE_BYTES = _MAX_FILE_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Frontend"], include_in_schema=False)
def serve_frontend():
    """Serve the frontend HTML page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index_path), media_type="text/html")


@app.get("/health", tags=["Utility"])
def health() -> dict[str, str]:
    """Quick liveness check."""
    return {"status": "ok", "version": "4.0.0"}


@app.post(
    "/analyze",
    tags=["Analysis"],
    response_description="Multi-dimensional resume analysis",
    responses={
        200: {
            "description": "Successful analysis",
            "content": {
                "application/json": {
                    "example": {
                        "overall_score": 72.45,
                        "grade": "B",
                        "dimension_scores": {
                            "keyword_match": 65.3,
                            "section_structure": 80.0,
                            "impact_metrics": 55.0,
                            "skills_depth": 70.5,
                            "formatting": 85.0,
                            "action_verbs": 60.0,
                        },
                        "keywords_missing": ["kubernetes", "terraform"],
                        "suggestions": ["Add more quantifiable achievements"],
                        "strengths": ["Strong technical skills alignment"],
                        "section_analysis": {
                            "contact": True, "education": True,
                            "experience": True, "skills": True,
                        },
                        "contact_info": {
                            "email": True, "phone": True,
                            "linkedin": False, "github": True,
                        },
                    }
                }
            },
        },
        400: {"description": "Bad request"},
        500: {"description": "Unexpected server error"},
    },
)
async def analyze_resume(
    resume: UploadFile = File(
        ..., description="PDF resume file (text-based, not scanned image)"
    ),
    job_desc: str = Form(
        ..., description="Full job description text to compare the resume against"
    ),
) -> dict[str, Any]:
    """
    Analyze how well a resume matches a job description across 6 dimensions.

    Returns overall_score, grade, dimension_scores, keywords_missing,
    suggestions, strengths, section_analysis, and contact_info.
    """

    # ── input validation ──────────────────────────────────────────────────
    if not resume.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported. Got: '{resume.filename}'",
        )

    if not job_desc or not job_desc.strip():
        raise HTTPException(
            status_code=400, detail="Job description cannot be empty."
        )

    if len(job_desc.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Job description is too short. Provide at least a sentence.",
        )

    # ── read file ─────────────────────────────────────────────────────────
    try:
        file_bytes = await resume.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to read uploaded file: {exc}"
        ) from exc
    finally:
        await resume.close()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    if len(file_bytes) > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {_MAX_FILE_SIZE_MB} MB.",
        )

    # ── extract text from PDF ─────────────────────────────────────────────
    try:
        resume_text = extract_text_from_pdf(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error reading PDF: {exc}"
        ) from exc

    if not resume_text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No text could be extracted from the PDF. "
                "Make sure the file is a text-based PDF, not a scanned image."
            ),
        )

    # ── run ML analysis ───────────────────────────────────────────────────
    try:
        result = _analyzer.analyze(
            resume_text=resume_text,
            job_description=job_desc,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed unexpectedly: {exc}",
        ) from exc

    return result
