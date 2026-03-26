# AI Resume Analyzer — Backend

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

API runs at: http://127.0.0.1:8000  
Docs at:     http://127.0.0.1:8000/docs

## Endpoints

| Method | Path      | Description                        |
|--------|-----------|------------------------------------|
| GET    | /health   | Liveness check                     |
| POST   | /analyze  | Analyze resume vs job description  |

## POST /analyze

**Form data:**
- `resume` — PDF file (text-based, not scanned image)
- `job_desc` — Job description as plain text

**Response:**
```json
{
  "match_score": 72.45,
  "keywords_missing": ["kubernetes", "terraform", "grafana"]
}
```

## Bug Fixes (v3.0.0)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `match_score = 0` | TF-IDF fitted on one doc, no shared vocabulary | Fit vectorizer on BOTH docs together |
| Bad keywords ("string") | No quality filter on extracted tokens | Added `isalpha()` + min-length filter |
| PDF extraction failures | PyPDF2 didn't handle None page text | pdfplumber with per-page None guard |
| Smart quotes breaking tokenisation | PDF unicode not normalised | NFKD unicode normalisation before cleaning |
