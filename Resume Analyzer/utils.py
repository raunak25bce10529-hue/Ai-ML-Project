"""
utils.py — PDF extraction + text preprocessing + resume analysis helpers.

Provides:
  • extract_text_from_pdf  — pdfplumber-based PDF text extraction
  • preprocess_text        — tokenise / clean / remove stopwords
  • detect_sections        — find which standard resume sections exist
  • ACTION_VERBS           — strong resume action verbs
  • COMMON_SKILLS          — common tech & business skills
  • REQUIRED_SECTIONS      — expected resume section headings
"""
from __future__ import annotations

import re
import unicodedata
from io import BytesIO

import pdfplumber
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOP_WORDS: frozenset[str] = frozenset(ENGLISH_STOP_WORDS)

_KEEP_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_MIN_TOKEN_LEN = 2

# ---------------------------------------------------------------------------
# Resume section detection
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS: dict[str, list[str]] = {
    "contact":    ["contact", "email", "phone", "address", "linkedin", "github", "portfolio"],
    "education":  ["education", "academic", "qualification", "university", "college", "degree", "bachelor", "master", "phd"],
    "experience": ["experience", "work experience", "employment", "professional experience", "work history", "internship"],
    "skills":     ["skills", "technical skills", "technologies", "tools", "competencies", "proficiencies", "tech stack"],
    "projects":   ["projects", "personal projects", "academic projects", "portfolio", "side projects"],
    "summary":    ["summary", "objective", "about me", "profile", "professional summary", "career objective"],
    "certifications": ["certifications", "certificates", "courses", "training", "licenses"],
    "achievements":   ["achievements", "awards", "honors", "accomplishments", "recognition"],
}

# ---------------------------------------------------------------------------
# Action verbs — strong resume language
# ---------------------------------------------------------------------------

ACTION_VERBS: frozenset[str] = frozenset({
    # Leadership / Management
    "led", "managed", "directed", "supervised", "coordinated", "oversaw",
    "spearheaded", "mentored", "guided", "headed", "orchestrated",
    # Creation / Development
    "developed", "created", "designed", "built", "engineered", "architected",
    "implemented", "launched", "established", "founded", "initiated",
    "constructed", "formulated", "devised", "invented", "pioneered",
    # Improvement / Optimisation
    "improved", "enhanced", "optimized", "streamlined", "refactored",
    "revamped", "modernized", "upgraded", "accelerated", "boosted",
    "maximized", "strengthened", "transformed", "elevated",
    # Analysis / Research
    "analyzed", "evaluated", "assessed", "investigated", "researched",
    "examined", "identified", "diagnosed", "audited", "benchmarked",
    "surveyed", "explored", "measured", "quantified",
    # Communication / Collaboration
    "presented", "communicated", "collaborated", "negotiated", "facilitated",
    "advocated", "consulted", "influenced", "persuaded", "trained",
    "educated", "delivered", "reported", "documented", "published",
    # Achievement / Impact
    "achieved", "increased", "reduced", "decreased", "saved", "generated",
    "exceeded", "surpassed", "earned", "secured", "won", "delivered",
    "resolved", "solved", "eliminated", "prevented", "recovered",
    # Technical
    "programmed", "coded", "automated", "deployed", "integrated",
    "configured", "migrated", "debugged", "tested", "monitored",
    "maintained", "administered", "provisioned", "containerized",
    "scaled", "refactored", "compiled", "scripted",
})

# ---------------------------------------------------------------------------
# Common skills — tech & business
# ---------------------------------------------------------------------------

COMMON_SKILLS: frozenset[str] = frozenset({
    # Programming languages
    "python", "java", "javascript", "typescript", "csharp", "cpp", "golang",
    "ruby", "swift", "kotlin", "rust", "php", "scala", "perl", "matlab",
    "sql", "html", "css", "sass", "bash", "powershell", "dart", "lua",
    # Frameworks & libraries
    "react", "angular", "vue", "nextjs", "django", "flask", "fastapi",
    "spring", "express", "nodejs", "rails", "laravel", "flutter",
    "tensorflow", "pytorch", "keras", "pandas", "numpy", "scikit",
    "bootstrap", "tailwind", "jquery", "svelte", "nestjs", "graphql",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite",
    "oracle", "dynamodb", "cassandra", "firebase", "supabase", "neo4j",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "gitlab", "github", "circleci", "nginx", "apache",
    "cloudflare", "heroku", "vercel", "netlify", "digitalocean",
    # Tools & Practices
    "git", "linux", "agile", "scrum", "kanban", "jira", "confluence",
    "figma", "sketch", "postman", "swagger", "webpack", "vite",
    "rest", "api", "microservices", "cicd", "devops", "mlops",
    # Data & AI/ML
    "machinelearning", "deeplearning", "nlp", "computervision",
    "datascience", "dataengineering", "analytics", "bigdata",
    "hadoop", "spark", "airflow", "tableau", "powerbi", "excel",
    "statistics", "regression", "classification", "clustering",
    # Business / Soft skills
    "leadership", "communication", "teamwork", "problemsolving",
    "management", "strategy", "marketing", "sales", "finance",
    "accounting", "consulting", "product", "design", "ux", "ui",
})

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract plain text from every page of a PDF using pdfplumber.

    Args:
        file_bytes: Raw bytes of an uploaded PDF file.

    Returns:
        A single clean string of all extracted text.

    Raises:
        ValueError: If the file is empty, unreadable, or contains no text.
    """
    if not file_bytes:
        raise ValueError("Uploaded PDF is empty.")

    page_texts: list[str] = []

    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                raise ValueError("PDF contains no pages.")

            for page_num, page in enumerate(pdf.pages, start=1):
                raw: str | None = page.extract_text()
                if raw and raw.strip():
                    page_texts.append(raw.strip())

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc

    if not page_texts:
        raise ValueError(
            "No readable text found in the PDF. "
            "The file may be a scanned image — try a text-based PDF."
        )

    full_text = "\n".join(page_texts)
    return _normalise_unicode(full_text)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Prepare text for TF-IDF vectorisation.

    Pipeline:
      1. Normalise unicode (smart quotes, dashes → ASCII equivalents)
      2. Lowercase
      3. Strip punctuation (keep alphanumeric + underscore)
      4. Remove English stopwords
      5. Drop tokens shorter than _MIN_TOKEN_LEN characters

    Returns:
        A whitespace-separated string of meaningful tokens.
    """
    if not text or not text.strip():
        return ""

    normalised = _normalise_unicode(text).lower()
    no_punct = _KEEP_PATTERN.sub(" ", normalised)

    tokens = [
        tok
        for tok in _WHITESPACE_PATTERN.split(no_punct)
        if tok and len(tok) >= _MIN_TOKEN_LEN and tok not in STOP_WORDS
    ]

    return " ".join(tokens)


def clean_text(text: str) -> str:
    """Collapse repeated whitespace — lightweight version for display strings."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def detect_sections(text: str) -> dict[str, bool]:
    """
    Detect which standard resume sections are present in the text.

    Returns a dict like {"contact": True, "education": True, "experience": False, ...}
    """
    lower = text.lower()
    found: dict[str, bool] = {}

    for section, keywords in REQUIRED_SECTIONS.items():
        found[section] = any(kw in lower for kw in keywords)

    return found


# ---------------------------------------------------------------------------
# Quantifiable impact detection
# ---------------------------------------------------------------------------

_PERCENT_PATTERN = re.compile(r"\d+\s*%")
_DOLLAR_PATTERN = re.compile(r"\$\s*[\d,]+")
_NUMBER_PATTERN = re.compile(r"\b\d{2,}\b")  # numbers with 2+ digits

IMPACT_PHRASES: frozenset[str] = frozenset({
    "increased", "decreased", "reduced", "improved", "grew", "saved",
    "generated", "boosted", "cut", "lowered", "raised", "doubled",
    "tripled", "halved", "revenue", "profit", "efficiency", "performance",
    "conversion", "retention", "engagement", "throughput", "uptime",
    "latency", "accuracy", "coverage", "adoption",
})


def count_impact_indicators(text: str) -> dict[str, int]:
    """
    Count quantifiable impact indicators in the text.

    Returns:
        Dict with counts of percentages, dollar amounts, numbers, and impact phrases.
    """
    lower = text.lower()
    return {
        "percentages": len(_PERCENT_PATTERN.findall(text)),
        "dollar_amounts": len(_DOLLAR_PATTERN.findall(text)),
        "numbers": len(_NUMBER_PATTERN.findall(text)),
        "impact_phrases": sum(1 for phrase in IMPACT_PHRASES if phrase in lower),
    }


# ---------------------------------------------------------------------------
# Contact info detection
# ---------------------------------------------------------------------------

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"[\+]?[\d\-\(\)\s]{7,15}")
_LINKEDIN_PATTERN = re.compile(r"linkedin\.com/in/", re.IGNORECASE)
_GITHUB_PATTERN = re.compile(r"github\.com/", re.IGNORECASE)


def detect_contact_info(text: str) -> dict[str, bool]:
    """Detect presence of common contact information."""
    return {
        "email": bool(_EMAIL_PATTERN.search(text)),
        "phone": bool(_PHONE_PATTERN.search(text)),
        "linkedin": bool(_LINKEDIN_PATTERN.search(text)),
        "github": bool(_GITHUB_PATTERN.search(text)),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_unicode(text: str) -> str:
    """
    Convert smart-quotes, em-dashes, ligatures etc. to plain ASCII equivalents.
    """
    normalised = unicodedata.normalize("NFKD", text)
    return normalised.encode("ascii", errors="ignore").decode("ascii")
