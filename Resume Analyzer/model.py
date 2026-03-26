"""
model.py — ResumeAnalyzer: Multi-dimensional resume scoring system.

Scores resumes across 6 dimensions:
  1. Keyword Match  — TF-IDF cosine similarity vs job description
  2. Section Structure — Checks for expected resume sections
  3. Impact & Metrics — Counts quantifiable achievements
  4. Skills Depth — Matches against common industry skills
  5. Formatting Quality — Word count, line balance, contact info
  6. Action Verb Usage — Checks for strong action verbs

Overall score = weighted average → letter grade + suggestions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .utils import (
        preprocess_text, detect_sections, count_impact_indicators,
        detect_contact_info, ACTION_VERBS, COMMON_SKILLS,
    )
except ImportError:
    from utils import (
        preprocess_text, detect_sections, count_impact_indicators,
        detect_contact_info, ACTION_VERBS, COMMON_SKILLS,
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisResult:
    overall_score: float
    grade: str
    dimension_scores: dict[str, float]
    keywords_missing: list[str]
    suggestions: list[str]
    strengths: list[str]
    section_analysis: dict[str, bool]
    contact_info: dict[str, bool]

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "grade": self.grade,
            "dimension_scores": self.dimension_scores,
            "keywords_missing": self.keywords_missing,
            "suggestions": self.suggestions,
            "strengths": self.strengths,
            "section_analysis": self.section_analysis,
            "contact_info": self.contact_info,
        }


# ---------------------------------------------------------------------------
# Grade mapping
# ---------------------------------------------------------------------------

def _score_to_grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 85: return "A"
    if score >= 80: return "B+"
    if score >= 70: return "B"
    if score >= 60: return "C+"
    if score >= 50: return "C"
    if score >= 40: return "D"
    return "F"


# ---------------------------------------------------------------------------
# Dimension weights
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: dict[str, float] = {
    "keyword_match":   0.25,
    "section_structure": 0.20,
    "impact_metrics":  0.20,
    "skills_depth":    0.15,
    "formatting":      0.10,
    "action_verbs":    0.10,
}


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class ResumeAnalyzer:
    """
    Multi-dimensional resume scorer.

    Evaluates resumes across 6 dimensions and produces an overall score,
    letter grade, actionable suggestions, and identified strengths.
    """

    def __init__(
        self,
        max_features: int = 8000,
        top_keywords: int = 10,
        min_keyword_len: int = 3,
    ) -> None:
        self.max_features = max_features
        self.top_keywords = top_keywords
        self.min_keyword_len = min_keyword_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        resume_text: str,
        job_description: str,
    ) -> dict:
        """
        Run the full 6-dimension analysis pipeline.

        Args:
            resume_text:     Raw text extracted from the PDF resume.
            job_description: Raw job description string from the user.

        Returns:
            Dict with overall_score, grade, dimension_scores, keywords_missing,
            suggestions, strengths, section_analysis, and contact_info.
        """
        cleaned_resume = preprocess_text(resume_text)
        cleaned_job = preprocess_text(job_description)

        if not cleaned_resume:
            raise ValueError(
                "Resume has no meaningful text after cleaning. "
                "The PDF may be image-only or almost entirely stopwords."
            )
        if not cleaned_job:
            raise ValueError(
                "Job description has no meaningful text after cleaning."
            )

        # ── compute each dimension ────────────────────────────────────
        scores: dict[str, float] = {}
        suggestions: list[str] = []
        strengths: list[str] = []

        # 1. Keyword Match (TF-IDF cosine similarity)
        scores["keyword_match"] = self._keyword_match_score(cleaned_resume, cleaned_job)

        # 2. Section Structure
        sections = detect_sections(resume_text)
        scores["section_structure"] = self._section_score(sections)

        # 3. Impact & Metrics
        impact = count_impact_indicators(resume_text)
        scores["impact_metrics"] = self._impact_score(impact)

        # 4. Skills Depth
        scores["skills_depth"] = self._skills_score(cleaned_resume, cleaned_job)

        # 5. Formatting Quality
        contact = detect_contact_info(resume_text)
        scores["formatting"] = self._formatting_score(resume_text, contact)

        # 6. Action Verbs
        scores["action_verbs"] = self._action_verb_score(cleaned_resume)

        # ── overall score ──────────────────────────────────────────────
        overall = sum(
            scores[dim] * weight
            for dim, weight in DIMENSION_WEIGHTS.items()
        )
        overall = round(float(np.clip(overall, 0.0, 100.0)), 2)

        # ── missing keywords ───────────────────────────────────────────
        missing = self._missing_keywords(cleaned_resume, cleaned_job)

        # ── generate suggestions and strengths ─────────────────────────
        self._generate_feedback(scores, sections, impact, contact, suggestions, strengths)

        grade = _score_to_grade(overall)

        return AnalysisResult(
            overall_score=overall,
            grade=grade,
            dimension_scores={k: round(v, 2) for k, v in scores.items()},
            keywords_missing=missing,
            suggestions=suggestions,
            strengths=strengths,
            section_analysis=sections,
            contact_info=contact,
        ).to_dict()

    # ------------------------------------------------------------------
    # Dimension 1: Keyword Match
    # ------------------------------------------------------------------

    def _keyword_match_score(self, cleaned_resume: str, cleaned_job: str) -> float:
        vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 1),
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_job])
        sim: float = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(np.clip(sim * 100, 0.0, 100.0)), 2)

    # ------------------------------------------------------------------
    # Dimension 2: Section Structure
    # ------------------------------------------------------------------

    def _section_score(self, sections: dict[str, bool]) -> float:
        # Core sections are weighted more heavily
        core_sections = ["contact", "education", "experience", "skills"]
        bonus_sections = ["projects", "summary", "certifications", "achievements"]

        core_found = sum(1 for s in core_sections if sections.get(s, False))
        bonus_found = sum(1 for s in bonus_sections if sections.get(s, False))

        # Core: 60 points max (15 each), Bonus: 40 points max (10 each)
        score = (core_found / len(core_sections)) * 60 + (bonus_found / len(bonus_sections)) * 40
        return round(min(score, 100.0), 2)

    # ------------------------------------------------------------------
    # Dimension 3: Impact & Metrics
    # ------------------------------------------------------------------

    def _impact_score(self, impact: dict[str, int]) -> float:
        total = (
            impact["percentages"] * 8 +
            impact["dollar_amounts"] * 8 +
            min(impact["numbers"], 15) * 3 +
            min(impact["impact_phrases"], 10) * 4
        )
        # Cap at 100
        return round(min(total, 100.0), 2)

    # ------------------------------------------------------------------
    # Dimension 4: Skills Depth
    # ------------------------------------------------------------------

    def _skills_score(self, cleaned_resume: str, cleaned_job: str) -> float:
        resume_words = set(cleaned_resume.split())
        job_words = set(cleaned_job.split())

        # Skills found in resume that are in our known skills list
        resume_skills = resume_words & COMMON_SKILLS
        # Skills from job description that are in our known skills list
        job_skills = job_words & COMMON_SKILLS

        # Score based on: how many job-relevant skills the resume has +
        # bonus for having extra skills showing breadth
        if job_skills:
            match_ratio = len(resume_skills & job_skills) / len(job_skills)
        else:
            match_ratio = 0.0

        # Breadth bonus: having many skills regardless of job match
        breadth_bonus = min(len(resume_skills), 20) * 1.5  # up to 30 points

        # Match is worth 70%, breadth 30%
        score = match_ratio * 70 + breadth_bonus
        return round(min(score, 100.0), 2)

    # ------------------------------------------------------------------
    # Dimension 5: Formatting Quality
    # ------------------------------------------------------------------

    def _formatting_score(self, raw_text: str, contact: dict[str, bool]) -> float:
        score = 0.0
        words = raw_text.split()
        word_count = len(words)
        lines = raw_text.strip().split("\n")
        line_count = len(lines)

        # Word count: ideal range is 300-800 words
        if 300 <= word_count <= 800:
            score += 30
        elif 200 <= word_count <= 1000:
            score += 20
        elif 100 <= word_count <= 1500:
            score += 10
        # else: too short or too long = 0

        # Line count: ideal range is 30-80 lines
        if 30 <= line_count <= 80:
            score += 15
        elif 20 <= line_count <= 120:
            score += 10
        elif 10 <= line_count:
            score += 5

        # Contact info presence
        if contact.get("email"):
            score += 15
        if contact.get("phone"):
            score += 10
        if contact.get("linkedin"):
            score += 10
        if contact.get("github"):
            score += 10

        # Excessive caps check — more than 15% caps text is bad
        if raw_text:
            caps_ratio = sum(1 for c in raw_text if c.isupper()) / max(len(raw_text), 1)
            if caps_ratio < 0.15:
                score += 10
            elif caps_ratio < 0.25:
                score += 5

        return round(min(score, 100.0), 2)

    # ------------------------------------------------------------------
    # Dimension 6: Action Verb Usage
    # ------------------------------------------------------------------

    def _action_verb_score(self, cleaned_resume: str) -> float:
        resume_words = set(cleaned_resume.split())
        verbs_found = resume_words & ACTION_VERBS
        count = len(verbs_found)

        # Scoring: 0 verbs = 0, 1-3 = 20, 4-6 = 40, 7-10 = 60, 11-15 = 80, 16+ = 100
        if count >= 16:
            score = 100
        elif count >= 11:
            score = 80
        elif count >= 7:
            score = 60
        elif count >= 4:
            score = 40
        elif count >= 1:
            score = 20
        else:
            score = 0

        return float(score)

    # ------------------------------------------------------------------
    # Missing keywords
    # ------------------------------------------------------------------

    def _missing_keywords(
        self, cleaned_resume: str, cleaned_job: str
    ) -> list[str]:
        kw_vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 1),
            max_features=self.max_features,
            sublinear_tf=True,
        )
        job_matrix = kw_vectorizer.fit_transform([cleaned_job])
        vocab: list[str] = kw_vectorizer.get_feature_names_out().tolist()
        scores: list[float] = job_matrix.toarray()[0].tolist()

        ranked = sorted(
            ((term, sc) for term, sc in zip(vocab, scores) if sc > 0),
            key=lambda x: x[1],
            reverse=True,
        )

        resume_words: set[str] = set(cleaned_resume.split())
        missing: list[str] = []
        seen: set[str] = set()

        for term, _ in ranked:
            if len(term) < self.min_keyword_len:
                continue
            if not term.isalpha():
                continue
            if term in seen:
                continue
            if term not in resume_words and not any(term in rw for rw in resume_words):
                missing.append(term)
                seen.add(term)
            if len(missing) == self.top_keywords:
                break

        return missing

    # ------------------------------------------------------------------
    # Feedback generation
    # ------------------------------------------------------------------

    def _generate_feedback(
        self,
        scores: dict[str, float],
        sections: dict[str, bool],
        impact: dict[str, int],
        contact: dict[str, bool],
        suggestions: list[str],
        strengths: list[str],
    ) -> None:
        """Populate suggestions and strengths based on dimension scores."""

        # Keyword match
        if scores["keyword_match"] >= 60:
            strengths.append("Your resume strongly matches the job description keywords")
        elif scores["keyword_match"] >= 30:
            suggestions.append("Add more relevant keywords from the job description to your resume")
        else:
            suggestions.append("Your resume has very few keywords matching the job description — tailor it to the role")

        # Section structure
        missing_sections = [s for s, found in sections.items() if not found]
        if scores["section_structure"] >= 80:
            strengths.append("Resume has a well-organized section structure")
        if missing_sections:
            readable = [s.replace("_", " ").title() for s in missing_sections]
            suggestions.append(f"Add missing sections: {', '.join(readable)}")

        # Impact metrics
        if scores["impact_metrics"] >= 50:
            strengths.append("Great use of quantifiable achievements and metrics")
        elif scores["impact_metrics"] >= 20:
            suggestions.append("Add more quantifiable achievements (e.g., 'Improved performance by 40%')")
        else:
            suggestions.append("Include measurable results — use numbers, percentages, and dollar amounts to quantify your impact")

        # Skills depth
        if scores["skills_depth"] >= 60:
            strengths.append("Strong technical skills alignment with the job requirements")
        elif scores["skills_depth"] >= 30:
            suggestions.append("Consider adding more relevant technical skills that match the job description")
        else:
            suggestions.append("Your skills section needs significant improvement — list specific technologies, tools, and frameworks")

        # Formatting
        if scores["formatting"] >= 70:
            strengths.append("Resume has good formatting with appropriate length and contact details")
        else:
            if not contact.get("email"):
                suggestions.append("Add your email address to the resume")
            if not contact.get("phone"):
                suggestions.append("Add your phone number to the resume")
            if not contact.get("linkedin"):
                suggestions.append("Consider adding your LinkedIn profile URL")

        # Action verbs
        if scores["action_verbs"] >= 60:
            strengths.append("Good use of strong action verbs throughout the resume")
        elif scores["action_verbs"] >= 20:
            suggestions.append("Use more strong action verbs (e.g., 'developed', 'led', 'implemented', 'optimized')")
        else:
            suggestions.append("Replace weak language with strong action verbs — start bullet points with words like 'Built', 'Led', 'Designed', 'Increased'")
