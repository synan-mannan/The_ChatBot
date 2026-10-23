from __future__ import annotations

import re
import math
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from server.app.models.schemas import Job, JobFilters, JobRecommendation, ResumeProfile

SKILL_SYNONYMS = {
    "js": "javascript",
    "node": "node.js",
    "nodejs": "node.js",
    "py": "python",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "reactjs": "react",
    "ts": "typescript",
}


def _canonical(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9+#.]", "", value.casefold())
    return SKILL_SYNONYMS.get(normalized, normalized)


def _skill_overlap(profile_skills: Iterable[str], job_skills: Iterable[str]) -> tuple[float, list[str]]:
    candidate = {_canonical(skill): skill for skill in profile_skills if skill.strip()}
    listed = {_canonical(skill): skill for skill in job_skills if skill.strip()}
    overlap = sorted(candidate.keys() & listed.keys())
    if not candidate:
        return 0.0, []
    return len(overlap) / len(candidate), [candidate[key] for key in overlap]


def _title_similarity_scores(candidate_title: str | None, job_titles: list[str]) -> list[float]:
    if not candidate_title or not candidate_title.strip():
        return [0.0] * len(job_titles)
    if not job_titles:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        matrix = vectorizer.fit_transform([candidate_title, *job_titles])
    except ValueError:
        return [0.0] * len(job_titles)
    scores = cosine_similarity(matrix[0:1], matrix[1:])[0]
    return [float(score) for score in scores]


def _experience_range(value: str) -> tuple[float, float] | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*years?", value, re.I)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*years?", value, re.I)
    if match:
        return float(match.group(1)), float("inf")
    match = re.search(r"(\d+(?:\.\d+)?)\s*years?", value, re.I)
    if match:
        years = float(match.group(1))
        return years, years
    return None


def _experience_fit(candidate_years: float | None, requirement: str) -> tuple[float, bool | None]:
    target = _experience_range(requirement)
    if target is None or candidate_years is None:
        return 0.0, None
    minimum, maximum = target
    fits = minimum <= candidate_years <= maximum
    if fits:
        return 1.0, True
    distance = minimum - candidate_years if candidate_years < minimum else candidate_years - maximum
    return max(0.0, 1.0 - distance / max(minimum, 1.0)), False


def _location_score(profile_location: str | None, job: Job) -> float:
    if not profile_location:
        return 0.0
    if profile_location.casefold() in job.location.casefold():
        return 1.0
    if job.is_remote:
        return 1.0
    return 0.0


def _passes_filters(job: Job, filters: JobFilters) -> bool:
    if filters.remote_only is True and not job.is_remote:
        return False
    if filters.location and filters.location.casefold() not in job.location.casefold() and not job.is_remote:
        return False
    if filters.required_skill and _canonical(filters.required_skill) not in {
        _canonical(skill) for skill in job.skills
    }:
        return False
    if filters.company and filters.company.casefold() not in job.company.casefold():
        return False
    job_exp = _experience_range(job.experience_required)
    if filters.experience_min is not None and job_exp and job_exp[1] < filters.experience_min:
        return False
    if filters.experience_max is not None and job_exp and job_exp[0] > filters.experience_max:
        return False
    return True


def recommend_jobs(
    jobs: list[Job],
    profile: ResumeProfile,
    filters: JobFilters | None = None,
    top_k: int = 10,
    profile_location: str | None = None,
) -> list[JobRecommendation]:
    active_filters = filters or JobFilters()
    designation = profile.designation or ""
    eligible_jobs = [job for job in jobs if _passes_filters(job, active_filters)]
    title_scores = _title_similarity_scores(
        designation, [job.title for job in eligible_jobs]
    )
    ranked: list[JobRecommendation] = []
    for job, title_score in zip(eligible_jobs, title_scores, strict=True):
        skill_score, overlaps = _skill_overlap(profile.skills, job.skills)
        experience_score, experience_fits = _experience_fit(
            profile.experience_years, job.experience_required
        )
        location_score = _location_score(profile_location, job)
        score = (
            0.50 * skill_score
            + 0.25 * title_score
            + 0.15 * experience_score
            + 0.10 * location_score
        )
        reasons: list[str] = []
        if profile.skills:
            reasons.append(f"Matches {len(overlaps)}/{len(profile.skills)} skills")
        if overlaps:
            reasons[-1] += f": {', '.join(overlaps[:6])}"
        if designation and title_score > 0:
            reasons.append(f"title similar to '{designation}'")
        if experience_fits is True:
            bounds = _experience_range(job.experience_required)
            if math.isinf(bounds[1]):
                required = f"{bounds[0]:g}+y"
            else:
                required = f"{bounds[0]:g}-{bounds[1]:g}y"
            reasons.append(f"experience {profile.experience_years:g}y fits {required}")
        elif experience_fits is False:
            reasons.append("experience is outside the preferred range")
        if location_score and profile_location:
            reasons.append("location preference matches")
        if not reasons:
            reasons.append("Relevant role from the available job listings")
        ranked.append(
            JobRecommendation(job=job, score=round(score, 4), match_reasons=reasons)
        )
    ranked.sort(key=lambda item: (-item.score, item.job.title.casefold(), item.job.id))
    return ranked[: max(0, top_k)]
