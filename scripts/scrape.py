#!/usr/bin/env python3
"""Collect and normalize job listings from public job-board APIs."""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "jobs.json"
USER_AGENT = "AIJobJobAssistant/1.0 (+https://github.com/aijob)"
REQUEST_TIMEOUT = httpx.Timeout(20.0, connect=8.0)
REQUEST_DELAY_SECONDS = 1.25
MAX_PAGES_PER_SOURCE = 100
MAX_TARGET = 2_000

SKILL_TERMS: dict[str, tuple[str, ...]] = {
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure", "microsoft azure"),
    "C": (" c ", " c/c++", " c programming"),
    "C++": ("c++", "cpp"),
    "C#": ("c#", "c sharp"),
    "CSS": ("css", "cascading style sheets"),
    "Django": ("django",),
    "Docker": ("docker",),
    "Express": ("express.js", "expressjs"),
    "Figma": ("figma",),
    "Flask": ("flask",),
    "GCP": ("gcp", "google cloud platform"),
    "Go": ("golang", "go programming"),
    "GraphQL": ("graphql",),
    "HTML": ("html",),
    "Java": ("java",),
    "JavaScript": ("javascript", " js ", "ecmascript"),
    "Jenkins": ("jenkins",),
    "Kotlin": ("kotlin",),
    "Kubernetes": ("kubernetes", "k8s"),
    "Machine Learning": ("machine learning", " ml "),
    "MongoDB": ("mongodb", "mongo db"),
    "MySQL": ("mysql",),
    "Node.js": ("node.js", "nodejs", "node js"),
    "PHP": ("php",),
    "PostgreSQL": ("postgresql", "postgres"),
    "Python": ("python", " py "),
    "React": ("react.js", "reactjs", "react"),
    "Redis": ("redis",),
    "Ruby": ("ruby",),
    "Ruby on Rails": ("ruby on rails", "rails"),
    "Rust": ("rust",),
    "Svelte": ("svelte",),
    "Swift": ("swift",),
    "SQL": ("sql",),
    "TypeScript": ("typescript", " ts "),
    "Vue": ("vue.js", "vuejs"),
    "WordPress": ("wordpress",),
}

API_SOURCES = ("arbeitnow", "remotive", "jobicy", "himalayas", "remoteok")
SOURCE_URLS = {
    "arbeitnow": "https://www.arbeitnow.com/api/job-board-api",
    "remotive": "https://remotive.com/api/remote-jobs?limit=50",
    "jobicy": "https://jobicy.com/api/v2/remote-jobs?count=50",
    "himalayas": "https://himalayas.app/jobs/api?limit=50",
    "remoteok": "https://remoteok.com/api",
}

log = logging.getLogger("aijob.scrape")


def _retryable_http_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def plain_text(value: Any) -> str:
    extractor = _TextExtractor()
    extractor.feed(html.unescape(str(value or "")))
    return re.sub(r"\s+", " ", " ".join(extractor.parts)).strip()


def extract_skills(description: str, tags: Any = None) -> list[str]:
    text = description.lower()
    tag_values = tags if isinstance(tags, list) else [tags] if tags else []
    tag_text = " ".join(map(str, tag_values)).lower()
    found: list[str] = []
    for canonical, aliases in SKILL_TERMS.items():
        if any(_contains_term(text, alias) or _contains_term(tag_text, alias) for alias in aliases):
            found.append(canonical)
    return sorted(found)


def _contains_term(text: str, term: str) -> bool:
    pattern = re.escape(term.strip())
    return bool(re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text, re.I))


def extract_experience(description: str, level: str = "") -> str:
    text = f"{description} {level}"
    patterns = (
        re.compile(r"(\d{1,2})\s*(?:-|to)\s*(\d{1,2})\s*\+?\s*years?", re.I),
        re.compile(r"(\d{1,2})\s*\+\s*years?", re.I),
        re.compile(r"at least\s+(\d{1,2})\s+years?", re.I),
        re.compile(r"(\d{1,2})\s+years?(?:\s+of)?\s+experience", re.I),
    )
    match = patterns[0].search(text)
    if match:
        return f"{match.group(1)}-{match.group(2)} years"
    match = patterns[1].search(text) or patterns[2].search(text)
    if match:
        years = int(match.group(1))
        return f"{years}+ years"
    match = patterns[3].search(text)
    return f"{int(match.group(1))}+ years" if match else "Not specified"


def _is_remote(record: dict[str, Any], location: str, description: str) -> bool:
    declared = record.get("remote")
    text = " ".join(
        str(value or "")
        for value in (
            declared,
            record.get("isRemote"),
            record.get("jobType"),
            record.get("workplaceType"),
            location,
        )
    ).lower()
    return bool(declared is True or re.search(r"\b(remote|worldwide|anywhere)\b", text))


def _contact_email(record: dict[str, Any]) -> str | None:
    for key in (
        "contact_email",
        "contactEmail",
        "applicationEmail",
        "apply_email",
        "applyEmail",
        "email",
    ):
        value = record.get(key)
        if isinstance(value, str):
            candidate = value.strip()
            if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", candidate):
                return candidate
    return None


def normalize(source: str, record: dict[str, Any]) -> dict[str, Any] | None:
    if source == "arbeitnow":
        title = record.get("title")
        company = record.get("company_name")
        location = record.get("location", "")
        description = plain_text(record.get("description"))
        link = record.get("url") or (
            f"https://www.arbeitnow.com/view/{record['slug']}" if record.get("slug") else ""
        )
        date = record.get("created_at") or record.get("updated_at")
        experience_hint = ""
        tags = record.get("tags", [])
    elif source == "remotive":
        title = record.get("title")
        company = record.get("company_name")
        location = record.get("candidate_required_location", "")
        description = plain_text(record.get("description"))
        link = record.get("url", "")
        date = record.get("publication_date")
        experience_hint = record.get("job_type", "")
        tags = record.get("tags", [])
    elif source == "jobicy":
        title = record.get("jobTitle")
        company = record.get("companyName")
        location = record.get("jobGeo", "")
        description = plain_text(record.get("jobDescription"))
        link = record.get("url") or record.get("jobUrl", "")
        date = record.get("pubDate")
        experience_hint = record.get("jobLevel", "")
        tags = record.get("jobIndustry", [])
    elif source == "himalayas":
        title = record.get("title")
        company = record.get("companyName") or record.get("company")
        location = record.get("location") or record.get("locationRestrictions", "")
        description = plain_text(record.get("description"))
        link = record.get("applicationLink") or record.get("url", "")
        date = record.get("pubDate") or record.get("publishedAt")
        experience_hint = record.get("seniority", "")
        tags = record.get("categories", []) + record.get("technologies", [])
    else:
        title = record.get("position") or record.get("title")
        company = record.get("company")
        location = record.get("location", "")
        description = plain_text(record.get("description"))
        link = record.get("url", "")
        date = record.get("date")
        experience_hint = ""
        tags = record.get("tags", [])

    title = str(title or "").strip()
    company = str(company or "").strip()
    location = str(location or "Not specified").strip()
    if not title or not company or not link:
        return None
    if not urlparse(str(link)).scheme:
        return None
    return {
        "title": title,
        "company": company,
        "location": location,
        "contact_email": _contact_email(record),
        "is_remote": _is_remote(record, location, description),
        "experience_required": extract_experience(description, str(experience_hint or "")),
        "skills": extract_skills(description, tags),
        "description": description[:8_000],
        "source_link": str(link),
        "source": source,
        "posted_date": str(date or "")[:40],
    }


class JobCollector:
    def __init__(self, client: httpx.AsyncClient, delay: float) -> None:
        self.client = client
        self.delay = delay
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request: dict[str, float] = {}

    async def _allowed_by_robots(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            robots_url = urljoin(origin, "/robots.txt")
            try:
                response = await self.client.get(robots_url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("Skipping %s: robots.txt could not be checked (%s)", origin, exc)
                return False
            parser = RobotFileParser(robots_url)
            parser.parse(response.text.splitlines())
            self._robots[origin] = parser
        return self._robots[origin].can_fetch(USER_AGENT, url)

    async def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        request_url = str(httpx.URL(url).copy_merge_params(params or {}))
        if not await self._allowed_by_robots(request_url):
            raise PermissionError(f"robots.txt disallows or cannot verify {request_url}")
        host = urlparse(url).netloc
        elapsed = asyncio.get_running_loop().time() - self._last_request.get(host, 0.0)
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_retryable_http_error),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            stop=stop_after_attempt(3),
            reraise=True,
        ):
            with attempt:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        self._last_request[host] = asyncio.get_running_loop().time()
        return payload

    async def collect(self, source: str, limit: int) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        cursor: str | None = None
        for page in range(1, MAX_PAGES_PER_SOURCE + 1):
            try:
                if source == "arbeitnow":
                    payload = await self._get_json(SOURCE_URLS[source], {"page": page})
                    batch = payload.get("data", [])
                elif source == "remotive":
                    payload = await self._get_json(SOURCE_URLS[source], {"limit": 50, "page": page})
                    batch = payload.get("jobs", [])
                elif source == "jobicy":
                    params: dict[str, Any] = {"count": 50}
                    if cursor:
                        params["cursor"] = cursor
                    payload = await self._get_json(SOURCE_URLS[source], params)
                    batch = payload.get("jobs", [])
                    cursor = payload.get("nextCursor")
                elif source == "himalayas":
                    params = {"limit": 50}
                    if cursor:
                        params["cursor"] = cursor
                    payload = await self._get_json(SOURCE_URLS[source], params)
                    batch = payload.get("jobs", [])
                    cursor = payload.get("nextCursor")
                else:
                    payload = await self._get_json(SOURCE_URLS[source])
                    batch = payload[1:] if isinstance(payload, list) else payload.get("jobs", [])
                    batch = batch[:limit]
                if not batch:
                    break
                for item in batch:
                    normalized = normalize(source, item)
                    if normalized:
                        records.append(normalized)
                log.info("%s page %d: %d usable listings", source, page, len(batch))
                if source in ("jobicy", "himalayas") and not cursor:
                    break
                if source == "remoteok":
                    break
                if len(records) >= limit:
                    break
            except PermissionError as exc:
                log.warning("%s", exc)
                break
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                log.error("%s page %d failed: %s", source, page, exc)
                break
        return records[:limit]


def stable_jobs(records: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            record["title"].casefold(),
            record["company"].casefold(),
            record["location"].casefold(),
        )
        unique.setdefault(key, record)
    ordered = sorted(
        unique.values(),
        key=lambda item: (item["source"], item["company"].casefold(), item["title"].casefold()),
    )[:target]
    return [{"id": str(index), **job} for index, job in enumerate(ordered, start=1)]


async def scrape(sources: list[str], target: int, delay: float) -> list[dict[str, Any]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT, headers=headers, follow_redirects=True
    ) as client:
        collector = JobCollector(client, delay)
        collected: list[dict[str, Any]] = []
        per_source_limit = target
        for source in sources:
            batch = await collector.collect(source, per_source_limit)
            log.info("%s contributed %d listings", source, len(batch))
            collected.extend(batch)
            deduplicated = stable_jobs(collected, target)
            if len(deduplicated) >= target:
                break
        return stable_jobs(collected, target)


def print_stats(jobs: list[dict[str, Any]]) -> None:
    sources = Counter(job["source"] for job in jobs)
    with_skills = sum(bool(job["skills"]) for job in jobs)
    with_experience = sum(job["experience_required"] != "Not specified" for job in jobs)
    total = len(jobs)
    print(f"Listings: {total}")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}")
    print(f"With skills: {with_skills}/{total} ({with_skills / total:.1%})" if total else "With skills: 0")
    print(
        f"With experience: {with_experience}/{total} ({with_experience / total:.1%})"
        if total
        else "With experience: 0"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1_800)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS)
    parser.add_argument("--sources", nargs="+", choices=API_SOURCES, default=list(API_SOURCES))
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.target <= MAX_TARGET or args.delay < 0:
        raise SystemExit(f"--target must be between 1 and {MAX_TARGET}; --delay cannot be negative")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    jobs = await scrape(args.sources, args.target, args.delay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_stats(jobs)
    if len(jobs) < args.target:
        log.warning("Only %d/%d unique listings were available from allowed API sources", len(jobs), args.target)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit("Interrupted")
