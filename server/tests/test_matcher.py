from server.app.models.schemas import Job, JobFilters, ResumeProfile
from server.app.services.matcher import recommend_jobs


def make_job(**overrides: object) -> Job:
    values: dict[str, object] = {
        "id": "job-1",
        "title": "Backend Developer",
        "company": "Example Co",
        "location": "London, UK",
        "is_remote": False,
        "experience_required": "1-3 years",
        "skills": ["JavaScript", "Node.js", "PostgreSQL"],
        "description": "Build services.",
        "source_link": "https://example.com/job",
        "source": "test",
        "posted_date": "",
    }
    values.update(overrides)
    return Job.model_validate(values)


def test_exact_skill_title_experience_and_location_match_scores_one() -> None:
    profile = ResumeProfile(
        designation="Backend Developer",
        skills=["JS", "Node", "PostgreSQL"],
        experience_years=2,
    )
    result = recommend_jobs(
        [make_job()],
        profile,
        profile_location="London",
    )

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].match_reasons == [
        "Matches 3/3 skills: JS, Node, PostgreSQL",
        "title similar to 'Backend Developer'",
        "experience 2y fits 1-3y",
        "location preference matches",
    ]


def test_hard_filters_remove_non_matching_jobs() -> None:
    jobs = [
        make_job(id="1"),
        make_job(id="2", is_remote=True, location="Remote"),
        make_job(id="3", skills=["Python"]),
    ]
    results = recommend_jobs(
        jobs,
        ResumeProfile(skills=["React"]),
        JobFilters(remote_only=True, required_skill="React"),
    )

    assert [result.job.id for result in results] == []


def test_required_skill_synonyms_are_matched() -> None:
    results = recommend_jobs(
        [make_job(skills=["JavaScript"])],
        ResumeProfile(skills=["JS"]),
        JobFilters(required_skill="js"),
    )

    assert len(results) == 1
    assert results[0].score == 0.5
    assert results[0].match_reasons[0] == "Matches 1/1 skills: JS"


def test_top_k_limits_ranked_results() -> None:
    jobs = [make_job(id=str(i), title=f"Backend Developer {i}") for i in range(8)]

    assert len(recommend_jobs(jobs, ResumeProfile(), top_k=3)) == 3
