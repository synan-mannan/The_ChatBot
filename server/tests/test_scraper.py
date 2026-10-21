from scripts.scrape import normalize


def test_normalize_preserves_direct_posting_url_and_optional_contact_email() -> None:
    job = normalize(
        "arbeitnow",
        {
            "title": "Backend Engineer",
            "company_name": "Example",
            "location": "Remote",
            "url": "https://boards.greenhouse.io/example/jobs/123",
            "contact_email": "careers@example.com",
        },
    )

    assert job is not None
    assert job["source_link"] == "https://boards.greenhouse.io/example/jobs/123"
    assert job["contact_email"] == "careers@example.com"


def test_normalize_allows_missing_or_invalid_contact_email() -> None:
    job = normalize(
        "arbeitnow",
        {
            "title": "Backend Engineer",
            "company_name": "Example",
            "location": "Remote",
            "url": "https://example.com/jobs/123",
            "contact_email": "not an email",
        },
    )

    assert job is not None
    assert job["contact_email"] is None
