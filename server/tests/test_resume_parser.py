from io import BytesIO

import pytest
from docx import Document
from pydantic import ValidationError

from server.app.models.schemas import ResumeProfile
from server.app.services.resume_parser import (
    ResumeValidationError,
    extract_resume_text,
    validate_resume_profile,
)

MAX_BYTES = 5 * 1024 * 1024


def test_extracts_text_from_docx() -> None:
    document = Document()
    document.add_paragraph(
        "Jordan Lee jordan@example.com Python developer with five years of experience. "
        "Built customer-facing projects and studied computer science."
    )
    buffer = BytesIO()
    document.save(buffer)

    text = extract_resume_text("resume.docx", buffer.getvalue(), MAX_BYTES)

    assert "Jordan Lee" in text
    assert "computer science" in text


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("resume.txt", b"x" * 200, "isn't supported"),
        ("resume.pdf", b"", "empty"),
        ("resume.pdf", b"x" * (MAX_BYTES + 1), "over 5 MB"),
        ("resume.pdf", b"not a valid PDF", "couldn't read"),
    ],
)
def test_rejects_invalid_resume_files(
    filename: str, content: bytes, message: str
) -> None:
    with pytest.raises(ResumeValidationError, match=message):
        extract_resume_text(filename, content, MAX_BYTES)


def test_rejects_short_or_scanned_pdf_text() -> None:
    document = Document()
    document.add_paragraph("Short text")
    buffer = BytesIO()
    document.save(buffer)

    with pytest.raises(ResumeValidationError, match="enough readable text"):
        extract_resume_text("scan.docx", buffer.getvalue(), MAX_BYTES)


def test_profile_schema_validates_extracted_fields() -> None:
    profile = validate_resume_profile(
        {
            "name": "Jordan Lee",
            "email": "jordan@example.com",
            "phone": None,
            "designation": "Python Developer",
            "skills": ["Python"],
            "experience_years": 5,
            "experience_summary": "Five years building APIs",
            "education": ["BSc Computer Science"],
            "projects": ["Customer platform"],
        }
    )

    assert isinstance(profile, ResumeProfile)
    assert profile.experience_years == 5
    with pytest.raises(ValidationError):
        validate_resume_profile({"experience_years": -1})
