from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from server.app.models.db import ResumeProfileRow
from server.app.tools.base import ToolContext, ToolDefinition

ALLOWED_ANSWER_KEYS = {
    "work_authorization",
    "requires_sponsorship",
    "notice_period",
    "expected_salary",
    "current_salary",
    "willing_to_relocate",
    "earliest_start_date",
    "gender",
    "ethnicity",
    "disability_status",
    "veteran_status",
    "criminal_history",
    "terms_consent",
    "privacy_consent",
    "legal_attestation",
}


class SaveApplicationAnswersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: dict[str, str | bool | float] = Field(min_length=1)

    @field_validator("answers")
    @classmethod
    def validate_answer_keys(
        cls, answers: dict[str, str | bool | float]
    ) -> dict[str, str | bool | float]:
        unknown = set(answers) - ALLOWED_ANSWER_KEYS
        if unknown:
            raise ValueError(f"Unsupported application answer keys: {', '.join(sorted(unknown))}")
        return answers


async def save_application_answers_tool(
    context: ToolContext, arguments: SaveApplicationAnswersInput
) -> dict[str, Any]:
    context.session_store.set_answers(context.session_id, arguments.answers)
    row = await context.db_session.get(ResumeProfileRow, context.session_id)
    if row is None:
        row = ResumeProfileRow(
            session_id=context.session_id,
            profile_data={},
            answers_data=dict(context.state.answers),
            approved_cover_letters_data={},
        )
        context.db_session.add(row)
    else:
        row.answers_data = dict(context.state.answers)
    await context.db_session.commit()
    return {
        "status": "saved",
        "saved_keys": sorted(arguments.answers),
        "message": "Saved those application answers for this session.",
    }


SAVE_APPLICATION_ANSWERS_DEFINITION = ToolDefinition(
    name="save_application_answers",
    description=(
        "Save application answers explicitly provided by the user. Call this when they tell "
        "you details such as notice period, work authorization, salary, relocation, availability, "
        "or optional EEO information. Never infer answers."
    ),
    input_model=SaveApplicationAnswersInput,
    execute=save_application_answers_tool,
)
