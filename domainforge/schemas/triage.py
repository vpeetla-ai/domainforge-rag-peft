from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TriageResponse(BaseModel):
    """Strict JSON envelope for support triage (PEFT target + API response)."""

    intent: str
    category: str
    priority: Priority
    entities: dict[str, Any] = Field(default_factory=dict)
    suggested_action: str
    cite_faq_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None

    @field_validator("suggested_action")
    @classmethod
    def action_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("suggested_action must not be empty")
        return value

    def to_json_str(self) -> str:
        return self.model_dump_json()
