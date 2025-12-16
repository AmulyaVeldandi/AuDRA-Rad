from __future__ import annotations

"""Agent state tracking and persistence utilities."""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class AgentState(BaseModel):
    """Represents the evolving decision state for a radiology report session."""

    session_id: str
    report_id: str
    report_text: str
    patient_id: Optional[str] = None
    patient_context: Optional[Dict[str, Any]] = None
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_guidelines: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    tasks_generated: List[Dict[str, Any]] = Field(default_factory=list)
    decision_trace: List[Dict[str, Any]] = Field(default_factory=list)
    status: Literal[
        "initialized",
        "parsing",
        "retrieving",
        "matching",
        "validating",
        "completed",
        "failed",
        "requires_review",
    ] = "initialized"
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    # ------------------------------------------------------------------ #
    # Mutation helpers

    def add_finding(self, finding: Dict[str, Any]) -> None:
        """Append a parsed finding to the state."""

        self.findings.append(finding)
        self._touch()

    def add_guideline(self, guideline: Dict[str, Any]) -> None:
        """Record a retrieved guideline snippet."""

        self.retrieved_guidelines.append(guideline)
        self._touch()

    def add_recommendation(self, recommendation: Dict[str, Any]) -> None:
        """Store an agent-generated recommendation."""

        self.recommendations.append(recommendation)
        self._touch()

    def add_decision_step(self, step: str, data: Dict[str, Any]) -> None:
        """Log a decision step for audit/compliance tracking."""

        entry = {
            "step": step,
            "timestamp": _utcnow().isoformat(),
        }
        entry.update(deepcopy(data))
        self.decision_trace.append(entry)
        self._touch()

    # ------------------------------------------------------------------ #
    # Serialization helpers

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the state into a JSON-compatible dictionary."""

        payload = self.model_dump()
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Deserialize a dictionary back into an AgentState instance."""

        cleaned = deepcopy(data)
        cleaned["created_at"] = cls._parse_datetime(cleaned.get("created_at"))
        cleaned["updated_at"] = cls._parse_datetime(cleaned.get("updated_at"))
        return cls(**cleaned)

    # ------------------------------------------------------------------ #
    # Internal utilities

    def _touch(self) -> None:
        """Refresh the updated_at timestamp."""

        self.updated_at = _utcnow()

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        raise TypeError(f"Cannot parse datetime value: {value!r}")


class StateManager:
    """Simple in-memory persistence layer for AgentState instances."""

    _memory_store: Dict[str, AgentState] = {}

    @classmethod
    def save_state(cls, state: AgentState) -> None:
        """Persist the supplied state using in-memory storage."""

        cls._memory_store[state.session_id] = state

    @classmethod
    def load_state(cls, session_id: str) -> AgentState:
        """Return the state for the given session identifier."""

        try:
            return cls._memory_store[session_id]
        except KeyError as exc:
            raise KeyError(f"No state found for session '{session_id}'.") from exc

    @classmethod
    def list_active_sessions(cls) -> List[str]:
        """Return session identifiers that are still in progress."""

        return [
            session_id
            for session_id, state in cls._memory_store.items()
            if state.status not in {"completed", "failed"}
        ]

    @classmethod
    def clear_state(cls, session_id: str) -> None:
        """Remove a state object from the memory store."""

        cls._memory_store.pop(session_id, None)
