"""Convert guideline recommendations into actionable follow-up tasks."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from src.utils.logger import get_logger

TaskPriority = Literal["routine", "priority", "urgent", "stat"]

PROCEDURE_CODES: Dict[str, Dict[str, Dict[str, str]]] = {
    "CT Chest": {
        "CPT": {"code": "71250", "display": "CT thorax without contrast"},
        "LOINC": {"code": "30621-7", "display": "CT Chest"},
    },
    "CT Abdomen": {
        "CPT": {"code": "74150", "display": "CT abdomen without contrast"},
        "LOINC": {"code": "30704-1", "display": "CT Abdomen"},
    },
    "MRI Brain": {
        "CPT": {"code": "70551", "display": "MRI brain without contrast"},
        "LOINC": {"code": "24558-9", "display": "MR Brain"},
    },
    "PET-CT": {
        "CPT": {"code": "78815", "display": "PET-CT skull base to mid-thigh"},
    },
    "Pulmonology Referral": {
        "CUSTOM": {"code": "REF-PULM", "display": "Pulmonology referral consultation"},
    },
}


@dataclass(frozen=True)
class Task:
    """Structured representation of an actionable follow-up task."""

    task_id: str
    procedure_name: str
    procedure_code: Dict[str, Any]
    scheduled_date: date
    priority: TaskPriority
    clinical_reason: str
    patient_id: str
    ordering_provider: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the task into a JSON-friendly dictionary."""

        payload = asdict(self)
        payload["scheduled_date"] = self.scheduled_date.isoformat()
        payload["created_at"] = self.created_at.isoformat()
        return payload


class TaskGenerator:
    """Construct structured follow-up tasks from guideline recommendations."""

    def __init__(
        self,
        *,
        default_ordering_provider: str = "AuDRA-Rad Automated Coordinator",
    ) -> None:
        self._default_ordering_provider = default_ordering_provider
        self._logger = get_logger("audra.tasks.generator")

    # ------------------------------------------------------------------ #
    def generate_task(
        self,
        recommendation: Dict[str, Any],
        finding: Dict[str, Any],
        patient_id: str,
    ) -> Task:
        """Convert a recommendation and finding into an actionable Task."""

        rec_data = self._ensure_dict(recommendation)
        finding_data = self._ensure_dict(finding)

        if not rec_data:
            raise ValueError("Recommendation payload is required.")
        if not finding_data:
            raise ValueError("Finding payload is required.")
        if not patient_id:
            raise ValueError("Patient identifier is required.")

        procedure_name = self._extract_procedure_name(rec_data)
        procedure_code = self.map_procedure_to_code(procedure_name)

        priority = self._normalise_priority(rec_data.get("urgency"))

        timeframe = self._safe_int(rec_data.get("timeframe_months"))
        scheduled_date = self.calculate_scheduled_date(timeframe)

        clinical_reason = self.format_clinical_reason(rec_data, finding_data)
        if not clinical_reason:
            raise ValueError("Clinical reason could not be constructed.")

        ordering_provider = rec_data.get("ordering_provider") or self._default_ordering_provider
        created_at = datetime.now(timezone.utc)

        metadata = {"recommendation": rec_data, "finding": finding_data}

        task = Task(
            task_id=str(uuid4()),
            procedure_name=procedure_name,
            procedure_code=procedure_code,
            scheduled_date=scheduled_date,
            priority=priority,
            clinical_reason=clinical_reason,
            patient_id=patient_id,
            ordering_provider=ordering_provider,
            created_at=created_at,
            metadata=metadata,
        )

        self._logger.info(
            "Generated follow-up task.",
            extra={"context": self._build_log_context(task)},
        )
        return task

    # ------------------------------------------------------------------ #
    def calculate_scheduled_date(self, timeframe_months: Optional[int], base_date: Optional[date] = None) -> date:
        """Return a future scheduled date based on the supplied timeframe."""

        base = base_date or date.today()
        if timeframe_months is None:
            timeframe_months = 0
        if timeframe_months < 0:
            raise ValueError("Timeframe must be zero or positive.")

        if timeframe_months == 0:
            scheduled = base + timedelta(days=1)
        else:
            scheduled = self._add_months(base, timeframe_months)

        if scheduled <= base:
            scheduled = base + timedelta(days=1)
        return scheduled

    # ------------------------------------------------------------------ #
    def map_procedure_to_code(self, procedure_name: str) -> Dict[str, Any]:
        """Map a procedure name to its coding representation (CPT/LOINC/custom)."""

        normalized = procedure_name.strip().lower()
        for key, coding in PROCEDURE_CODES.items():
            key_norm = key.lower()
            if normalized == key_norm or key_norm in normalized:
                return self._construct_code_payload(coding)

        raise ValueError(f"Procedure '{procedure_name}' is not recognised.")

    # ------------------------------------------------------------------ #
    def format_clinical_reason(self, recommendation: Dict[str, Any], finding: Dict[str, Any]) -> str:
        """Produce a concise clinical justification for the scheduled task."""

        finding_type = finding.get("finding_type") or finding.get("type") or "finding"
        location = finding.get("location")
        size_mm = finding.get("size_mm")

        characteristics = finding.get("characteristics") or []
        if isinstance(characteristics, str):
            characteristics = [characteristics]
        characteristic_text = ""
        if characteristics:
            characteristic_text = " " + ", ".join(sorted(set(characteristics)))

        follow_up = recommendation.get("follow_up_type") or "follow-up imaging"
        guideline = recommendation.get("citation") or recommendation.get("guideline") or "clinical guidelines"
        timeframe = recommendation.get("timeframe_months")

        timeframe_text = ""
        if isinstance(timeframe, (int, float)) and timeframe > 0:
            timeframe_text = f" in {int(timeframe)} month{'s' if int(timeframe) != 1 else ''}"

        finding_description_parts = ["Follow-up"]
        if size_mm:
            finding_description_parts.append(f"{self._format_number(size_mm)}mm")
        finding_description_parts.append(f"{finding_type}{characteristic_text}".strip())
        if location:
            finding_description_parts.append(f"in {location}")

        finding_sentence = " ".join(part for part in finding_description_parts if part).strip()
        reason = (
            f"{finding_sentence} per {guideline}. "
            f"Recommend {follow_up}{timeframe_text}."
        )

        reasoning = recommendation.get("reasoning")
        if reasoning:
            reason = f"{reason} {reasoning.strip()}"

        reason = " ".join(reason.split())
        if len(reason) > 500:
            reason = f"{reason[:497]}..."
        return reason

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _construct_code_payload(self, coding: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        preferred_order = ["CPT", "LOINC"]
        ordered_items = []

        for system in preferred_order:
            details = coding.get(system)
            if details:
                ordered_items.append((system, details))

        for system, details in coding.items():
            if system not in preferred_order:
                ordered_items.append((system, details))

        primary = None
        alternates = []
        for system, details in ordered_items:
            entry = {"system": system, **details}
            if primary is None:
                primary = entry
            else:
                alternates.append(entry)

        if primary is None:
            raise ValueError("Procedure coding map is empty.")

        payload: Dict[str, Any] = dict(primary)
        if alternates:
            payload["alternate_codes"] = alternates
        return payload

    def _extract_procedure_name(self, recommendation: Dict[str, Any]) -> str:
        name = recommendation.get("procedure") or recommendation.get("follow_up_type") or ""
        name = name.strip()
        if not name:
            raise ValueError("Recommendation must include a follow-up procedure name.")
        return name

    def _normalise_priority(self, urgency: Optional[str]) -> TaskPriority:
        if not urgency:
            return "routine"

        sanitized = urgency.strip().lower()
        if sanitized == "semi-urgent":
            sanitized = "priority"
        if sanitized not in {"routine", "priority", "urgent", "stat"}:
            raise ValueError(f"Unsupported urgency level '{urgency}'.")
        return sanitized  # type: ignore[return-value]

    def _add_months(self, base: date, months: int) -> date:
        total_months = base.month - 1 + months
        year = base.year + total_months // 12
        month = total_months % 12 + 1

        day = min(base.day, self._days_in_month(year, month))
        return date(year, month, day)

    def _days_in_month(self, year: int, month: int) -> int:
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        return (next_month - date(year, month, 1)).days

    def _ensure_dict(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "__dict__"):
            return {k: v for k, v in vars(payload).items() if not k.startswith("_")}
        raise TypeError("Payload must be a dictionary-like object.")

    def _safe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            result = int(value)
        except (TypeError, ValueError):
            return None
        return result

    def _format_number(self, value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.1f}".rstrip("0").rstrip(".")

    def _build_log_context(self, task: Task) -> Dict[str, Any]:
        primary_code = {
            "system": task.procedure_code.get("system"),
            "code": task.procedure_code.get("code"),
            "display": task.procedure_code.get("display"),
        }
        context = {
            "task_id": task.task_id,
            "patient_id": task.patient_id,
            "procedure_name": task.procedure_name,
            "scheduled_date": task.scheduled_date.isoformat(),
            "priority": task.priority,
            "clinical_reason": task.clinical_reason,
            "ordering_provider": task.ordering_provider,
            "procedure_code": primary_code,
        }
        if "alternate_codes" in task.procedure_code:
            context["alternate_codes"] = task.procedure_code["alternate_codes"]
        context["metadata"] = task.metadata
        return context


__all__ = ["Task", "TaskGenerator", "PROCEDURE_CODES"]
