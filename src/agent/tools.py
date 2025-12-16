from __future__ import annotations

"""Callable tools exposed to the AuDRA agent."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import ValidationError

from src.agent.state import AgentState
from src.guidelines.indexer import GuidelineChunk
from src.guidelines.matcher import Recommendation, RecommendationMatcher
from src.guidelines.retriever import GuidelineRetriever
from src.parsers.report_parser import Finding, ReportParser
from src.parsers.fhir_models import (
    Annotation,
    CodeableConcept,
    Reference,
    ServiceRequest,
    Timing,
    TimingRepeat,
)
from src.services.ehr_client import EHRClient
from src.utils.logger import get_logger


_LOGGER = get_logger("audra.agent.tools")


@dataclass
class TaskGenerator:
    """Construct follow-up service request payloads for EHR submission."""

    default_patient_id: str = "patient-demo"

    def build_order(self, state: AgentState, recommendation: Dict[str, Any]) -> ServiceRequest:
        """Create a FHIR ServiceRequest describing the follow-up action."""

        patient_id = (state.patient_context or {}).get("patient_id") or self.default_patient_id
        follow_up_type = recommendation.get("follow_up_type") or "Follow-up imaging"
        timeframe_months = recommendation.get("timeframe_months")
        urgency = recommendation.get("urgency", "routine")
        reasoning = recommendation.get("reasoning") or "Automated follow-up recommendation."

        period_unit = "mo"
        period_value: Optional[float] = None
        if isinstance(timeframe_months, (int, float)) and timeframe_months > 0:
            period_value = float(timeframe_months)
        elif urgency in {"urgent", "stat"}:
            period_unit = "wk"
            period_value = 1.0

        timing_repeat = (
            TimingRepeat(frequency=1, period=period_value, periodUnit=period_unit)
            if period_value
            else None
        )

        occurrence = Timing(repeat=timing_repeat)

        note = Annotation(
            text=f"{urgency.title()} follow-up: {reasoning}",
            time=datetime.now(timezone.utc),
        )

        return ServiceRequest(
            status="draft",
            intent="order",
            code=CodeableConcept(text=follow_up_type),
            subject=Reference(reference=f"Patient/{patient_id}"),
            authoredOn=datetime.now(timezone.utc),
            occurrenceTiming=occurrence,
            reasonReference=Reference(reference=f"DiagnosticReport/{state.report_id}"),
            note=[note],
        )


@dataclass
class ToolDependencies:
    """Container for tool-level dependencies."""

    parser: ReportParser
    retriever: Optional[GuidelineRetriever]
    matcher: RecommendationMatcher
    ehr_client: EHRClient
    task_generator: TaskGenerator


_DEPENDENCIES: Optional[ToolDependencies] = None


def configure_tool_dependencies(deps: ToolDependencies) -> None:
    """Inject concrete dependencies for tool execution."""

    global _DEPENDENCIES
    _DEPENDENCIES = deps


def _require_dependencies() -> ToolDependencies:
    if _DEPENDENCIES is None:
        raise RuntimeError("Tool dependencies have not been configured.")
    return _DEPENDENCIES


def _convert_finding(finding: Finding) -> Dict[str, Any]:
    payload = asdict(finding)
    payload["type"] = payload.pop("finding_type", None)
    return payload


def _convert_guideline(chunk: GuidelineChunk, finding: Dict[str, Any]) -> Dict[str, Any]:
    data = asdict(chunk)
    data["finding_id"] = finding.get("id") or finding.get("finding_id")
    return data


def _convert_recommendation(rec: Recommendation, finding: Dict[str, Any]) -> Dict[str, Any]:
    payload = asdict(rec)
    payload["finding_id"] = finding.get("id") or finding.get("finding_id")
    return payload


def _log_decision(
    state: AgentState,
    step: str,
    *,
    input_payload: Dict[str, Any],
    output_payload: Dict[str, Any],
    duration_ms: float,
) -> None:
    trace_payload = {
        "input": input_payload,
        "output": output_payload,
        "duration_ms": duration_ms,
    }
    state.add_decision_step(step, trace_payload)


def parse_report_tool(state: AgentState) -> Tuple[AgentState, List[Dict[str, Any]]]:
    """Extract structured findings from the raw radiology report."""

    deps = _require_dependencies()
    start = perf_counter()
    state.status = "parsing"
    correlation_context = {"session_id": state.session_id, "step": "parse_report"}
    try:
        findings = deps.parser.parse(state.report_text)
        finding_dicts = []
        for finding in findings:
            finding_dict = _convert_finding(finding)
            finding_dict.setdefault("id", finding_dict.get("finding_id") or str(uuid4()))
            state.add_finding(finding_dict)
            finding_dicts.append(finding_dict)

        duration_ms = (perf_counter() - start) * 1000.0
        _LOGGER.info(
            "Parsed report into findings.",
            extra={"context": {**correlation_context, "finding_count": len(finding_dicts)}},
        )
        _log_decision(
            state,
            "parse_report",
            input_payload={"report_id": state.report_id, "report_length": len(state.report_text)},
            output_payload={"findings": finding_dicts},
            duration_ms=duration_ms,
        )
        return state, finding_dicts
    except Exception as exc:  # pragma: no cover - defensive logging
        state.status = "failed"
        state.error = str(exc)
        duration_ms = (perf_counter() - start) * 1000.0
        _LOGGER.error(
            "Failed to parse radiology report.",
            extra={"context": {**correlation_context, "error": str(exc)}},
        )
        _log_decision(
            state,
            "parse_report",
            input_payload={"report_id": state.report_id},
            output_payload={"error": str(exc)},
            duration_ms=duration_ms,
        )
        raise


def retrieve_guidelines_tool(
    state: AgentState,
    finding: Dict[str, Any],
) -> Tuple[AgentState, List[Dict[str, Any]]]:
    """Retrieve relevant guideline chunks for a clinical finding."""

    deps = _require_dependencies()
    start = perf_counter()
    state.status = "retrieving"
    correlation_context = {
        "session_id": state.session_id,
        "step": "retrieve_guidelines",
        "finding_id": finding.get("id"),
    }

    # If no retriever available (Ollama mode), return empty guidelines
    if deps.retriever is None:
        _LOGGER.info(
            "Guideline retrieval skipped (no retriever configured).",
            extra={"context": correlation_context},
        )
        return state, []

    try:
        chunks = deps.retriever.retrieve(finding, top_k=5)
        chunk_dicts = [_convert_guideline(chunk, finding) for chunk in chunks]
        for chunk_dict in chunk_dicts:
            state.add_guideline(chunk_dict)

        duration_ms = (perf_counter() - start) * 1000.0
        _LOGGER.info(
            "Retrieved guideline chunks.",
            extra={"context": {**correlation_context, "count": len(chunk_dicts)}},
        )
        _log_decision(
            state,
            "retrieve_guidelines",
            input_payload={"finding": finding},
            output_payload={"guidelines": chunk_dicts},
            duration_ms=duration_ms,
        )
        return state, chunk_dicts
    except Exception as exc:  # pragma: no cover - defensive logging
        state.status = "failed"
        state.error = str(exc)
        duration_ms = (perf_counter() - start) * 1000.0
        _LOGGER.error(
            "Guideline retrieval failed.",
            extra={"context": {**correlation_context, "error": str(exc)}},
        )
        _log_decision(
            state,
            "retrieve_guidelines",
            input_payload={"finding": finding},
            output_payload={"error": str(exc)},
            duration_ms=duration_ms,
        )
        raise


def match_recommendation_tool(
    state: AgentState,
    finding: Dict[str, Any],
    guidelines: List[Dict[str, Any]],
) -> Tuple[AgentState, Dict[str, Any]]:
    """Match a finding to the most appropriate guideline recommendation."""

    deps = _require_dependencies()
    start = perf_counter()
    state.status = "matching"
    correlation_context = {
        "session_id": state.session_id,
        "step": "match_recommendation",
        "finding_id": finding.get("id"),
    }
    try:
        guideline_chunks = []
        for chunk_dict in guidelines:
            payload = dict(chunk_dict)
            payload.pop("finding_id", None)
            guideline_chunks.append(GuidelineChunk(**payload))
    except TypeError as exc:
        _LOGGER.error(
            "Invalid guideline payload supplied to matcher.",
            extra={"context": {**correlation_context, "error": str(exc)}},
        )
        raise

    try:
        recommendation = deps.matcher.match(
            finding,
            guideline_chunks,
            patient_context=state.patient_context,
        )
        recommendation_dict = _convert_recommendation(recommendation, finding)
        recommendation_dict.setdefault("id", str(uuid4()))
        recommendation_dict.setdefault("requires_human_review", False)

        state.add_recommendation(recommendation_dict)

        duration_ms = (perf_counter() - start) * 1000.0
        _LOGGER.info(
            "Generated recommendation from guidelines.",
            extra={
                "context": {
                    **correlation_context,
                    "follow_up": recommendation_dict.get("follow_up_type"),
                    "urgency": recommendation_dict.get("urgency"),
                }
            },
        )
        _log_decision(
            state,
            "match_recommendation",
            input_payload={"finding": finding, "guideline_count": len(guidelines)},
            output_payload={"recommendation": recommendation_dict},
            duration_ms=duration_ms,
        )
        return state, recommendation_dict
    except Exception as exc:  # pragma: no cover - defensive logging
        state.status = "failed"
        state.error = str(exc)
        duration_ms = (perf_counter() - start) * 1000.0
        _LOGGER.error(
            "Failed to match recommendation.",
            extra={"context": {**correlation_context, "error": str(exc)}},
        )
        _log_decision(
            state,
            "match_recommendation",
            input_payload={"finding": finding},
            output_payload={"error": str(exc)},
            duration_ms=duration_ms,
        )
        raise


def validate_safety_tool(
    state: AgentState,
    recommendation: Dict[str, Any],
) -> Tuple[AgentState, bool]:
    """Validate that a recommendation is safe and guideline-aligned."""

    start = perf_counter()
    state.status = "validating"
    correlation_context = {
        "session_id": state.session_id,
        "step": "validate_safety",
        "recommendation_follow_up": recommendation.get("follow_up_type"),
    }

    try:
        concerns: List[str] = []
        finding_id = recommendation.get("finding_id")
        finding = next((item for item in state.findings if item.get("id") == finding_id), None)

        size_mm = (finding or {}).get("size_mm")
        if isinstance(size_mm, (int, float)) and size_mm > 30:
            concerns.append("Lesion larger than 30mm")

        characteristics = set((finding or {}).get("characteristics") or [])
        if any(term in characteristics for term in {"spiculated", "irregular"}):
            concerns.append("Suspicious imaging characteristics")

        urgency = recommendation.get("urgency", "").lower()
        if urgency in {"urgent", "stat"}:
            concerns.append(f"Urgency marked as {urgency}")

        guideline_citation = recommendation.get("citation")
        if not guideline_citation:
            concerns.append("Missing guideline citation")

        is_safe = not concerns
        recommendation.setdefault("requires_human_review", False)
        if concerns:
            recommendation["requires_human_review"] = True

        validation_result = {
            "is_safe": is_safe,
            "concerns": concerns,
            "requires_human_review": recommendation["requires_human_review"],
        }
        recommendation["concerns"] = concerns

        duration_ms = (perf_counter() - start) * 1000.0
        _LOGGER.info(
            "Validated recommendation safety.",
            extra={
                "context": {
                    **correlation_context,
                    "is_safe": is_safe,
                    "concern_count": len(concerns),
                }
            },
        )
        _log_decision(
            state,
            "validate_safety",
            input_payload={"recommendation": recommendation},
            output_payload=validation_result,
            duration_ms=duration_ms,
        )
        return state, is_safe
    except Exception as exc:  # pragma: no cover - defensive logging
        state.status = "failed"
        state.error = str(exc)
        duration_ms = (perf_counter() - start) * 1000.0
        _LOGGER.error(
            "Recommendation safety validation failed.",
            extra={"context": {**correlation_context, "error": str(exc)}},
        )
        _log_decision(
            state,
            "validate_safety",
            input_payload={"recommendation": recommendation},
            output_payload={"error": str(exc)},
            duration_ms=duration_ms,
        )
        raise


def generate_task_tool(
    state: AgentState,
    recommendation: Dict[str, Any],
) -> Tuple[AgentState, str]:
    """Create a follow-up task in the EHR system."""

    deps = _require_dependencies()
    start = perf_counter()
    correlation_context = {
        "session_id": state.session_id,
        "step": "generate_task",
        "recommendation_follow_up": recommendation.get("follow_up_type"),
    }

    try:
        service_request = deps.task_generator.build_order(state, recommendation)
        order_id = deps.ehr_client.create_service_request(service_request)
        state.tasks_generated.append(order_id)

        duration_ms = (perf_counter() - start) * 1000.0
        _LOGGER.info(
            "Generated follow-up task.",
            extra={"context": {**correlation_context, "order_id": order_id}},
        )
        _log_decision(
            state,
            "generate_task",
            input_payload={"recommendation": recommendation},
            output_payload={"order_id": order_id},
            duration_ms=duration_ms,
        )
        return state, order_id
    except (ValidationError, TypeError, ValueError) as exc:
        state.status = "failed"
        state.error = str(exc)
        duration_ms = (perf_counter() - start) * 1000.0
        _LOGGER.error(
            "Failed to build ServiceRequest payload.",
            extra={"context": {**correlation_context, "error": str(exc)}},
        )
        _log_decision(
            state,
            "generate_task",
            input_payload={"recommendation": recommendation},
            output_payload={"error": str(exc)},
            duration_ms=duration_ms,
        )
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        state.status = "failed"
        state.error = str(exc)
        duration_ms = (perf_counter() - start) * 1000.0
        _LOGGER.error(
            "Failed to create follow-up task in EHR.",
            extra={"context": {**correlation_context, "error": str(exc)}},
        )
        _log_decision(
            state,
            "generate_task",
            input_payload={"recommendation": recommendation},
            output_payload={"error": str(exc)},
            duration_ms=duration_ms,
        )
        raise


TOOLS = {
    "parse_report": parse_report_tool,
    "retrieve_guidelines": retrieve_guidelines_tool,
    "match_recommendation": match_recommendation_tool,
    "validate_safety": validate_safety_tool,
    "generate_task": generate_task_tool,
}
