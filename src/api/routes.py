"""FastAPI route definitions for the AuDRA API."""

import asyncio
from datetime import date, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.agent.orchestrator import AuDRAAgent, ProcessingResult
from src.agent.state import AgentState, StateManager
from src.api.models import (
    BatchProcessRequest,
    ErrorResponse,
    HealthResponse,
    ProcessReportRequest,
    ProcessReportResponse,
    RecommendationResponse,
    TaskResponse,
)
from src.api.models import FindingResponse, HealthStatus
from src.utils.logger import clear_correlation_id, get_logger, set_correlation_id


logger = get_logger("audra.api.routes")

router = APIRouter(
    tags=["Reports", "Health", "Metrics"],
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse}},
)

limiter = Limiter(key_func=get_remote_address)


# --------------------------------------------------------------------------- #
# Metrics tracking
# --------------------------------------------------------------------------- #


class _MetricsTracker:
    """In-memory tracker for API metrics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._total_reports = 0
        self._total_processing_time_ms = 0.0
        self._findings_detected = 0
        self._tasks_created = 0
        self._error_count = 0

    def record_response(self, response: ProcessReportResponse) -> None:
        with self._lock:
            self._total_reports += 1
            self._total_processing_time_ms += response.processing_time_ms
            self._findings_detected += len(response.findings)
            self._tasks_created += len(response.tasks)
            if response.status == "error":
                self._error_count += 1

    def record_error(self) -> None:
        with self._lock:
            self._total_reports += 1
            self._error_count += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            average_time = (
                self._total_processing_time_ms / self._total_reports if self._total_reports else 0.0
            )
            error_rate = (
                self._error_count / self._total_reports if self._total_reports else 0.0
            )
            return {
                "total_reports_processed": self._total_reports,
                "average_processing_time_ms": average_time,
                "total_findings_detected": self._findings_detected,
                "total_tasks_created": self._tasks_created,
                "error_rate": error_rate,
            }


_metrics = _MetricsTracker()


# --------------------------------------------------------------------------- #
# Dependency helpers
# --------------------------------------------------------------------------- #


def get_agent(request: Request) -> AuDRAAgent:
    """Return the shared AuDRA agent instance."""

    agent: Optional[AuDRAAgent] = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AuDRA agent is not initialised.",
        )
    return agent


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #


def _clamp_confidence(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _build_finding(payload: Dict[str, Any]) -> FindingResponse:
    finding_id = payload.get("id") or payload.get("finding_id") or str(uuid4())
    characteristics = payload.get("characteristics") or []
    if not isinstance(characteristics, list):
        characteristics = [str(characteristics)]
    return FindingResponse(
        finding_id=str(finding_id),
        type=str(payload.get("type") or payload.get("finding_type") or "unknown"),
        size_mm=float(payload["size_mm"]) if payload.get("size_mm") is not None else None,
        location=str(payload.get("location") or payload.get("site") or "unspecified"),
        characteristics=[str(item) for item in characteristics],
        confidence=_clamp_confidence(payload.get("confidence"), default=0.0),
    )


def _normalize_urgency(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"routine", "priority", "urgent", "stat"}:
            return normalized
    return "routine"


def _build_recommendation(payload: Dict[str, Any]) -> RecommendationResponse:
    return RecommendationResponse(
        recommendation_id=str(payload.get("id") or payload.get("recommendation_id") or uuid4()),
        follow_up_type=str(payload.get("follow_up_type") or "Follow-up procedure"),
        timeframe_months=payload.get("timeframe_months"),
        urgency=_normalize_urgency(payload.get("urgency")),
        reasoning=str(payload.get("reasoning") or payload.get("explanation") or "Reasoning unavailable."),
        citation=str(payload.get("citation") or "Unspecified guideline."),
        confidence=_clamp_confidence(payload.get("confidence"), default=0.5),
    )


def _build_tasks(task_payloads: List[Dict[str, Any]]) -> List[TaskResponse]:
    tasks: List[TaskResponse] = []
    fallback_date = date.today() + timedelta(days=1)
    for payload in task_payloads:
        raw_date = payload.get("scheduled_date") or fallback_date
        if isinstance(raw_date, str):
            try:
                scheduled_date = date.fromisoformat(raw_date)
            except ValueError:
                scheduled_date = fallback_date
        elif isinstance(raw_date, date):
            scheduled_date = raw_date
        else:
            scheduled_date = fallback_date
        if scheduled_date <= date.today():
            scheduled_date = date.today() + timedelta(days=1)

        procedure = payload.get("procedure") or payload.get("follow_up_type") or "Follow-up procedure"
        reason = payload.get("reason") or payload.get("reasoning") or "Automated recommendation generated by AuDRA."
        order_id = payload.get("order_id")

        tasks.append(
            TaskResponse(
                task_id=str(payload.get("task_id") or order_id or uuid4()),
                procedure=str(procedure),
                scheduled_date=scheduled_date,
                reason=str(reason),
                order_id=str(order_id) if order_id else None,
            )
        )
    return tasks


def _requires_human_review(
    status_text: str,
    recommendations: List[Dict[str, Any]],
) -> bool:
    if status_text == "requires_review":
        return True
    for rec in recommendations:
        if rec.get("requires_human_review"):
            return True
        concerns = rec.get("concerns")
        if isinstance(concerns, list) and concerns:
            return True
    return False


def _build_response(result: ProcessingResult, request_payload: ProcessReportRequest) -> ProcessReportResponse:
    state = result.state
    if state is None:
        raise ValueError("Processing result did not include state information.")

    findings = [_build_finding(item) for item in result.findings]
    raw_recommendations = result.recommendations
    recommendations = [_build_recommendation(item) for item in raw_recommendations]
    tasks = _build_tasks(result.tasks)

    response = ProcessReportResponse(
        status=result.status,
        session_id=state.session_id,
        report_id=state.report_id,
        findings=findings,
        recommendations=recommendations,
        tasks=tasks,
        processing_time_ms=result.processing_time_ms,
        message=result.message,
        requires_human_review=_requires_human_review(result.status, raw_recommendations),
        decision_trace=result.decision_trace or (state.decision_trace or None),
    )
    return response


def _build_response_from_state(state: AgentState) -> ProcessReportResponse:
    findings = [_build_finding(item) for item in state.findings]
    recommendations = [_build_recommendation(item) for item in state.recommendations]
    tasks = _build_tasks(state.tasks_generated)
    status_map = {
        "completed": "success",
        "failed": "error",
        "parsing": "requires_review",
        "retrieving": "requires_review",
        "matching": "requires_review",
        "validating": "requires_review",
        "initialized": "requires_review",
    }
    status_text = state.status if state.status in {"success", "no_findings", "requires_review", "error"} else status_map.get(state.status, "requires_review")

    return ProcessReportResponse(
        status=status_text,  # type: ignore[arg-type]
        session_id=state.session_id,
        report_id=state.report_id,
        findings=findings,
        recommendations=recommendations,
        tasks=tasks,
        processing_time_ms=0.0,
        message=state.error,
        requires_human_review=_requires_human_review(status_text, state.recommendations),
        decision_trace=state.decision_trace or None,
    )


async def _run_agent(
    agent: AuDRAAgent,
    payload: ProcessReportRequest,
    correlation_id: str,
) -> ProcessReportResponse:
    set_correlation_id(correlation_id)
    logger.info(
        "Processing report request.",
        extra={
            "context": {
                "correlation_id": correlation_id,
                "has_patient_context": bool(payload.patient_context),
                "report_id": payload.report_id,
            }
        },
    )

    try:
        result = await asyncio.to_thread(
            agent.process_report,
            payload.report_text,
            patient_id=payload.patient_id,
            patient_context=payload.patient_context,
            report_id=payload.report_id,
        )
        response = _build_response(result, payload)
        logger.info(
            "Report processed.",
            extra={
                "context": {
                    "correlation_id": correlation_id,
                    "status": response.status,
                    "findings": len(response.findings),
                    "recommendations": len(response.recommendations),
                }
            },
        )
        _metrics.record_response(response)
        return response
    except Exception as exc:
        _metrics.record_error()
        logger.error(
            "Report processing failed.",
            extra={
                "context": {
                    "correlation_id": correlation_id,
                    "error": str(exc),
                }
            },
        )
        raise
    finally:
        clear_correlation_id()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.post(
    "/process-report",
    response_model=ProcessReportResponse,
    summary="Process a radiology report.",
    description=(
        "Process a single radiology report to extract findings, match guidelines, "
        "generate recommendations, and optionally create follow-up tasks."
    ),
)
@limiter.limit("10/minute")
async def process_report_endpoint(
    request: Request,
    payload: ProcessReportRequest = Body(...),
    agent: AuDRAAgent = Depends(get_agent),
    correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
) -> ProcessReportResponse:
    correlation = correlation_id or str(uuid4())
    request.state.correlation_id = correlation
    return await _run_agent(agent, payload, correlation)


@router.post(
    "/batch-process",
    response_model=List[ProcessReportResponse],
    summary="Process multiple reports in batch.",
    description="Process up to 10 reports concurrently and return the results.",
)
async def batch_process_endpoint(
    request: BatchProcessRequest,
    agent: AuDRAAgent = Depends(get_agent),
) -> List[ProcessReportResponse]:
    async def _process_single(report_request: ProcessReportRequest) -> ProcessReportResponse:
        correlation = str(uuid4())
        try:
            return await _run_agent(agent, report_request, correlation)
        except Exception as exc:
            logger.error(
                "Failed to process report within batch.",
                extra={
                    "context": {
                        "correlation_id": correlation,
                        "error": str(exc),
                    }
                },
            )
            raise

    tasks = [_process_single(report) for report in request.reports]
    return await asyncio.gather(*tasks)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check.",
    description="Return health status for the API and dependent services.",
)
async def health_check_endpoint(request: Request) -> HealthResponse:
    app = request.app
    version = getattr(app.state, "version", "unknown")

    def _check(service_name: str, instance: Any) -> HealthStatus:
        if instance is None:
            return "unhealthy"
        try:
            if service_name == "vector_store" and hasattr(instance, "_client"):
                if not instance._client.ping():  # type: ignore[attr-defined]
                    return "degraded"
            return "healthy"
        except Exception as exc:
            logger.warning(
                "Service health check failed.",
                extra={"context": {"service": service_name, "error": str(exc)}},
            )
            return "degraded"

    services: Dict[str, HealthStatus] = {
        "llm": _check("llm", getattr(app.state, "llm_client", None)),
        "embeddings": _check("embeddings", getattr(app.state, "embedding_client", None)),
        "vector_store": _check("vector_store", getattr(app.state, "vector_store", None)),
        "ehr": _check("ehr", getattr(app.state, "ehr_client", None)),
    }

    aggregate_status: HealthStatus = "healthy"
    if "unhealthy" in services.values():
        aggregate_status = "unhealthy"
    elif "degraded" in services.values():
        aggregate_status = "degraded"

    return HealthResponse(status=aggregate_status, services=services, version=version)


@router.get(
    "/session/{session_id}",
    response_model=ProcessReportResponse,
    summary="Retrieve a processing session.",
    description="Return the results of a previously processed report using its session identifier.",
)
async def get_session_endpoint(session_id: str) -> ProcessReportResponse:
    try:
        state = StateManager.load_state(session_id)
    except KeyError as exc:
        logger.warning(
            "Requested session not found.",
            extra={"context": {"session_id": session_id}},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    response = _build_response_from_state(state)
    return response


@router.get(
    "/metrics",
    summary="Retrieve processing metrics.",
    description=(
        "Return cumulative metrics including total reports processed, average processing time, "
        "findings detected, tasks created, and error rate."
    ),
)
async def get_metrics_endpoint() -> Dict[str, Any]:
    return _metrics.snapshot()


# --------------------------------------------------------------------------- #
# Error handlers
# --------------------------------------------------------------------------- #


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    logger.warning(
        "Rate limit exceeded.",
        extra={
            "context": {
                "path": request.url.path,
                "detail": str(exc.detail),
            }
        },
    )
    response = ErrorResponse(
        error_code="RATE_LIMIT_EXCEEDED",
        message="Too many requests. Please slow down.",
        details={"limit": str(exc.detail)},
    )
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=response.model_dump(mode="json"),
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning(
        "Validation error encountered.",
        extra={"context": {"path": request.url.path, "error": str(exc)}},
    )
    response = ErrorResponse(
        error_code="VALIDATION_ERROR",
        message=str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=response.model_dump(mode="json"),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception during request.",
        extra={"context": {"path": request.url.path, "error": str(exc)}},
    )
    response = ErrorResponse(
        error_code="INTERNAL_ERROR",
        message="An unexpected error occurred.",
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response.model_dump(mode="json"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach route-level exception handlers to the FastAPI application."""

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, general_exception_handler)
