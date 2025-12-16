from __future__ import annotations

"""Core orchestration logic for the AuDRA agent."""

import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from src.agent.prompts import REACT_PROMPT, SYSTEM_PROMPT, format_prompt
from src.agent.state import AgentState, StateManager
from src.agent.tools import (
    TOOLS,
    ToolDependencies,
    TaskGenerator,
    configure_tool_dependencies,
    generate_task_tool,
    match_recommendation_tool,
    parse_report_tool,
    retrieve_guidelines_tool,
    validate_safety_tool,
)
from src.guidelines.matcher import RecommendationMatcher
from src.guidelines.retriever import GuidelineRetriever
from src.parsers.report_parser import ReportParser
from src.services.ehr_client import EHRClient
from src.services.nim_embeddings import EmbeddingClient
from src.services.nim_llm import NemotronClient, NIMServiceError
from src.services.ollama_llm import OllamaClient
from src.services.vector_store import VectorStore
from src.utils.logger import get_logger


_LOGGER = get_logger("audra.agent.orchestrator")


@dataclass
class ProcessingResult:
    """Aggregate outcome from processing a radiology report."""

    status: Literal["success", "no_findings", "requires_review", "error"]
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    decision_trace: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    message: Optional[str] = None
    state: Optional[AgentState] = None


class AuDRAAgent:
    """Agent implementing a multi-step ReAct workflow over radiology reports."""

    def __init__(
        self,
        llm_client: NemotronClient | OllamaClient,
        embedding_client: Optional[EmbeddingClient],
        vector_store: Optional[VectorStore],
        ehr_client: EHRClient,
    ) -> None:
        self._llm = llm_client
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._ehr_client = ehr_client

        self._parser = ReportParser()

        # Only initialize retriever if we have embeddings and vector store
        if embedding_client and vector_store:
            self._retriever = GuidelineRetriever(embedding_client, vector_store)
        else:
            self._retriever = None
            _LOGGER.info("Running without guideline retrieval (no embedding/vector store).")

        self._matcher = RecommendationMatcher(llm_client)
        self._task_generator = TaskGenerator()

        configure_tool_dependencies(
            ToolDependencies(
                parser=self._parser,
                retriever=self._retriever,
                matcher=self._matcher,
                ehr_client=self._ehr_client,
                task_generator=self._task_generator,
            )
        )

        self.max_iterations = 20

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process_report(
        self,
        report_text: str,
        patient_context: Optional[Dict[str, Any]] = None,
        report_id: Optional[str] = None,
    ) -> ProcessingResult:
        """Run the agent workflow over a radiology report."""

        start_time = perf_counter()
        session_id = str(uuid4())
        resolved_report_id = report_id or str(uuid4())
        state = AgentState(
            session_id=session_id,
            report_id=resolved_report_id,
            report_text=report_text,
            patient_context=patient_context,
            status="initialized",
        )
        StateManager.save_state(state)

        try:
            # Step 1: Parse the report for findings.
            state, findings = self._execute_with_retries(parse_report_tool, state)
            _LOGGER.info(
                "Report parsed.",
                extra={"context": {"session_id": state.session_id, "finding_count": len(findings)}},
            )

            if not findings:
                state.status = "completed"
                processing_time_ms = (perf_counter() - start_time) * 1000.0
                return ProcessingResult(
                    status="no_findings",
                    findings=state.findings,
                    recommendations=state.recommendations,
                    tasks=state.tasks_generated,
                    decision_trace=state.decision_trace,
                    processing_time_ms=processing_time_ms,
                    message="No actionable findings detected",
                    state=state,
                )

            # Step 2: Process each finding sequentially.
            for finding in findings:
                try:
                    state, guidelines = self._execute_with_retries(
                        retrieve_guidelines_tool,
                        state,
                        finding,
                    )
                    state, recommendation = self._execute_with_retries(
                        match_recommendation_tool,
                        state,
                        finding,
                        guidelines,
                    )
                    state, is_safe = self._execute_with_retries(
                        validate_safety_tool,
                        state,
                        recommendation,
                    )

                    follow_up_type = str(recommendation.get("follow_up_type") or "").strip()
                    if follow_up_type.lower() == "none":
                        continue

                    if is_safe and not recommendation.get("requires_human_review"):
                        state, order_id = self._execute_with_retries(
                            generate_task_tool,
                            state,
                            recommendation,
                        )
                        _LOGGER.info(
                            "Created follow-up order.",
                            extra={
                                "context": {
                                    "session_id": state.session_id,
                                    "order_id": order_id,
                                    "finding_id": finding.get("id"),
                                }
                            },
                        )
                    else:
                        state.status = "requires_review"
                        _LOGGER.warning(
                            "Recommendation flagged for human review.",
                            extra={
                                "context": {
                                    "session_id": state.session_id,
                                    "finding": finding,
                                    "concerns": recommendation.get("concerns"),
                                }
                            },
                        )
                except Exception as exc:
                    state.status = "failed"
                    state.error = str(exc)
                    _LOGGER.error(
                        "Processing failed for finding.",
                        extra={
                            "context": {
                                "session_id": state.session_id,
                                "finding": finding,
                                "error": str(exc),
                            }
                        },
                    )
                    processing_time_ms = (perf_counter() - start_time) * 1000.0
                    return ProcessingResult(
                        status="error",
                        findings=state.findings,
                        recommendations=state.recommendations,
                        tasks=state.tasks_generated,
                        decision_trace=state.decision_trace,
                        processing_time_ms=processing_time_ms,
                        message="Agent encountered an error while processing findings.",
                        state=state,
                    )

            if state.status not in {"requires_review", "failed"}:
                state.status = "completed"

            processing_time_ms = (perf_counter() - start_time) * 1000.0
            return ProcessingResult(
                status="success" if state.status == "completed" else state.status,  # type: ignore[arg-type]
                findings=state.findings,
                recommendations=state.recommendations,
                tasks=state.tasks_generated,
                decision_trace=state.decision_trace,
                processing_time_ms=processing_time_ms,
                state=state,
            )
        except Exception as exc:
            state.status = "failed"
            state.error = str(exc)
            _LOGGER.error(
                "Unexpected agent failure.",
                extra={"context": {"session_id": state.session_id, "error": str(exc)}},
            )
            processing_time_ms = (perf_counter() - start_time) * 1000.0
            return ProcessingResult(
                status="error",
                findings=state.findings,
                recommendations=state.recommendations,
                tasks=state.tasks_generated,
                decision_trace=state.decision_trace,
                processing_time_ms=processing_time_ms,
                message="Unexpected agent failure occurred.",
                state=state,
            )
        finally:
            StateManager.save_state(state)

    # ------------------------------------------------------------------ #
    # ReAct support
    # ------------------------------------------------------------------ #

    def _react_step(self, state: AgentState) -> Tuple[AgentState, str, Dict[str, Any]]:
        """Execute a single ReAct iteration driven by the LLM."""

        summary = self._summarize_state(state)
        prompt = format_prompt(REACT_PROMPT, state_summary=summary)

        try:
            response = self._llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        except NIMServiceError as exc:
            _LOGGER.error(
                "LLM call failed during ReAct step.",
                extra={"context": {"session_id": state.session_id, "error": str(exc)}},
            )
            raise

        action_name, action_input = self._parse_react_response(response)
        if action_name == "FINISH" or action_name not in TOOLS:
            return state, "FINISH", {"message": "Agent elected to finish."}

        handler = TOOLS[action_name]
        try:
            if action_name == "parse_report":
                return handler(state)
            if action_name == "retrieve_guidelines":
                finding = self._resolve_finding(state, action_input.get("finding_id"))
                return handler(state, finding)
            if action_name == "match_recommendation":
                finding = self._resolve_finding(state, action_input.get("finding_id"))
                guidelines = action_input.get("guidelines") or self._resolve_guidelines(
                    state, finding.get("id")
                )
                return handler(state, finding, guidelines)
            if action_name == "validate_safety":
                recommendation = self._resolve_recommendation(
                    state, action_input.get("recommendation_id")
                )
                return handler(state, recommendation)
            if action_name == "generate_task":
                recommendation = self._resolve_recommendation(
                    state, action_input.get("recommendation_id")
                )
                return handler(state, recommendation)
        except Exception as exc:
            _LOGGER.error(
                "Tool execution failed during ReAct step.",
                extra={
                    "context": {
                        "session_id": state.session_id,
                        "action": action_name,
                        "input": action_input,
                        "error": str(exc),
                    }
                },
            )
            raise

        _LOGGER.warning(
            "No handler matched LLM action; defaulting to FINISH.",
            extra={"context": {"session_id": state.session_id, "action": action_name}},
        )
        return state, "FINISH", {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _execute_with_retries(self, tool, state: AgentState, *args: Any, retries: int = 2):
        attempts = 0
        last_error: Optional[Exception] = None
        while attempts < retries:
            try:
                result = tool(state, *args)
                StateManager.save_state(state)
                return result
            except Exception as exc:
                attempts += 1
                last_error = exc
                _LOGGER.warning(
                    "Tool execution failed; attempt %d/%d.",
                    attempts,
                    retries,
                    extra={
                        "context": {
                            "session_id": state.session_id,
                            "tool": getattr(tool, "__name__", str(tool)),
                            "error": str(exc),
                        }
                    },
                )
                if attempts >= retries:
                    raise
        raise last_error if last_error else RuntimeError("Retry attempts exhausted.")

    def _summarize_state(self, state: AgentState) -> str:
        summary_parts = [
            f"Session: {state.session_id}",
            f"Status: {state.status}",
            f"Findings: {len(state.findings)}",
            f"Recommendations: {len(state.recommendations)}",
            f"Tasks: {len(state.tasks_generated)}",
        ]
        if state.decision_trace:
            summary_parts.append(f"Last Step: {state.decision_trace[-1]['step']}")
        return "\n".join(summary_parts)

    def _parse_react_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        action_name = "FINISH"
        action_input: Dict[str, Any] = {}

        for line in response.splitlines():
            normalized = line.strip()
            if normalized.lower().startswith("action:"):
                action_name = normalized.split(":", 1)[1].strip()
            elif normalized.lower().startswith("action input:"):
                raw_input = normalized.split(":", 1)[1].strip()
                try:
                    action_input = json.loads(raw_input) if raw_input else {}
                except json.JSONDecodeError:
                    action_input = {"value": raw_input}
        return action_name, action_input

    @staticmethod
    def _resolve_finding(state: AgentState, finding_id: Optional[str]) -> Dict[str, Any]:
        if not finding_id:
            raise ValueError("Action input must include 'finding_id'.")
        for finding in state.findings:
            if finding.get("id") == finding_id:
                return finding
        raise ValueError(f"Finding '{finding_id}' not found in state.")

    @staticmethod
    def _resolve_guidelines(state: AgentState, finding_id: Optional[str]) -> List[Dict[str, Any]]:
        if not finding_id:
            raise ValueError("Action input must include 'finding_id'.")
        return [
            guideline
            for guideline in state.retrieved_guidelines
            if guideline.get("finding_id") == finding_id
        ]

    @staticmethod
    def _resolve_recommendation(
        state: AgentState, recommendation_id: Optional[str]
    ) -> Dict[str, Any]:
        if not recommendation_id:
            raise ValueError("Action input must include 'recommendation_id'.")
        for recommendation in state.recommendations:
            if recommendation.get("id") == recommendation_id:
                return recommendation
        raise ValueError(f"Recommendation '{recommendation_id}' not found in state.")
