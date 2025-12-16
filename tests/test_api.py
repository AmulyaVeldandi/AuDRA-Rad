from __future__ import annotations

from datetime import date, timedelta
import sys
import types
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from src.agent.orchestrator import ProcessingResult
from src.agent.state import AgentState, StateManager
from src.api import app as app_module
from src.api.app import create_app
from src.api.routes import (
    get_agent,
    limiter,
    _build_finding,
    _build_tasks,
    _clamp_confidence,
    health_check_endpoint,
)

if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")

    class _OpenAIBaseError(Exception):
        pass

    class APIConnectionError(_OpenAIBaseError):
        pass

    class APIError(_OpenAIBaseError):
        pass

    class APITimeoutError(_OpenAIBaseError):
        pass

    class OpenAIError(_OpenAIBaseError):
        pass

    class RateLimitError(_OpenAIBaseError):
        pass

    class OpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    openai_module.APIConnectionError = APIConnectionError
    openai_module.APIError = APIError
    openai_module.APITimeoutError = APITimeoutError
    openai_module.OpenAIError = OpenAIError
    openai_module.RateLimitError = RateLimitError
    openai_module.OpenAI = OpenAI

    sys.modules["openai"] = openai_module

if "jsonschema" not in sys.modules:
    jsonschema_module = types.ModuleType("jsonschema")

    class ValidationError(Exception):
        pass

    def validate(instance: Any, schema: Any) -> None:
        return None

    jsonschema_module.ValidationError = ValidationError
    jsonschema_module.validate = validate
    sys.modules["jsonschema"] = jsonschema_module

if "boto3" not in sys.modules:
    boto3_module = types.ModuleType("boto3")

    class Session:
        def __init__(self, region_name: str | None = None) -> None:
            self.region_name = region_name

        def get_credentials(self) -> Any:
            class _Creds:
                access_key = "test"
                secret_key = "test"
                token = None

            return _Creds()

    boto3_module.Session = Session
    sys.modules["boto3"] = boto3_module

if "botocore" not in sys.modules:
    botocore_module = types.ModuleType("botocore")
    sys.modules["botocore"] = botocore_module

if "botocore.exceptions" not in sys.modules:
    exceptions_module = types.ModuleType("botocore.exceptions")

    class BotoCoreError(Exception):
        pass

    class NoCredentialsError(Exception):
        pass

    exceptions_module.BotoCoreError = BotoCoreError
    exceptions_module.NoCredentialsError = NoCredentialsError
    sys.modules["botocore.exceptions"] = exceptions_module

if "opensearchpy" not in sys.modules:
    opensearch_module = types.ModuleType("opensearchpy")

    class AWSV4SignerAuth:
        def __init__(self, credentials: Any, region: str, service: str = "aoss") -> None:
            self.credentials = credentials
            self.region = region
            self.service = service

    class RequestsHttpConnection:
        pass

    class OpenSearch:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.indices = types.SimpleNamespace(
                exists=lambda index: True,
                create=lambda **kwargs: None,
            )

        def index(self, *args: Any, **kwargs: Any) -> None:
            return None

        def search(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"hits": {"hits": []}}

    def helpers_bulk(*args: Any, **kwargs: Any) -> tuple[list[object], list[object]]:
        return ([], [])

    opensearch_module.AWSV4SignerAuth = AWSV4SignerAuth
    opensearch_module.OpenSearch = OpenSearch
    opensearch_module.RequestsHttpConnection = RequestsHttpConnection
    opensearch_module.helpers = types.SimpleNamespace(bulk=helpers_bulk)

    sys.modules["opensearchpy"] = opensearch_module

if "opensearchpy.exceptions" not in sys.modules:
    exceptions_module = types.ModuleType("opensearchpy.exceptions")

    class OpenSearchException(Exception):
        pass

    class TransportError(Exception):
        pass

    class ConnectionError(Exception):
        pass

    exceptions_module.OpenSearchException = OpenSearchException
    exceptions_module.TransportError = TransportError
    exceptions_module.ConnectionError = ConnectionError
    sys.modules["opensearchpy.exceptions"] = exceptions_module

if "tqdm.auto" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm.auto")

    def tqdm(iterable: Any, *args: Any, **kwargs: Any) -> Any:
        return iterable

    tqdm_module.tqdm = tqdm
    sys.modules["tqdm.auto"] = tqdm_module

if "slowapi" not in sys.modules:
    slowapi_module = types.ModuleType("slowapi")

    class _DummyLimiter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def limit(self, *args: Any, **kwargs: Any):
            def decorator(func):
                return func

            return decorator

    slowapi_module.Limiter = _DummyLimiter

    errors_module = types.ModuleType("slowapi.errors")

    class _DummyRateLimitExceeded(Exception):
        def __init__(self, detail: str = "Rate limit exceeded") -> None:
            super().__init__(detail)
            self.detail = detail

    errors_module.RateLimitExceeded = _DummyRateLimitExceeded
    slowapi_module.errors = errors_module

    sys.modules["slowapi"] = slowapi_module
    sys.modules["slowapi.errors"] = errors_module

if "slowapi.middleware" not in sys.modules:
    middleware_module = types.ModuleType("slowapi.middleware")

    class _DummySlowAPIMiddleware:
        def __init__(self, app: Any, **kwargs: Any) -> None:
            self.app = app

    middleware_module.SlowAPIMiddleware = _DummySlowAPIMiddleware
    sys.modules["slowapi.middleware"] = middleware_module

if "slowapi.util" not in sys.modules:
    util_module = types.ModuleType("slowapi.util")

    def _dummy_remote_address(request: Any) -> str:
        return "test-client"

    util_module.get_remote_address = _dummy_remote_address
    sys.modules["slowapi.util"] = util_module

class _DummyRequest:
    def __init__(self, app: Any) -> None:
        self.app = app


class _Pinger:
    def __init__(self, should_succeed: bool) -> None:
        self._result = should_succeed

    def ping(self) -> bool:
        return self._result


def test_clamp_confidence_handles_invalid_input() -> None:
    assert _clamp_confidence("0.8") == 0.8
    assert _clamp_confidence(None, default=0.42) == 0.42
    assert _clamp_confidence(5) == 1.0
    assert _clamp_confidence(-2) == 0.0


def test_build_finding_normalises_characteristics() -> None:
    payload = {
        "finding_id": "abc",
        "type": "nodule",
        "size_mm": "6",
        "location": "RUL",
        "characteristics": "spiculated",
        "confidence": "0.75",
    }
    finding = _build_finding(payload)
    assert finding.finding_id == "abc"
    assert finding.size_mm == 6.0
    assert finding.characteristics == ["spiculated"]
    assert finding.confidence == pytest.approx(0.75)


def test_build_tasks_materialises_stored_payloads() -> None:
    future_date = date.today() + timedelta(days=5)
    payloads = [
        {
            "task_id": "task-1",
            "order_id": "ORD-1",
            "procedure": "CT Chest",
            "scheduled_date": future_date,
            "reason": "Guideline matched.",
        },
        {
            "order_id": None,
            "procedure": "MRI Abdomen",
            "scheduled_date": (date.today() + timedelta(days=60)).isoformat(),
            "reason": "Incidental finding.",
        },
    ]

    tasks = _build_tasks(payloads)
    assert len(tasks) == 2
    assert tasks[0].task_id == "task-1"
    assert tasks[0].scheduled_date == future_date
    assert tasks[0].order_id == "ORD-1"

    second = tasks[1]
    assert second.procedure == "MRI Abdomen"
    assert second.order_id is None
    assert second.scheduled_date > date.today()


@pytest.mark.asyncio
async def test_health_check_endpoint_reflects_service_states(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch heavy dependencies so the startup hook in create_app remains lightweight.
    class _DummyAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(app_module, "AuDRAAgent", _DummyAgent)
    monkeypatch.setattr(app_module, "NemotronClient", lambda: object())
    monkeypatch.setattr(app_module, "EmbeddingClient", lambda: object())

    class _DummyVectorStore:
        def __init__(self, index_name: str) -> None:
            self.index_name = index_name
            self._client = _Pinger(should_succeed=False)

    monkeypatch.setattr(app_module, "VectorStore", _DummyVectorStore)

    class _DummyEHR:
        use_mock = True

        def close(self) -> None:
            pass

    monkeypatch.setattr(app_module, "EHRClient", lambda use_mock=True: _DummyEHR())

    app = create_app()
    app.state.version = "test-version"
    app.state.llm_client = object()
    app.state.embedding_client = object()
    app.state.vector_store = _DummyVectorStore("medical_guidelines")
    app.state.ehr_client = None  # Force an unhealthy service.

    response = await health_check_endpoint(_DummyRequest(app))

    assert response.version == "test-version"
    assert response.status == "unhealthy"
    assert response.services["vector_store"] == "degraded"
    assert response.services["ehr"] == "unhealthy"


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Any]:
    class DummyAgent:
        def __init__(self) -> None:
            self.calls: list[Dict[str, Any]] = []
            self._session = 0
            self.base_date = date(2030, 1, 1)

        def process_report(
            self,
            report_text: str,
            patient_id: str | None = None,
            patient_context: Dict[str, Any] | None = None,
            report_id: str | None = None,
        ) -> ProcessingResult:
            self._session += 1
            scheduled_date = self.base_date + timedelta(days=self._session)
            recommendation = {
                "id": f"rec-{self._session}",
                "follow_up_type": "CT Chest",
                "timeframe_months": 3,
                "urgency": "priority",
                "reasoning": "Test reasoning",
                "citation": "Fleischner 2017",
                "confidence": 0.9,
            }
            finding = {
                "id": f"finding-{self._session}",
                "type": "nodule",
                "size_mm": 6.0,
                "location": "RUL",
                "confidence": 0.9,
            }
            task_payload = {
                "task_id": f"task-{self._session}",
                "order_id": f"order-{self._session}",
                "procedure": "CT Chest",
                "reason": "Stored reason",
                "scheduled_date": scheduled_date,
            }
            state = AgentState(
                session_id=f"session-{self._session}",
                report_id=report_id or f"report-{self._session}",
                report_text=report_text,
                patient_id=patient_id,
                patient_context=patient_context,
                findings=[finding],
                recommendations=[recommendation],
                tasks_generated=[task_payload],
                status="completed",  # type: ignore[arg-type]
            )
            StateManager.save_state(state)
            self.calls.append(
                {
                    "report_text": report_text,
                    "patient_id": patient_id,
                    "patient_context": patient_context,
                    "report_id": report_id,
                }
            )
            return ProcessingResult(
                status="success",
                findings=state.findings,
                recommendations=state.recommendations,
                tasks=state.tasks_generated,
                decision_trace=[],
                processing_time_ms=1.0,
                message="ok",
                state=state,
            )

    dummy_agent = DummyAgent()

    def _fake_init(app) -> None:
        app.state.agent = dummy_agent
        app.state.llm_client = object()
        app.state.embedding_client = object()
        app.state.vector_store = object()
        app.state.ehr_client = object()

    monkeypatch.setattr(app_module, "_initialise_services", _fake_init)
    monkeypatch.setattr(app_module, "_shutdown_services", lambda app: None)
    StateManager._memory_store.clear()  # type: ignore[attr-defined]
    test_app = create_app()
    test_app.dependency_overrides[get_agent] = lambda: dummy_agent
    limiter._storage.storage.clear()  # type: ignore[attr-defined]

    client = TestClient(test_app)
    yield client, dummy_agent
    client.close()
    StateManager._memory_store.clear()  # type: ignore[attr-defined]


def test_process_report_rate_limit_enforced(api_client: tuple[TestClient, object]) -> None:
    client, _ = api_client
    payload = {
        "report_text": "Findings " + ("lorem ipsum dolor sit amet. " * 3),
        "patient_id": "MRN-1",
    }

    for _ in range(10):
        response = client.post("/api/v1/process-report", json=payload)
        assert response.status_code == 200

    response = client.post("/api/v1/process-report", json=payload)
    assert response.status_code == 429


def test_process_report_includes_patient_id_and_persists_tasks(api_client: tuple[TestClient, object]) -> None:
    client, agent = api_client
    payload = {
        "report_text": "Extensive findings text describing nodules and impressions." * 2,
        "patient_id": "MRN-42",
    }

    response = client.post("/api/v1/process-report", json=payload)
    assert response.status_code == 200
    body = response.json()

    first_call = agent.calls[0]
    assert first_call["patient_id"] == "MRN-42"

    tasks = body["tasks"]
    expected_date = (agent.base_date + timedelta(days=1)).isoformat()
    assert tasks[0]["scheduled_date"] == expected_date

    session_id = body["session_id"]
    session_response = client.get(f"/api/v1/session/{session_id}")
    assert session_response.status_code == 200
    assert session_response.json()["tasks"] == tasks
