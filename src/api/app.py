"""FastAPI application factory for the AuDRA API."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import uuid4

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef, assignment]

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi.middleware import SlowAPIMiddleware

from src.agent.orchestrator import AuDRAAgent
from src.api.routes import register_exception_handlers, router
from src.services.ehr_client import EHRClient
from src.services.nim_embeddings import EmbeddingClient, NIMServiceError as EmbeddingServiceError
from src.services.nim_llm import NIMServiceError, NemotronClient
from src.services.ollama_llm import OllamaClient, OllamaServiceError
from src.services.ollama_embeddings import OllamaEmbeddingClient, OllamaEmbeddingError
from src.services.vector_store import VectorStore, VectorStoreError
from src.services.simple_vector_store import SimpleVectorStore
from src.utils.config import get_settings
from src.utils.logger import clear_correlation_id, get_logger, set_correlation_id


def _load_version() -> str:
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return "1.0.0"
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):  # pragma: no cover - defensive
        return "1.0.0"
    return data.get("project", {}).get("version", "1.0.0")


SETTINGS = get_settings()
APP_VERSION = _load_version()
logger = get_logger("audra.api.app")


def _resolve_origins(value: Optional[Iterable[str]]) -> list[str]:
    if value is None:
        return ["*"]
    return [origin.strip() for origin in value if origin]


def _middleware_cors(app: FastAPI) -> None:
    origins_attr = getattr(SETTINGS, "CORS_ORIGINS", None)
    if isinstance(origins_attr, str):
        origins = [item.strip() for item in origins_attr.split(",") if item.strip()]
    elif isinstance(origins_attr, Iterable):
        origins = _resolve_origins(origins_attr)
    else:
        origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def _initialise_services(app: FastAPI) -> None:
    logger.info("Starting AuDRA-Rad API services...")
    llm_client: Optional[NemotronClient | OllamaClient] = None
    embedding_client: Optional[EmbeddingClient] = None
    vector_store: Optional[VectorStore] = None
    ehr_client: Optional[EHRClient] = None

    # Initialize LLM client based on configuration
    if SETTINGS.LLM_BACKEND == "ollama":
        try:
            llm_client = OllamaClient(
                model_name=SETTINGS.OLLAMA_MODEL_NAME,
                base_url=SETTINGS.OLLAMA_BASE_URL
            )
            logger.info(f"Ollama client initialised (model={SETTINGS.OLLAMA_MODEL_NAME}).")
        except OllamaServiceError as exc:
            logger.error(
                "Ollama client initialisation failed.",
                extra={"context": {"error": str(exc)}},
            )
    else:
        try:
            llm_client = NemotronClient()
            logger.info("Nemotron client initialised.")
        except NIMServiceError as exc:
            logger.error(
                "Nemotron client initialisation failed.",
                extra={"context": {"error": str(exc)}},
            )

    # Initialize embeddings and vector store based on backend
    if SETTINGS.LLM_BACKEND == "ollama":
        # For Ollama, use Ollama embeddings and simple local vector store
        try:
            embedding_client = OllamaEmbeddingClient(
                model_name="nomic-embed-text",
                base_url=SETTINGS.OLLAMA_BASE_URL
            )
            logger.info("Ollama embedding client initialised (model=nomic-embed-text).")
        except OllamaEmbeddingError as exc:
            logger.error(
                "Ollama embedding client initialisation failed.",
                extra={"context": {"error": str(exc)}},
            )

        try:
            vector_store = SimpleVectorStore(index_name="medical_guidelines")
            logger.info(f"Simple vector store initialised ({vector_store.get_document_count()} documents indexed).")
        except Exception as exc:
            logger.error(
                "Simple vector store initialisation failed.",
                extra={"context": {"error": str(exc)}},
            )
    else:
        # For NIM backend, initialize embeddings and vector store
        try:
            embedding_client = EmbeddingClient()
            logger.info("Embedding client initialised.")
        except (EmbeddingServiceError, NIMServiceError) as exc:
            logger.error(
                "Embedding client initialisation failed.",
                extra={"context": {"error": str(exc)}},
            )

        try:
            vector_store = VectorStore(index_name="medical_guidelines")
            logger.info("Vector store initialised.")
        except VectorStoreError as exc:
            logger.error(
                "Vector store initialisation failed.",
                extra={"context": {"error": str(exc)}},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Unexpected error initialising vector store.",
                extra={"context": {"error": str(exc)}},
            )

    try:
        ehr_client = EHRClient(use_mock=True)
        logger.info("EHR client initialised (mock=%s).", ehr_client.use_mock)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "EHR client initialisation failed.",
            extra={"context": {"error": str(exc)}},
        )

    # All backends now require all services
    required_services_met = llm_client and embedding_client and vector_store and ehr_client

    if required_services_met:
        app.state.agent = AuDRAAgent(
            llm_client=llm_client,
            embedding_client=embedding_client,
            vector_store=vector_store,
            ehr_client=ehr_client,
        )
        app.state.llm_client = llm_client
        app.state.embedding_client = embedding_client
        app.state.vector_store = vector_store
        app.state.ehr_client = ehr_client
        logger.info("AuDRA agent initialised successfully.")
    else:
        app.state.agent = None
        app.state.llm_client = llm_client
        app.state.embedding_client = embedding_client
        app.state.vector_store = vector_store
        app.state.ehr_client = ehr_client
        logger.warning("AuDRA agent not initialised - one or more dependencies unavailable.")


def _shutdown_services(app: FastAPI) -> None:
    logger.info("Shutting down AuDRA-Rad API services...")
    ehr_client = getattr(app.state, "ehr_client", None)
    if ehr_client and hasattr(ehr_client, "close"):
        try:
            ehr_client.close()
        except Exception:  # pragma: no cover - defensive cleanup
            pass


@asynccontextmanager
async def _app_lifespan(app: FastAPI):
    _initialise_services(app)
    try:
        yield
    finally:
        _shutdown_services(app)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="AuDRA-Rad API",
        description="Autonomous Radiology Follow-up Assistant - Transforming findings into follow-through.",
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=_app_lifespan,
    )

    app.state.settings = SETTINGS
    app.state.version = APP_VERSION
    app.state.agent = None
    app.state.llm_client = None
    app.state.embedding_client = None
    app.state.vector_store = None
    app.state.ehr_client = None

    _middleware_cors(app)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(SlowAPIMiddleware)

    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid4())
        set_correlation_id(correlation_id)
        start_time = time.perf_counter()

        logger.info(
            "Incoming request.",
            extra={
                "context": {
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                }
            },
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception(
                "Unhandled exception during request.",
                extra={
                    "context": {
                        "correlation_id": correlation_id,
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": duration_ms,
                    }
                },
            )
            clear_correlation_id()
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Correlation-ID"] = correlation_id
        logger.info(
            "Request completed.",
            extra={
                "context": {
                    "correlation_id": correlation_id,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                }
            },
        )
        clear_correlation_id()
        return response

    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")

    @app.get("/", tags=["Health"])
    async def root() -> dict[str, Any]:
        return {
            "service": "AuDRA-Rad",
            "version": APP_VERSION,
            "status": "running",
            "docs": "/docs",
        }

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
