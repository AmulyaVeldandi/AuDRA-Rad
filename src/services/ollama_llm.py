from __future__ import annotations

"""Client for Ollama local language model service."""

import json
import threading
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict, Iterator, Optional

import requests
from jsonschema import ValidationError, validate
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.config import get_settings
from src.utils.logger import get_logger, log_error


class OllamaServiceError(RuntimeError):
    """Raised when calls to the Ollama service fail."""


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class OllamaClient:
    """Client wrapper for Ollama running locally."""

    MODEL_NAME = "llama3.1:8b"
    BASE_URL = "http://localhost:11434"
    REQUEST_TIMEOUT_SECONDS = 120.0

    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None) -> None:
        """Initialize Ollama client.

        Args:
            model_name: Name of the Ollama model to use (default: llama3.1:8b)
            base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.model_name = model_name or self.MODEL_NAME
        self.base_url = base_url or self.BASE_URL

        self._logger = get_logger("audra.services.ollama_llm")
        self._metrics_lock = threading.Lock()
        self._total_tokens = 0
        self._latencies_ms: list[float] = []
        self._total_calls = 0
        self._error_count = 0

        self._retryer = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((requests.exceptions.RequestException,)),
            reraise=True,
        )

        # Verify Ollama is running
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify Ollama service is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]

            if self.model_name not in model_names:
                self._logger.warning(
                    f"Model {self.model_name} not found. Available models: {model_names}. "
                    f"Run: ollama pull {self.model_name}"
                )
        except requests.exceptions.RequestException as exc:
            raise OllamaServiceError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running: ollama serve"
            ) from exc

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a text completion from the Ollama model."""

        request_context = {
            "operation": "generate",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_preview": prompt[:200],
        }
        self._logger.debug("Submitting Ollama generate request.", extra={"context": request_context})

        start_time = time.perf_counter()
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            response = self._run_with_retry(
                lambda: requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                ),
                operation="generate",
                context=request_context,
            )
            response.raise_for_status()
            result = response.json()

        except OllamaServiceError:
            raise
        except Exception as exc:
            self._record_error()
            self._log_exception("generate", exc, request_context)
            raise OllamaServiceError("Ollama generate request failed.") from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        usage = _Usage(
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
        )
        self._record_success(usage, latency_ms)

        content = result.get("response", "")
        response_context = {
            "operation": "generate",
            "latency_ms": latency_ms,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "response_preview": content[:200],
        }
        self._logger.debug(
            "Received Ollama generate response.", extra={"context": response_context}
        )
        return content

    def generate_json(
        self,
        prompt: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Force a JSON response and validate the payload."""

        attempt = 0
        last_error: Exception | None = None

        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only."

        while attempt < max_retries:
            attempt += 1
            attempt_context = {
                "operation": "generate_json",
                "attempt": attempt,
                "prompt_preview": prompt[:200],
            }
            self._logger.debug(
                "Submitting Ollama JSON generation request.", extra={"context": attempt_context}
            )
            start_time = time.perf_counter()
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": json_prompt,
                    "stream": False,
                    "format": "json",  # Ollama's JSON mode
                    "options": {
                        "temperature": 0.1,
                    }
                }

                response = self._run_with_retry(
                    lambda: requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self.REQUEST_TIMEOUT_SECONDS,
                    ),
                    operation="generate_json",
                    context=attempt_context,
                )
                response.raise_for_status()
                result = response.json()

            except OllamaServiceError as exc:
                last_error = exc
                break
            except Exception as exc:
                last_error = exc
                self._record_error()
                self._log_exception("generate_json", exc, attempt_context)
                break

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            usage = _Usage(
                prompt_tokens=result.get("prompt_eval_count", 0),
                completion_tokens=result.get("eval_count", 0),
            )
            self._record_success(usage, latency_ms)

            raw_content = result.get("response", "")
            try:
                payload_data = json.loads(raw_content)
            except JSONDecodeError as exc:
                last_error = exc
                self._logger.debug(
                    "Ollama JSON decode failed; retrying.",
                    extra={
                        "context": {
                            "operation": "generate_json",
                            "attempt": attempt,
                            "error": str(exc),
                            "raw_preview": raw_content[:200],
                        }
                    },
                )
                continue

            if schema is not None:
                try:
                    validate(instance=payload_data, schema=schema)
                except ValidationError as exc:
                    last_error = exc
                    self._logger.debug(
                        "Ollama JSON schema validation failed; retrying.",
                        extra={
                            "context": {
                                "operation": "generate_json",
                                "attempt": attempt,
                                "error": exc.message,
                            }
                        },
                    )
                    continue

            self._logger.debug(
                "Ollama JSON generation succeeded.",
                extra={
                    "context": {
                        "operation": "generate_json",
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                    }
                },
            )
            return payload_data

        error = last_error or RuntimeError("Ollama JSON generation failed.")
        self._record_error()
        self._log_exception(
            "generate_json",
            error,
            {
                "operation": "generate_json",
                "prompt_preview": prompt[:200],
                "attempts": attempt,
            },
        )
        raise OllamaServiceError("Ollama JSON generation failed.") from error

    def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream tokens back as they are produced."""

        temperature = 0.1
        max_tokens = 2048
        request_context = {
            "operation": "generate_stream",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_preview": prompt[:200],
        }
        self._logger.debug(
            "Submitting Ollama streaming request.", extra={"context": request_context}
        )

        def _stream() -> Iterator[str]:
            start_time = time.perf_counter()
            prompt_tokens = 0
            completion_tokens = 0
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                }

                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()

            except Exception as exc:
                self._record_error()
                self._log_exception("generate_stream", exc, request_context)
                raise OllamaServiceError("Ollama streaming request failed.") from exc

            success = True
            try:
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]

                        # Final chunk has token counts
                        if chunk.get("done", False):
                            prompt_tokens = chunk.get("prompt_eval_count", 0)
                            completion_tokens = chunk.get("eval_count", 0)

            except Exception as exc:
                success = False
                self._record_error()
                self._log_exception("generate_stream", exc, request_context)
                raise OllamaServiceError("Ollama streaming request failed.") from exc
            finally:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                usage = _Usage(prompt_tokens, completion_tokens)
                if success:
                    self._record_success(usage, latency_ms)
                    self._logger.debug(
                        "Ollama streaming request completed.",
                        extra={
                            "context": {
                                "operation": "generate_stream",
                                "latency_ms": latency_ms,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                            }
                        },
                    )

        return _stream()

    # --------------------------------------------------------------------- #
    # Metrics helpers
    # --------------------------------------------------------------------- #
    @property
    def total_tokens(self) -> int:
        with self._metrics_lock:
            return self._total_tokens

    @property
    def latencies_ms(self) -> list[float]:
        with self._metrics_lock:
            return list(self._latencies_ms)

    @property
    def error_rate(self) -> float:
        with self._metrics_lock:
            if self._total_calls == 0:
                return 0.0
            return self._error_count / self._total_calls

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _run_with_retry(
        self,
        func: Any,
        *,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        try:
            return self._retryer(func)
        except requests.exceptions.RequestException as exc:
            self._record_error()
            self._log_exception(operation, exc, context)
            raise OllamaServiceError(f"Ollama {operation} failed.") from exc

    def _record_success(self, usage: _Usage, latency_ms: float) -> None:
        with self._metrics_lock:
            self._total_calls += 1
            self._total_tokens += usage.total_tokens
            self._latencies_ms.append(latency_ms)

    def _record_error(self) -> None:
        with self._metrics_lock:
            self._error_count += 1
            self._total_calls += 1

    def _log_exception(
        self,
        operation: str,
        error: Exception,
        context: Optional[Dict[str, Any]],
    ) -> None:
        detail = {"operation": operation}
        if context:
            detail.update(context)
        log_error(error, context=detail)
