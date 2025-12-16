from __future__ import annotations

"""Use LLM reasoning to map findings to structured recommendations."""

import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from src.guidelines.indexer import GuidelineChunk
from src.services.nim_llm import NemotronClient, NIMServiceError
from src.utils.logger import get_logger


@dataclass
class Recommendation:
    """Structured follow-up advice distilled from guideline chunks."""

    follow_up_type: str
    timeframe_months: Optional[int]
    urgency: Literal["routine", "priority", "urgent", "stat"]
    reasoning: str
    citation: str
    confidence: float


class RecommendationMatcher:
    """Select the best-matching guideline recommendation via LLM reasoning."""

    _OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "follow_up_type": {"type": "string", "minLength": 1},
            "timeframe_months": {"type": ["integer", "null"], "minimum": 0},
            "urgency": {
                "type": "string",
                "enum": ["routine", "priority", "urgent", "stat"],
            },
            "reasoning": {"type": "string", "minLength": 1},
            "citation": {"type": "string", "minLength": 1},
        },
        "required": ["follow_up_type", "urgency", "reasoning", "citation"],
        "additionalProperties": False,
    }

    def __init__(self, llm_client: NemotronClient) -> None:
        self._llm_client = llm_client
        self._logger = get_logger("audra.guidelines.matcher")

    # ------------------------------------------------------------------ #
    def match(
        self,
        finding: Dict[str, object],
        guidelines: List[GuidelineChunk],
        patient_context: Optional[Dict[str, object]] = None,
    ) -> Recommendation:
        """Return the most appropriate follow-up recommendation."""

        if not guidelines:
            # Ollama mode without RAG - use LLM's general medical knowledge
            self._logger.info("No guidelines available, using LLM general knowledge.")
            prompt = self._build_prompt_without_guidelines(finding, patient_context)
        else:
            prompt = self._build_prompt(finding, guidelines, patient_context)
        self._logger.info(
            "Requesting recommendation from LLM.",
            extra={"context": {"finding": finding, "guideline_count": len(guidelines)}},
        )

        try:
            response = self._llm_client.generate_json(prompt, schema=self._OUTPUT_SCHEMA)
        except Exception as exc:
            self._logger.warning(
                "LLM call failed; falling back to heuristic recommendation.",
                extra={"context": {"error": str(exc)}},
            )
            if guidelines:
                return self._fallback_recommendation(guidelines)
            else:
                return self._default_recommendation(finding)

        if guidelines:
            if not self.validate_recommendation(response, guidelines):
                self._logger.warning(
                    "LLM response did not align with provided guidelines; applying fallback.",
                    extra={"context": {"response": response}},
                )
                return self._fallback_recommendation(guidelines)

        timeframe = response.get("timeframe_months")
        if timeframe is not None:
            timeframe = int(timeframe)

        recommendation = Recommendation(
            follow_up_type=response["follow_up_type"],
            timeframe_months=timeframe,
            urgency=response["urgency"],
            reasoning=response["reasoning"],
            citation=response["citation"],
            confidence=0.85,
        )

        self._logger.info(
            "Recommendation generated.",
            extra={
                "context": {
                    "follow_up_type": recommendation.follow_up_type,
                    "timeframe_months": recommendation.timeframe_months,
                    "urgency": recommendation.urgency,
                    "citation": recommendation.citation,
                }
            },
        )
        return recommendation

    def validate_recommendation(
        self,
        rec: Dict[str, object],
        guidelines: List[GuidelineChunk],
    ) -> bool:
        """Ensure the LLM response is plausible and grounded."""

        citation = str(rec.get("citation", "")).lower()
        if not citation:
            return False

        known_sources = {chunk.source.lower() for chunk in guidelines}
        if not any(source in citation for source in known_sources):
            return False

        timeframe = rec.get("timeframe_months")
        if timeframe is not None:
            try:
                timeframe = int(timeframe)
            except (TypeError, ValueError):
                return False
            if timeframe < 0 or timeframe > 60:
                return False

        urgency = rec.get("urgency")
        if urgency not in {"routine", "priority", "urgent", "stat"}:
            return False

        follow_up_type = rec.get("follow_up_type")
        if not follow_up_type:
            return False

        return True

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        finding: Dict[str, object],
        guidelines: List[GuidelineChunk],
        patient_context: Optional[Dict[str, object]],
    ) -> str:
        finding_description = json.dumps(finding, ensure_ascii=False)
        patient_description = json.dumps(patient_context or {}, ensure_ascii=False)

        guideline_lines = []
        for index, chunk in enumerate(guidelines, start=1):
            snippet = self._truncate(chunk.recommendation or chunk.text, 700)
            guideline_lines.append(
                f"{index}. [{chunk.source}] {chunk.category}: {snippet}"
            )

        prompt = (
            "You are a clinical decision support system for radiology follow-up.\n\n"
            f"Finding: {finding_description}\n"
            f"Patient: {patient_description}\n\n"
            "Relevant Guidelines:\n"
            f"{chr(10).join(guideline_lines)}\n\n"
            "Based on the guidelines, determine:\n"
            "1. Is follow-up needed?\n"
            "2. If yes, what type (CT, MRI, biopsy, referral)?\n"
            "3. What timeframe (in months)?\n"
            "4. What urgency level (routine, urgent, stat)?\n"
            "5. Which guideline supports this recommendation?\n\n"
            "Return as JSON: "
            '{"follow_up_type": "...", "timeframe_months": 0, "urgency": "routine", '
            '"reasoning": "...", "citation": "..."}'
        )
        return prompt

    def _build_prompt_without_guidelines(
        self,
        finding: Dict[str, object],
        patient_context: Optional[Dict[str, object]],
    ) -> str:
        """Build a prompt for LLM without retrieved guidelines (Ollama mode)."""
        finding_description = json.dumps(finding, ensure_ascii=False)
        patient_description = json.dumps(patient_context or {}, ensure_ascii=False)

        prompt = (
            "You are a clinical decision support system for radiology follow-up.\n\n"
            f"Finding: {finding_description}\n"
            f"Patient: {patient_description}\n\n"
            "Based on standard clinical practice and medical guidelines "
            "(such as Fleischner Society guidelines for pulmonary nodules), determine:\n"
            "1. Is follow-up needed for this finding?\n"
            "2. If yes, what type of follow-up (CT, MRI, biopsy, clinical evaluation)?\n"
            "3. What timeframe in months?\n"
            "4. What urgency level (routine, priority, urgent, stat)?\n"
            "5. What is the clinical reasoning?\n\n"
            "Return as JSON: "
            '{"follow_up_type": "...", "timeframe_months": 0, "urgency": "routine", '
            '"reasoning": "...", "citation": "Standard clinical practice"}'
        )
        return prompt

    def _default_recommendation(self, finding: Dict[str, object]) -> Recommendation:
        """Generate a conservative default recommendation when no guidelines available and LLM fails."""
        finding_type = str(finding.get("type", "finding"))
        size_mm = finding.get("size_mm")

        # Conservative defaults based on finding type
        if finding_type.lower() in ("nodule", "pulmonary_nodule"):
            if size_mm and float(size_mm) >= 6:
                timeframe = 6
                reasoning = "Conservative follow-up recommended for nodule ≥6mm based on general clinical practice."
            else:
                timeframe = 12
                reasoning = "Annual follow-up recommended for small nodule based on general clinical practice."
            follow_up_type = "CT chest"
        else:
            timeframe = 6
            follow_up_type = "Clinical evaluation"
            reasoning = f"Follow-up recommended for {finding_type} based on general clinical practice."

        return Recommendation(
            follow_up_type=follow_up_type,
            timeframe_months=timeframe,
            urgency="routine",
            reasoning=reasoning,
            citation="Standard clinical practice (no specific guideline available)",
            confidence=0.3,
        )

    def _fallback_recommendation(self, guidelines: List[GuidelineChunk]) -> Recommendation:
        guideline = guidelines[0]
        timeframe = self._extract_timeframe_months(guideline.recommendation)

        follow_up_type = guideline.modality or self._infer_modality_from_text(guideline.recommendation)
        if not follow_up_type:
            follow_up_type = "Clinical follow-up"

        reasoning = (
            "Fallback recommendation derived directly from the most specific available guideline chunk. "
            f"Category: {guideline.category}."
        )

        return Recommendation(
            follow_up_type=follow_up_type,
            timeframe_months=timeframe,
            urgency="routine",
            reasoning=reasoning,
            citation=guideline.citation or guideline.source,
            confidence=0.5,
        )

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _extract_timeframe_months(text: str) -> Optional[int]:
        if not text:
            return None

        matches = re.findall(r"(\d+)\s*(?:-|to)?\s*(\d+)?\s*(?:month|mo)", text, flags=re.IGNORECASE)
        timeframes: List[int] = []
        for start, end in matches:
            start_val = int(start)
            if end:
                end_val = int(end)
                timeframes.append(min(start_val, end_val))
            else:
                timeframes.append(start_val)

        if not timeframes:
            numeric_matches = re.findall(r"\b(\d{1,2})\s*(?:wk|week)\b", text, flags=re.IGNORECASE)
            for weeks in numeric_matches:
                try:
                    months = max(1, math.ceil(int(weeks) / 4.0))
                    timeframes.append(months)
                except ValueError:
                    continue

        return min(timeframes) if timeframes else None

    @staticmethod
    def _infer_modality_from_text(text: str) -> Optional[str]:
        lowered = text.lower()
        if "pet" in lowered and "ct" in lowered:
            return "PET-CT"
        if "biopsy" in lowered:
            return "Biopsy"
        if "mri" in lowered:
            return "MRI"
        if "ultrasound" in lowered or "sonograph" in lowered:
            return "Ultrasound"
        if "ct" in lowered:
            return "CT"
        return None
