# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Observability module providing Intent vs. Outcome tracking and active PII redaction."""

from __future__ import annotations

import datetime
import enum
import logging
import re
from typing import Any, Optional
import uuid

from opentelemetry import trace
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Active PII Redaction Mechanisms
# ---------------------------------------------------------------------------

class PiiRedactor:
    """Active PII redaction engine to prevent sensitive customer data leaks in logs and traces.

    Redacts:
      - Email addresses: email@domain.com -> [REDACTED_EMAIL]
      - Phone numbers: US & international formats -> [REDACTED_PHONE]
      - Credit/Debit cards: 13-19 digit card numbers -> [REDACTED_CARD]
      - US Social Security Numbers: ###-##-#### -> [REDACTED_SSN]
      - API keys / Auth Bearer tokens -> [REDACTED_API_KEY]
    """

    # Email pattern
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )

    # Phone number patterns (various international and local formats)
    PHONE_PATTERN = re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )

    # Credit card pattern: 13-19 digits, optionally spaced or hyphenated
    CREDIT_CARD_PATTERN = re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b(?:\d{4}[-\s]?){2}\d{4}[-\s]?\d{3}\b"
    )

    # Social Security Number pattern: ###-##-####
    SSN_PATTERN = re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    )

    # API Keys / Bearer tokens / Secrets
    API_KEY_PATTERN = re.compile(
        r"(?:bearer\s+[A-Za-z0-9_\-\.]{20,}|AIza[0-9A-Za-z\-_]{35}|(?:api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{16,}['\"]?)",
        re.IGNORECASE,
    )

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Redact all detected PII entities from a text string.

        Args:
            text: Raw string that may contain sensitive customer PII.

        Returns:
            Sanitized string with sensitive tokens replaced with redaction tags.
        """
        if not isinstance(text, str):
            return text

        # Order matters: redact API keys and credit cards before phone numbers to prevent overlap
        redacted = cls.API_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)
        redacted = cls.CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", redacted)
        redacted = cls.SSN_PATTERN.sub("[REDACTED_SSN]", redacted)
        redacted = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
        redacted = cls.PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
        return redacted

    @classmethod
    def redact_data(cls, data: Any) -> Any:
        """Recursively redact PII in dictionaries, lists, and strings.

        Args:
            data: Arbitrary structured data or string.

        Returns:
            Sanitized data structure of matching shape.
        """
        if isinstance(data, str):
            return cls.redact_text(data)
        elif isinstance(data, dict):
            return {k: cls.redact_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.redact_data(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(cls.redact_data(item) for item in data)
        elif isinstance(data, BaseModel):
            dumped = data.model_dump()
            return cls.redact_data(dumped)
        return data


class PiiRedactionLoggingFilter(logging.Filter):
    """Logging filter that automatically scrubs PII from all log records before emission."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = PiiRedactor.redact_text(record.msg)
        elif isinstance(record.msg, dict):
            record.msg = PiiRedactor.redact_data(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = PiiRedactor.redact_data(record.args)
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(PiiRedactor.redact_data(a) for a in record.args)
        return True


# ---------------------------------------------------------------------------
# 2. Intent vs. Outcome Capture
# ---------------------------------------------------------------------------

class UserIntent(str, enum.Enum):
    """Categorized customer intent for incoming requests."""
    MENU_INQUIRY = "MENU_INQUIRY"
    DRINK_RECOMMENDATION = "DRINK_RECOMMENDATION"
    INVENTORY_CHECK = "INVENTORY_CHECK"
    ORDER_PLACEMENT = "ORDER_PLACEMENT"
    FEEDBACK = "FEEDBACK"
    GENERAL_INQUIRY = "GENERAL_INQUIRY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class TurnOutcome(str, enum.Enum):
    """Categorized outcome of the agent turn."""
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    OUT_OF_STOCK = "OUT_OF_STOCK"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    GUARDRAIL_BLOCKED = "GUARDRAIL_BLOCKED"
    ERROR = "ERROR"


class IntentOutcomeRecord(BaseModel):
    """Structured record capturing User Intent vs. Final Outcome for observability."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    session_id: str
    user_id: str
    query_text: str = Field(description="PII-sanitized customer query.")
    detected_intent: UserIntent
    intent_confidence: float = Field(ge=0.0, le=1.0)
    final_outcome: TurnOutcome
    intent_satisfied: bool
    tools_used: list[str] = Field(default_factory=list)
    model_routed: str = Field(description="Model used for this turn.")
    latency_ms: float = Field(ge=0.0)
    notes: Optional[str] = None
    log_type: str = "intent_vs_outcome"


class IntentClassifier:
    """Classifies user intent based on keyword and semantic pattern matching."""

    @classmethod
    def classify(cls, query: str) -> tuple[UserIntent, float]:
        """Classify customer intent with a confidence score.

        Args:
            query: The customer input string.

        Returns:
            Tuple of (UserIntent, confidence_score).
        """
        q = query.lower()

        # Check for order intent
        if any(w in q for w in ["order", "buy", "purchase", "get me a", "i'll have a", "i will take"]):
            return UserIntent.ORDER_PLACEMENT, 0.95

        # Check for inventory / stock intent
        if any(w in q for w in ["stock", "in stock", "available", "inventory", "do you have any", "remaining"]):
            return UserIntent.INVENTORY_CHECK, 0.90

        # Check for recommendation / preference intent
        if any(w in q for w in ["suggest", "recommend", "what do you suggest", "in the mood for", "what's good", "feeling like", "something sweet", "something cold", "something strong"]):
            return UserIntent.DRINK_RECOMMENDATION, 0.90

        # Check for menu browsing intent
        if any(w in q for w in ["menu", "price", "how much", "cost", "list", "options", "what drinks", "espresso", "latte", "cappuccino"]):
            return UserIntent.MENU_INQUIRY, 0.85

        # Feedback
        if any(w in q for w in ["feedback", "rating", "review", "star"]):
            return UserIntent.FEEDBACK, 0.85

        return UserIntent.GENERAL_INQUIRY, 0.70


class OutcomeEvaluator:
    """Evaluates turn outcome based on tools used and response content."""

    @classmethod
    def evaluate(
        cls,
        intent: UserIntent,
        response_text: str,
        tools_used: list[str],
        guardrail_blocked: bool = False,
        error_occurred: bool = False,
    ) -> tuple[TurnOutcome, bool]:
        """Determine the outcome and whether the initial intent was satisfied.

        Returns:
            Tuple of (TurnOutcome, intent_satisfied).
        """
        if guardrail_blocked:
            return TurnOutcome.GUARDRAIL_BLOCKED, False

        if error_occurred:
            return TurnOutcome.ERROR, False

        resp_lower = response_text.lower()

        if "out of stock" in resp_lower or "❌" in resp_lower:
            return TurnOutcome.OUT_OF_STOCK, True

        if "confirmation required" in resp_lower or "confirm" in resp_lower and "order" in resp_lower:
            return TurnOutcome.CONFIRMATION_REQUIRED, True

        if intent == UserIntent.MENU_INQUIRY and (
            "menu" in tools_used or any(t in tools_used for t in ["get_full_menu", "get_menu_item"]) or "$" in response_text
        ):
            return TurnOutcome.SUCCESS, True

        if intent == UserIntent.DRINK_RECOMMENDATION and (
            "suggest_drink" in tools_used or "suggest" in resp_lower or "recommend" in resp_lower
        ):
            return TurnOutcome.SUCCESS, True

        if intent == UserIntent.INVENTORY_CHECK and (
            any(t in tools_used for t in ["check_inventory", "check_full_inventory"]) or "stock" in resp_lower
        ):
            return TurnOutcome.SUCCESS, True

        if intent == UserIntent.ORDER_PLACEMENT and (
            "place_order" in tools_used or "order" in resp_lower
        ):
            return TurnOutcome.SUCCESS, True

        # Default success if agent produced a helpful response
        if len(response_text.strip()) > 10:
            return TurnOutcome.SUCCESS, True

        return TurnOutcome.PARTIAL_SUCCESS, False


class ObservabilityService:
    """Coordinates intent vs. outcome recording, OpenTelemetry traces, and structured logging."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("cafe_concierge.observability")
        self.tracer = trace.get_tracer("cafe_concierge")

    def record_turn(
        self,
        session_id: str,
        user_id: str,
        query_text: str,
        response_text: str,
        tools_used: list[str],
        model_routed: str,
        latency_ms: float,
        guardrail_blocked: bool = False,
        error_occurred: bool = False,
        notes: Optional[str] = None,
    ) -> IntentOutcomeRecord:
        """Capture intent vs outcome, redact PII, set trace attributes, and emit structured log.

        Args:
            session_id: Conversation session ID.
            user_id: User identifier.
            query_text: Raw incoming query.
            response_text: Generated agent response.
            tools_used: List of tool names executed during turn.
            model_routed: Strategic model identifier used.
            latency_ms: Turn latency in milliseconds.
            guardrail_blocked: Whether the request was halted by security guardrails.
            error_occurred: Whether an unhandled error occurred.
            notes: Optional diagnostic notes.

        Returns:
            The recorded IntentOutcomeRecord.
        """
        # 1. Active PII Redaction
        sanitized_query = PiiRedactor.redact_text(query_text)
        sanitized_response = PiiRedactor.redact_text(response_text)

        # 2. Classify Intent & Evaluate Outcome
        intent, confidence = IntentClassifier.classify(sanitized_query)
        outcome, satisfied = OutcomeEvaluator.evaluate(
            intent=intent,
            response_text=sanitized_response,
            tools_used=tools_used,
            guardrail_blocked=guardrail_blocked,
            error_occurred=error_occurred,
        )

        record = IntentOutcomeRecord(
            session_id=session_id,
            user_id=user_id,
            query_text=sanitized_query,
            detected_intent=intent,
            intent_confidence=confidence,
            final_outcome=outcome,
            intent_satisfied=satisfied,
            tools_used=tools_used,
            model_routed=model_routed,
            latency_ms=latency_ms,
            notes=notes,
        )

        # 3. Attach to active OpenTelemetry span
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("gen_ai.user_intent", record.detected_intent.value)
            current_span.set_attribute("gen_ai.intent_confidence", record.intent_confidence)
            current_span.set_attribute("gen_ai.outcome", record.final_outcome.value)
            current_span.set_attribute("gen_ai.intent_satisfied", record.intent_satisfied)
            current_span.set_attribute("gen_ai.tools_used", ",".join(record.tools_used))
            current_span.set_attribute("gen_ai.model", record.model_routed)
            current_span.set_attribute("gen_ai.latency_ms", record.latency_ms)

        # 4. Structured log emission
        self.logger.info(
            "Intent vs Outcome: %s -> %s (satisfied=%s)",
            record.detected_intent.value,
            record.final_outcome.value,
            record.intent_satisfied,
            extra={"structured_data": record.model_dump()},
        )

        return record


# Global singleton instance
observability_service = ObservabilityService()
