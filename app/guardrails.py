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

"""Runtime Security Guardrails module for input injection defense and output safety."""

from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("cafe_concierge.guardrails")


class GuardrailResult(BaseModel):
    """Result of a guardrail security validation."""

    allowed: bool = Field(description="Whether the prompt or response passes security policies.")
    violation_type: Optional[str] = Field(default=None, description="Classification of policy violation.")
    reason: Optional[str] = Field(default=None, description="Human-readable explanation of why input/output was blocked.")
    refusal_response: Optional[str] = Field(
        default=None, description="Polite, safe canned response to return when blocked."
    )

    @property
    def category(self) -> Optional[str]:
        """Convenience property alias for violation_type."""
        return self.violation_type


class RuntimeSecurityGuardrails:
    """Enterprise runtime security guardrails protecting against prompt injection, jailbreaks, and prompt leakage."""

    # Dedicated Jailbreak signatures
    JAILBREAK_PATTERNS = [
        re.compile(r"bypass\s+(?:all\s+)?(?:safety|guardrails?|restrictions?|filters?)", re.IGNORECASE),
        re.compile(r"\bdo\s+anything\s+now\b|\bDAN\s+mode\b", re.IGNORECASE),
        re.compile(r"\bdeveloper\s+(?:debug\s+)?mode\b", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(?:unrestricted|evil|chaos|hacked)", re.IGNORECASE),
    ]

    # Dedicated Prompt injection signatures
    PROMPT_INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.IGNORECASE),
        re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+prompts?", re.IGNORECASE),
        re.compile(r"(?:print|output|repeat|reveal|display|show)\s+(?:your\s+)?(?:system\s+prompt|raw\s+instructions|base\s+prompt)", re.IGNORECASE),
        re.compile(r"<system>|<\/system>|\[INST\]|\[\/INST\]", re.IGNORECASE),
    ]

    # Sensitive internal markers that must not leak to the customer
    SENSITIVE_LEAKAGE_PATTERNS = [
        re.compile(r"CAFE_INSTRUCTION", re.IGNORECASE),
        re.compile(r"system_prompt", re.IGNORECASE),
        re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
        re.compile(r"sk-[A-Za-z0-9]{32,}"),
    ]

    @classmethod
    def validate_input(cls, user_text: str) -> GuardrailResult:
        """Evaluate incoming customer prompt against runtime security guardrails.

        Args:
            user_text: The incoming text input from the customer.

        Returns:
            GuardrailResult indicating whether the prompt is safe or blocked.
        """
        if not user_text or not user_text.strip():
            return GuardrailResult(allowed=True)

        # 1. Check for jailbreak patterns
        for pattern in cls.JAILBREAK_PATTERNS:
            if pattern.search(user_text):
                logger.warning("Jailbreak attempt intercepted: pattern '%s'", pattern.pattern)
                return GuardrailResult(
                    allowed=False,
                    violation_type="jailbreak",
                    reason="Input contains adversarial jailbreak attempt.",
                    refusal_response=(
                        "Blocked: I am designed to be a helpful Café Concierge assistant. "
                        "I cannot alter my core instructions or execute system override commands. "
                        "How can I help you with our menu or drink recommendations today?"
                    ),
                )

        # 2. Check for prompt injection patterns
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if pattern.search(user_text):
                logger.warning("Prompt injection attempt intercepted: pattern '%s'", pattern.pattern)
                return GuardrailResult(
                    allowed=False,
                    violation_type="prompt_injection",
                    reason="Input contains adversarial prompt injection or instruction override patterns.",
                    refusal_response=(
                        "Blocked: I am designed to be a helpful Café Concierge assistant. "
                        "I cannot alter my core instructions or execute system override commands. "
                        "How can I help you with our menu or drink recommendations today?"
                    ),
                )

        # 3. Check for excessive input length (DoS protection)
        if len(user_text) > 4000:
            logger.warning("Input length exceeds safe threshold: %d characters", len(user_text))
            return GuardrailResult(
                allowed=False,
                violation_type="dos_length",
                reason="Input exceeds maximum permissible character length.",
                refusal_response="Blocked: Your message is too long for our café terminal. Please summarize your request.",
            )

        return GuardrailResult(allowed=True)

    @classmethod
    def validate_output(cls, agent_response: str) -> GuardrailResult:
        """Inspect generated agent response for accidental system prompt or credential leakage.

        Args:
            agent_response: Output text generated by the agent.

        Returns:
            GuardrailResult with validation decision.
        """
        if not agent_response:
            return GuardrailResult(allowed=True)

        for pattern in cls.SENSITIVE_LEAKAGE_PATTERNS:
            if pattern.search(agent_response):
                logger.error("System prompt or credential leakage detected in output: pattern '%s'", pattern.pattern)
                return GuardrailResult(
                    allowed=False,
                    violation_type="SYSTEM_PROMPT_LEAKAGE_PREVENTED",
                    reason="Response contained protected internal instructions or credentials.",
                    refusal_response=(
                        "I am the Café Concierge! How can I help you choose a drink or pastry today?"
                    ),
                )

        return GuardrailResult(allowed=True)
