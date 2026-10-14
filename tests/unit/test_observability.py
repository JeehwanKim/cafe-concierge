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

"""Unit tests for active PII redaction and Intent vs. Outcome observability."""

import logging
from app.observability import (
    IntentClassifier,
    IntentOutcomeRecord,
    ObservabilityService,
    OutcomeEvaluator,
    PiiRedactionLoggingFilter,
    PiiRedactor,
    TurnOutcome,
    UserIntent,
)


class TestPiiRedaction:
    """Validate active redaction of sensitive customer data."""

    def test_email_redacted(self):
        text = "Contact me at barista.fan@gmail.com with updates."
        redacted = PiiRedactor.redact_text(text)
        assert "barista.fan@gmail.com" not in redacted
        assert "[REDACTED_EMAIL]" in redacted

    def test_phone_number_redacted(self):
        text = "Call me at (555) 234-5678 or +1 800-555-0199."
        redacted = PiiRedactor.redact_text(text)
        assert "(555) 234-5678" not in redacted
        assert "[REDACTED_PHONE]" in redacted

    def test_credit_card_redacted(self):
        text = "My card number is 4111 2222 3333 4444 for the espresso."
        redacted = PiiRedactor.redact_text(text)
        assert "4111 2222 3333 4444" not in redacted
        assert "[REDACTED_CARD]" in redacted

    def test_ssn_redacted(self):
        text = "Customer SSN: 123-45-6789."
        redacted = PiiRedactor.redact_text(text)
        assert "123-45-6789" not in redacted
        assert "[REDACTED_SSN]" in redacted

    def test_api_key_redacted(self):
        text = "Authorization: Bearer ya29.a0AfH6SMAabc1234567890def1234567890."
        redacted = PiiRedactor.redact_text(text)
        assert "ya29.a0Af" not in redacted
        assert "[REDACTED_API_KEY]" in redacted

    def test_nested_data_structure_redacted(self):
        data = {
            "customer": "John",
            "contact": {
                "email": "john.smith@domain.com",
                "phone": "415-555-2671",
            },
            "notes": ["Card 4111 2222 3333 4444 stored"],
        }
        sanitized = PiiRedactor.redact_data(data)
        assert sanitized["contact"]["email"] == "[REDACTED_EMAIL]"
        assert sanitized["contact"]["phone"] == "[REDACTED_PHONE]"
        assert "[REDACTED_CARD]" in sanitized["notes"][0]

    def test_logging_filter_scrubs_record(self):
        filter_instance = PiiRedactionLoggingFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Customer phone is 555-890-1234",
            args=(),
            exc_info=None,
        )
        filter_instance.filter(record)
        assert "555-890-1234" not in record.msg
        assert "[REDACTED_PHONE]" in record.msg


class TestIntentClassification:
    """Validate user intent classifier."""

    def test_classify_order_intent(self):
        intent, conf = IntentClassifier.classify("I'd like to order a large latte please.")
        assert intent == UserIntent.ORDER_PLACEMENT
        assert conf >= 0.90

    def test_classify_inventory_intent(self):
        intent, conf = IntentClassifier.classify("Do you have any oat milk in stock?")
        assert intent == UserIntent.INVENTORY_CHECK
        assert conf >= 0.85

    def test_classify_recommendation_intent(self):
        intent, conf = IntentClassifier.classify("What do you suggest for something cold and sweet?")
        assert intent == UserIntent.DRINK_RECOMMENDATION
        assert conf >= 0.85

    def test_classify_menu_inquiry_intent(self):
        intent, conf = IntentClassifier.classify("How much does a cappuccino cost?")
        assert intent == UserIntent.MENU_INQUIRY

    def test_classify_feedback_intent(self):
        intent, conf = IntentClassifier.classify("I want to leave a 5-star review and feedback.")
        assert intent == UserIntent.FEEDBACK


class TestOutcomeEvaluation:
    """Validate turn outcome and satisfaction evaluation."""

    def test_out_of_stock_detected(self):
        outcome, satisfied = OutcomeEvaluator.evaluate(
            intent=UserIntent.INVENTORY_CHECK,
            response_text="Sorry, Blueberry Muffins are currently OUT OF STOCK.",
            tools_used=["check_inventory"],
        )
        assert outcome == TurnOutcome.OUT_OF_STOCK
        assert satisfied is True

    def test_confirmation_required_detected(self):
        outcome, satisfied = OutcomeEvaluator.evaluate(
            intent=UserIntent.ORDER_PLACEMENT,
            response_text="Action confirmation required: You are ordering 1x Latte.",
            tools_used=["place_order"],
        )
        assert outcome == TurnOutcome.CONFIRMATION_REQUIRED
        assert satisfied is True

    def test_guardrail_blocked_detected(self):
        outcome, satisfied = OutcomeEvaluator.evaluate(
            intent=UserIntent.GENERAL_INQUIRY,
            response_text="Action blocked by security guardrails.",
            tools_used=[],
            guardrail_blocked=True,
        )
        assert outcome == TurnOutcome.GUARDRAIL_BLOCKED
        assert satisfied is False


class TestObservabilityService:
    """Validate full intent vs outcome recording pipeline."""

    def test_record_turn_redacts_and_creates_record(self):
        svc = ObservabilityService()
        record = svc.record_turn(
            session_id="sess_123",
            user_id="user_123",
            query_text="My email is alice@cafe.com. Can I see the menu?",
            response_text="Here is our full menu with prices: Espresso $3.00, Latte $4.50.",
            tools_used=["get_full_menu"],
            model_routed="gemini-3.8-flash",
            latency_ms=120.5,
        )

        assert isinstance(record, IntentOutcomeRecord)
        assert record.session_id == "sess_123"
        # Query must be sanitized
        assert "alice@cafe.com" not in record.query_text
        assert "[REDACTED_EMAIL]" in record.query_text
        # Intent classification & outcome
        assert record.detected_intent == UserIntent.MENU_INQUIRY
        assert record.final_outcome == TurnOutcome.SUCCESS
        assert record.intent_satisfied is True
