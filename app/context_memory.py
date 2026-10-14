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

"""Context Bloat Management, History Compaction, and Background Async Memory Operations."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

from google.adk.events import Event
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger("cafe_concierge.context_memory")


class CompactionMetrics(BaseModel):
    """Metrics tracking context bloat reduction and history compaction performance."""

    compaction_performed: bool = Field(default=False)
    original_event_count: int = Field(default=0)
    compacted_event_count: int = Field(default=0)
    estimated_tokens_saved: int = Field(default=0)
    summary_generated: Optional[str] = None


class ContextBloatManager:
    """Manages context window bloat by compacting older conversation turns into structured summaries."""

    def __init__(
        self,
        max_turns: int = 6,
        retain_recent_turns: int = 3,
        max_estimated_tokens: int = 3000,
    ):
        """Initialize the context bloat manager.

        Args:
            max_turns: Maximum number of conversation turns before triggering compaction.
            retain_recent_turns: Number of recent turns to preserve verbatim.
            max_estimated_tokens: Maximum token budget before triggering compaction.
        """
        self.max_turns = max_turns
        self.retain_recent_turns = retain_recent_turns
        self.max_estimated_tokens = max_estimated_tokens

    def estimate_tokens(self, events_or_messages: list[Any]) -> int:
        """Estimate token count across a sequence of events or content parts (~4 chars per token)."""
        total_chars = 0
        for item in events_or_messages:
            if hasattr(item, "content") and item.content and hasattr(item.content, "parts"):
                for p in item.content.parts:
                    if hasattr(p, "text") and p.text:
                        total_chars += len(p.text)
            elif isinstance(item, dict):
                total_chars += len(str(item))
            elif isinstance(item, str):
                total_chars += len(item)
        return total_chars // 4

    def should_compact(self, events: list[Any]) -> bool:
        """Determine if context bloat exceeds thresholds and requires compaction.

        Args:
            events: Current list of conversation events in the session.

        Returns:
            True if compaction should be triggered, False otherwise.
        """
        if not events:
            return False

        if len(events) > (self.max_turns * 2):  # *2 for user + model pairs
            return True

        if self.estimate_tokens(events) > self.max_estimated_tokens:
            return True

        return False

    def extract_key_context(self, older_events: list[Any]) -> str:
        """Extract customer preferences, orders, and key context from older events.

        Args:
            older_events: Sequence of older events to summarize.

        Returns:
            A concise structured summary string.
        """
        extracted_facts: list[str] = []

        for ev in older_events:
            text = ""
            role = "unknown"
            if hasattr(ev, "content") and ev.content:
                role = getattr(ev.content, "role", "unknown")
                if hasattr(ev.content, "parts") and ev.content.parts:
                    text = " ".join(p.text for p in ev.content.parts if getattr(p, "text", None))
            elif isinstance(ev, dict):
                text = str(ev)

            if not text:
                continue

            text_lower = text.lower()
            if "oat milk" in text_lower or "whole milk" in text_lower or "dairy free" in text_lower:
                extracted_facts.append(f"Customer mentioned milk preference: {text[:80]}")
            elif "order" in text_lower and role == "user":
                extracted_facts.append(f"Customer previously discussed order: {text[:80]}")
            elif "price" in text_lower or "$" in text:
                extracted_facts.append(f"Items discussed: {text[:80]}")
            elif "cold" in text_lower or "hot" in text_lower or "sweet" in text_lower:
                extracted_facts.append(f"Taste preferences: {text[:80]}")

        if not extracted_facts:
            return "Customer engaged in menu inquiries and drink browsing."

        return " | ".join(extracted_facts[:5])

    def compact_history(self, events: list[Any]) -> tuple[list[Any], CompactionMetrics]:
        """Perform history compaction on the session events to eliminate context bloat.

        Args:
            events: Full list of session events.

        Returns:
            Tuple of (compacted_events, CompactionMetrics).
        """
        if not self.should_compact(events):
            return events, CompactionMetrics(
                compaction_performed=False,
                original_event_count=len(events),
                compacted_event_count=len(events),
                estimated_tokens_saved=0,
            )

        original_count = len(events)
        cutoff_index = max(0, len(events) - (self.retain_recent_turns * 2))
        older_events = events[:cutoff_index]
        recent_events = events[cutoff_index:]

        # Create structured summary of older events
        summary_text = self.extract_key_context(older_events)
        compaction_banner = (
            f"[CONTEXT COMPACTED — Prior Conversation Summary: {summary_text}]"
        )

        # Build synthetic summary event
        summary_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=compaction_banner)],
        )
        summary_event = Event(content=summary_content)

        compacted_events = [summary_event] + recent_events

        tokens_before = self.estimate_tokens(events)
        tokens_after = self.estimate_tokens(compacted_events)
        tokens_saved = max(0, tokens_before - tokens_after)

        metrics = CompactionMetrics(
            compaction_performed=True,
            original_event_count=original_count,
            compacted_event_count=len(compacted_events),
            estimated_tokens_saved=tokens_saved,
            summary_generated=compaction_banner,
        )

        logger.info(
            "History compaction executed: %d -> %d events, saved ~%d tokens",
            original_count,
            len(compacted_events),
            tokens_saved,
        )
        return compacted_events, metrics


class BackgroundMemoryManager:
    """Asynchronous background memory processor executing compaction and memory updates non-blockingly."""

    def __init__(self, bloat_manager: Optional[ContextBloatManager] = None):
        self.bloat_manager = bloat_manager or ContextBloatManager()

    async def async_process_session_memory(
        self,
        session_service: Any,
        session_id: str,
        user_id: str,
        app_name: str,
        last_user_query: Optional[str] = None,
        last_agent_response: Optional[str] = None,
    ) -> CompactionMetrics:
        """Background asynchronous task to compact history and persist user memory facts.

        Runs completely decoupled from the user request/response path to eliminate latency.

        Args:
            session_service: The ADK session service (Vertex AI or In-Memory).
            session_id: Session identifier.
            user_id: User identifier.
            app_name: ADK application name.
            last_user_query: Recent user message.
            last_agent_response: Recent agent response.

        Returns:
            CompactionMetrics recording results of background compaction.
        """
        logger.debug("Background memory task started for session '%s'", session_id)
        try:
            # 1. Fetch current session asynchronously
            if hasattr(session_service, "get_session"):
                session = await session_service.get_session(
                    session_id=session_id, user_id=user_id, app_name=app_name
                )
            elif hasattr(session_service, "get_session_sync"):
                session = session_service.get_session_sync(
                    session_id=session_id, user_id=user_id, app_name=app_name
                )
            else:
                session = None

            if not session:
                logger.warning("Session '%s' not found for background memory update", session_id)
                return CompactionMetrics()

            # 2. Extract and asynchronously update user preferences in session state
            if last_user_query:
                prefs = session.state.setdefault("customer_profile", {})
                q_lower = last_user_query.lower()
                if "oat milk" in q_lower:
                    prefs["preferred_milk"] = "oat"
                elif "almond milk" in q_lower:
                    prefs["preferred_milk"] = "almond"

                if "decaf" in q_lower:
                    prefs["caffeine"] = "decaf"
                elif "regular" in q_lower:
                    prefs["caffeine"] = "regular"

                if "sweet" in q_lower:
                    prefs["flavor_profile"] = "sweet"
                elif "strong" in q_lower or "bold" in q_lower:
                    prefs["flavor_profile"] = "bold/strong"

            # 3. Check and execute context bloat compaction
            events = getattr(session, "events", [])
            compacted_events, metrics = self.bloat_manager.compact_history(events)

            if metrics.compaction_performed:
                session.events = compacted_events
                session.state["last_compaction_metrics"] = metrics.model_dump()

            # 4. Persist updated state and events back into session service
            if hasattr(session_service, "sessions") and app_name in session_service.sessions:
                if user_id in session_service.sessions[app_name]:
                    if session_id in session_service.sessions[app_name][user_id]:
                        stored = session_service.sessions[app_name][user_id][session_id]
                        stored.state.update(session.state)
                        stored.events = session.events

            logger.debug(
                "Background memory task completed for session '%s' (compacted=%s)",
                session_id,
                metrics.compaction_performed,
            )
            return metrics

        except Exception as e:
            logger.exception("Error during background memory processing: %s", e)
            return CompactionMetrics()

    def schedule_background_memory(
        self,
        session_service: Any,
        session_id: str,
        user_id: str,
        app_name: str,
        last_user_query: Optional[str] = None,
        last_agent_response: Optional[str] = None,
    ) -> asyncio.Task[CompactionMetrics]:
        """Fire-and-forget background task for memory operations.

        Args:
            session_service: The ADK session service.
            session_id: Session identifier.
            user_id: User identifier.
            app_name: Application name.
            last_user_query: Recent user message.
            last_agent_response: Recent agent response.

        Returns:
            The scheduled asyncio.Task.
        """
        task = asyncio.create_task(
            self.async_process_session_memory(
                session_service=session_service,
                session_id=session_id,
                user_id=user_id,
                app_name=app_name,
                last_user_query=last_user_query,
                last_agent_response=last_agent_response,
            ),
            name=f"background-memory-{session_id[:8]}",
        )
        return task


# Global instances
context_bloat_manager = ContextBloatManager()
background_memory_manager = BackgroundMemoryManager(context_bloat_manager)
