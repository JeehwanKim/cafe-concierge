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

"""Unit tests for context bloat management, history compaction, and background async memory."""

import asyncio
import pytest
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.context_memory import (
    BackgroundMemoryManager,
    CompactionMetrics,
    ContextBloatManager,
)


def _make_event(role: str, text: str) -> Event:
    content = types.Content(role=role, parts=[types.Part.from_text(text=text)])
    return Event(content=content)


class TestContextBloatManager:
    """Validate bloat detection and history compaction."""

    def test_estimate_tokens(self):
        manager = ContextBloatManager()
        events = [_make_event("user", "Hello world! This is a test conversation turn.")]
        tokens = manager.estimate_tokens(events)
        assert tokens > 0

    def test_should_compact_turn_threshold(self):
        manager = ContextBloatManager(max_turns=3)
        # 3 turns = 6 events (user + model). 7 events should trigger compaction
        events = [
            _make_event("user" if i % 2 == 0 else "model", f"Turn message {i}")
            for i in range(7)
        ]
        assert manager.should_compact(events) is True

    def test_should_not_compact_small_history(self):
        manager = ContextBloatManager(max_turns=6)
        events = [_make_event("user", "Hi"), _make_event("model", "Hello!")]
        assert manager.should_compact(events) is False

    def test_compact_history_reduces_events(self):
        manager = ContextBloatManager(max_turns=2, retain_recent_turns=1)
        # 6 events total
        events = [
            _make_event("user", "I prefer oat milk in my latte."),
            _make_event("model", "Understood, oat milk noted!"),
            _make_event("user", "What is the price of a cold brew?"),
            _make_event("model", "Cold brew is $4.50."),
            _make_event("user", "Can you make that a large?"),
            _make_event("model", "Sure, large cold brew."),
        ]
        compacted, metrics = manager.compact_history(events)

        assert metrics.compaction_performed is True
        assert metrics.original_event_count == 6
        assert len(compacted) < 6
        # Recent turn (2 events) + 1 summary event = 3 events
        assert len(compacted) == 3
        # Check summary event banner
        first_event_text = compacted[0].content.parts[0].text
        assert "[CONTEXT COMPACTED" in first_event_text
        assert "oat milk" in first_event_text.lower()


class TestBackgroundMemoryManager:
    """Validate async non-blocking memory operations."""

    @pytest.mark.asyncio
    async def test_async_process_session_memory_preference_extraction(self):
        session_service = InMemorySessionService()
        session = session_service.create_session_sync(user_id="user_1", app_name="test_app")

        manager = BackgroundMemoryManager()
        metrics = await manager.async_process_session_memory(
            session_service=session_service,
            session_id=session.id,
            user_id="user_1",
            app_name="test_app",
            last_user_query="Can I please get a latte with oat milk?",
        )

        assert isinstance(metrics, CompactionMetrics)
        # Fetch updated session from session service
        updated = await session_service.get_session(
            session_id=session.id, user_id="user_1", app_name="test_app"
        )
        assert updated is not None
        assert "customer_profile" in updated.state
        assert updated.state["customer_profile"].get("preferred_milk") == "oat"

    @pytest.mark.asyncio
    async def test_schedule_background_memory_task(self):
        session_service = InMemorySessionService()
        session = session_service.create_session_sync(user_id="user_2", app_name="test_app")

        manager = BackgroundMemoryManager()
        task = manager.schedule_background_memory(
            session_service=session_service,
            session_id=session.id,
            user_id="user_2",
            app_name="test_app",
            last_user_query="Do you have sweet decaf drinks?",
        )

        assert isinstance(task, asyncio.Task)
        metrics = await task
        assert isinstance(metrics, CompactionMetrics)
        updated = await session_service.get_session(
            session_id=session.id, user_id="user_2", app_name="test_app"
        )
        assert updated is not None
        assert updated.state.get("customer_profile", {}).get("caffeine") == "decaf"
        assert updated.state.get("customer_profile", {}).get("flavor_profile") == "sweet"
