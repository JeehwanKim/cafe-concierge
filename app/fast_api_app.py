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

import contextlib
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import google.auth
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.cloud import logging as google_cloud_logging

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.app_utils.typing import Feedback
from app.context_memory import background_memory_manager, context_bloat_manager
from app.observability import (
    PiiRedactionLoggingFilter,
    PiiRedactor,
    observability_service,
)

load_dotenv()

# Active PII redaction logging filter installed on root and local loggers
logging.basicConfig(level=logging.INFO)
root_logger = logging.getLogger()
pii_filter = PiiRedactionLoggingFilter()
root_logger.addFilter(pii_filter)

_, project_id = google.auth.default()
logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.agent import app as adk_app
    from app.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name

    # Mount primary A2A RPC endpoint
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    # Mount compatibility alias for default /a2a/app client discovery
    if adk_app.name != "app":
        await attach_a2a_routes(
            app,
            agent=root_agent,
            runner=runner,
            task_store=InMemoryTaskStore(),
            rpc_path="/a2a/app",
        )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=True,
    lifespan=lifespan,
)
app.title = "cafe-concierge"
app.description = "API for interacting with the Agent cafe-concierge"


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback with active PII redaction.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    # Active PII redaction scrubbing emails, phone numbers, credit cards, SSNs, API keys
    sanitized_data = PiiRedactor.redact_data(feedback.model_dump())
    logger.log_struct(sanitized_data, severity="INFO")
    return {"status": "success"}


@app.post("/sessions/{session_id}/compact")
async def compact_session_history(
    session_id: str,
    user_id: str = "default_user",
) -> dict[str, Any]:
    """Execute context bloat management and history compaction for a conversation session.

    Summarizes older conversation turns into a structured context snapshot while preserving
    recent turns verbatim to eliminate context window bloat.
    """
    session_service = services.get_session_service()
    metrics = await background_memory_manager.async_process_session_memory(
        session_service=session_service,
        session_id=session_id,
        user_id=user_id,
        app_name=getattr(app.state, "agent_app_name", "cafe-concierge"),
    )
    return {
        "status": "success",
        "session_id": session_id,
        "compaction_performed": metrics.compaction_performed,
        "metrics": metrics.model_dump(),
    }


@app.post("/sessions/{session_id}/memory/async-update")
def async_update_session_memory(
    session_id: str,
    user_id: str,
    last_user_query: str,
    background_tasks: BackgroundTasks,
    last_agent_response: str | None = None,
) -> dict[str, str]:
    """Trigger background asynchronous memory processing and preference extraction without blocking turn latency."""
    session_service = services.get_session_service()
    background_tasks.add_task(
        background_memory_manager.async_process_session_memory,
        session_service=session_service,
        session_id=session_id,
        user_id=user_id,
        app_name=getattr(app.state, "agent_app_name", "cafe-concierge"),
        last_user_query=last_user_query,
        last_agent_response=last_agent_response,
    )
    return {
        "status": "scheduled",
        "message": "Memory operations running asynchronously in the background.",
    }


@app.post("/observability/intent-outcome")
def capture_intent_outcome(
    session_id: str,
    user_id: str,
    query_text: str,
    response_text: str,
    tools_used: list[str] = [],
    model_routed: str = "gemini-3.8-flash",
    latency_ms: float = 0.0,
) -> dict[str, Any]:
    """Capture Intent vs Outcome with active PII redaction and OpenTelemetry span attributes."""
    record = observability_service.record_turn(
        session_id=session_id,
        user_id=user_id,
        query_text=query_text,
        response_text=response_text,
        tools_used=tools_used,
        model_routed=model_routed,
        latency_ms=latency_ms,
    )
    return {
        "status": "recorded",
        "record": record.model_dump(),
    }


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
