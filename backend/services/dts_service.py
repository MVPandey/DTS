"""DTS service layer for orchestrating search sessions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from backend.api.schemas import SearchRequest
from backend.core.dts.config import DTSConfig
from backend.core.dts.engine import DTSEngine
from backend.llm.client import LLM
from backend.utils.config import config
from backend.utils.logging import logger


def create_llm_client() -> LLM:
    """Create LLM client using global config."""
    return LLM(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        model=config.llm_name,
    )


def create_dts_config(request: SearchRequest) -> DTSConfig:
    """Create DTSConfig from API request."""
    return DTSConfig(
        goal=request.goal,
        first_message=request.first_message,
        init_branches=request.init_branches,
        turns_per_branch=request.turns_per_branch,
        user_intents_per_branch=request.user_intents_per_branch,
        scoring_mode=request.scoring_mode,
        prune_threshold=request.prune_threshold,
        deep_research=request.deep_research,
        strategy_model=request.strategy_model,
        simulator_model=request.simulator_model,
        judge_model=request.judge_model,
    )


async def run_dts_session(request: SearchRequest) -> AsyncIterator[dict[str, Any]]:
    """
    Run a DTS search session and yield events.

    This is the main orchestration function that:
    1. Creates the LLM client with proper config
    2. Creates the DTSConfig from the request
    3. Runs the engine and streams events

    Args:
        request: Validated search request from API

    Yields:
        Event dictionaries to send over WebSocket
    """
    # Create components
    llm = create_llm_client()
    dts_config = create_dts_config(request)
    engine = DTSEngine(llm=llm, config=dts_config)

    # Collect events via callback and yield them
    # Using an async queue to bridge callback to generator
    import asyncio

    event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def event_callback(event_type: str, data: dict[str, Any]) -> None:
        """Callback that puts events into the queue."""
        await event_queue.put({"type": event_type, "data": data})

    engine.set_event_callback(event_callback)

    # Run engine in background task
    async def run_engine() -> dict[str, Any]:
        """Run the engine and return result."""
        try:
            result = await engine.run(rounds=request.rounds)
            return {
                "best_node_id": result.best_node_id,
                "best_score": result.best_score,
                "best_messages": [
                    {"role": m.role, "content": m.content} for m in result.best_messages
                ],
                "pruned_count": result.pruned_count,
                "total_rounds": result.total_rounds,
                "token_usage": result.token_usage,
                "exploration": result.to_exploration_dict(),
            }
        except Exception:
            logger.exception("Engine run failed")
            raise

    # Start engine task
    engine_task = asyncio.create_task(run_engine())

    # Yield events as they come in
    try:
        while True:
            # Check if engine is done
            if engine_task.done():
                # Drain remaining events
                while not event_queue.empty():
                    event = await event_queue.get()
                    if event is not None:
                        yield event

                # Get result or raise exception
                result = await engine_task
                yield {"type": "complete", "data": result}
                break

            # Wait for next event with timeout
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                if event is not None:
                    yield event
            except asyncio.TimeoutError:
                continue

    except Exception as e:
        engine_task.cancel()
        yield {"type": "error", "data": {"message": str(e)}}
        raise
