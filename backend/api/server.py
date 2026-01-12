"""FastAPI WebSocket server for DTS visualization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend.api.schemas import SearchRequest
from backend.services.dts_service import run_dts_session
from backend.utils.config import config
from backend.utils.logging import logger

# Models cache (5-minute TTL)
_models_cache: dict[str, Any] = {"data": None, "timestamp": 0}
MODELS_CACHE_TTL = 300  # 5 minutes


app = FastAPI(title="DTS Visualizer API", version="0.1.0")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, data: dict[str, Any]) -> None:
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.warning(f"Failed to send message: {e}")


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for DTS search with real-time updates."""
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "start_search":
                await handle_search(websocket, data.get("config", {}))
            elif data.get("type") == "ping":
                await manager.send_json(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def handle_search(websocket: WebSocket, config_data: dict[str, Any]) -> None:
    """Handle search request by validating and delegating to service layer."""
    # Validate request
    try:
        request = SearchRequest(**config_data)
    except ValidationError as e:
        await manager.send_json(
            websocket,
            {
                "type": "error",
                "data": {"message": "Invalid request", "details": e.errors()},
            },
        )
        return

    # Run search via service layer and stream events
    try:
        async for event in run_dts_session(request):
            await manager.send_json(websocket, event)
    except Exception as e:
        logger.exception("Search failed")
        await manager.send_json(
            websocket,
            {"type": "error", "data": {"message": str(e)}},
        )


# Frontend static files - check for React build first, then fallback to vanilla
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the main index.html (React build or vanilla fallback)."""
    from fastapi.responses import JSONResponse

    # Check React build first
    if FRONTEND_DIST_DIR.exists():
        index_path = FRONTEND_DIST_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

    # Fallback to vanilla frontend
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    return JSONResponse(  # type: ignore
        {"error": "Frontend not found. Run 'npm run build' in frontend/ directory."},
        status_code=404,
    )


# Mount static files - React build assets or vanilla frontend
if FRONTEND_DIST_DIR.exists():
    # React build: mount assets folder
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    # Also serve other static files from dist (like favicon)
    app.mount("/static", StaticFiles(directory=FRONTEND_DIST_DIR), name="static")
elif FRONTEND_DIR.exists():
    # Vanilla fallback
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/config")
async def get_config() -> dict[str, Any]:
    """Get default configuration for frontend."""
    return {
        "defaults": {
            "init_branches": 6,
            "turns_per_branch": 5,
            "user_intents_per_branch": 3,
            "scoring_mode": "comparative",
            "prune_threshold": 6.5,
            "rounds": 1,
        },
        "default_model": config.llm_name,
    }


@app.get("/api/models")
async def get_models() -> dict[str, Any]:
    """Fetch available models from OpenRouter with caching."""
    global _models_cache

    now = time.time()
    if _models_cache["data"] and (now - _models_cache["timestamp"]) < MODELS_CACHE_TTL:
        return _models_cache["data"]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {config.openai_api_key}"},
            )
            response.raise_for_status()
            data = response.json()

        # Filter and format models for frontend
        models = []
        for model in data.get("data", []):
            # Skip non-chat models (embeddings, image-only, etc.)
            arch = model.get("architecture", {})
            input_mods = arch.get("input_modalities", [])
            output_mods = arch.get("output_modalities", [])

            if "text" not in input_mods or "text" not in output_mods:
                continue

            pricing = model.get("pricing", {})
            prompt_cost = float(pricing.get("prompt", 0)) * 1_000_000  # per 1M tokens
            completion_cost = float(pricing.get("completion", 0)) * 1_000_000

            # Check if model supports reasoning tokens
            supported_params = model.get("supported_parameters", [])
            supports_reasoning = "reasoning" in supported_params

            models.append(
                {
                    "id": model.get("id"),
                    "name": model.get("name"),
                    "context_length": model.get("context_length", 0),
                    "prompt_cost": round(prompt_cost, 4),
                    "completion_cost": round(completion_cost, 4),
                    "supports_reasoning": supports_reasoning,
                }
            )

        # Sort by name
        models.sort(key=lambda m: m["name"].lower())

        result = {"models": models, "default_model": config.llm_name}
        _models_cache = {"data": result, "timestamp": now}
        return result

    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to fetch models: {e}")
        return {"models": [], "error": str(e), "default_model": config.llm_name}
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {"models": [], "error": str(e), "default_model": config.llm_name}


def create_app() -> FastAPI:
    """Factory function for creating the app (useful for testing)."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.server_host,
        port=config.server_port,
    )
