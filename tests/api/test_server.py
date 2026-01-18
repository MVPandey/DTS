"""Tests for backend/api/server.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.server import ConnectionManager, app, create_app, handle_search

# -----------------------------------------------------------------------------
# ConnectionManager Tests
# -----------------------------------------------------------------------------


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    def test_init_empty_connections(self) -> None:
        """Test initial state has no connections."""
        manager = ConnectionManager()
        assert manager.active_connections == []

    @pytest.mark.asyncio
    async def test_connect_adds_websocket(self) -> None:
        """Test connecting a websocket."""
        manager = ConnectionManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws)

        mock_ws.accept.assert_called_once()
        assert mock_ws in manager.active_connections

    def test_disconnect_removes_websocket(self) -> None:
        """Test disconnecting a websocket."""
        manager = ConnectionManager()
        mock_ws = MagicMock()
        manager.active_connections.append(mock_ws)

        manager.disconnect(mock_ws)

        assert mock_ws not in manager.active_connections

    def test_disconnect_nonexistent_websocket(self) -> None:
        """Test disconnecting a websocket not in list."""
        manager = ConnectionManager()
        mock_ws = MagicMock()

        # Should not raise
        manager.disconnect(mock_ws)
        assert manager.active_connections == []

    @pytest.mark.asyncio
    async def test_send_json_success(self) -> None:
        """Test sending JSON message."""
        manager = ConnectionManager()
        mock_ws = AsyncMock()

        await manager.send_json(mock_ws, {"type": "test"})

        mock_ws.send_json.assert_called_once_with({"type": "test"})

    @pytest.mark.asyncio
    async def test_send_json_handles_error(self) -> None:
        """Test send_json handles exceptions gracefully."""
        manager = ConnectionManager()
        mock_ws = AsyncMock()
        mock_ws.send_json.side_effect = Exception("Connection closed")

        # Should not raise
        await manager.send_json(mock_ws, {"type": "test"})


# -----------------------------------------------------------------------------
# HTTP Endpoint Tests
# -----------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_returns_ok(self) -> None:
        """Test health endpoint returns status ok."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestConfigEndpoint:
    """Tests for config endpoint."""

    def test_config_returns_defaults(self) -> None:
        """Test config endpoint returns default values."""
        client = TestClient(app)
        response = client.get("/config")

        assert response.status_code == 200
        data = response.json()
        assert "defaults" in data
        assert data["defaults"]["init_branches"] == 6
        assert data["defaults"]["turns_per_branch"] == 5
        assert data["defaults"]["scoring_mode"] == "comparative"


class TestModelsEndpoint:
    """Tests for models endpoint."""

    @pytest.mark.asyncio
    async def test_models_returns_cached(self) -> None:
        """Test models endpoint returns cached data."""
        import time

        from backend.api import server

        # Set up cache
        server._models_cache = {
            "data": {"models": [{"id": "test-model"}], "default_model": "test"},
            "timestamp": time.time(),
        }

        client = TestClient(app)
        response = client.get("/api/models")

        assert response.status_code == 200
        data = response.json()
        assert data["models"] == [{"id": "test-model"}]

        # Clean up cache
        server._models_cache = {"data": None, "timestamp": 0}

    @pytest.mark.asyncio
    async def test_models_fetches_from_api(self) -> None:
        """Test models endpoint fetches from OpenRouter."""
        from backend.api import server

        # Clear cache
        server._models_cache = {"data": None, "timestamp": 0}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "openai/gpt-4",
                    "name": "GPT-4",
                    "context_length": 8192,
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                    "pricing": {"prompt": "0.00003", "completion": "0.00006"},
                    "supported_parameters": ["reasoning"],
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            client = TestClient(app)
            response = client.get("/api/models")

            assert response.status_code == 200
            data = response.json()
            assert "models" in data

        # Clean up
        server._models_cache = {"data": None, "timestamp": 0}

    @pytest.mark.asyncio
    async def test_models_handles_api_error(self) -> None:
        """Test models endpoint handles API errors."""
        import httpx

        from backend.api import server

        # Clear cache
        server._models_cache = {"data": None, "timestamp": 0}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = (
                httpx.HTTPStatusError("Error", request=MagicMock(), response=MagicMock())
            )

            client = TestClient(app)
            response = client.get("/api/models")

            assert response.status_code == 200
            data = response.json()
            assert data["models"] == []
            assert "error" in data


class TestIndexEndpoint:
    """Tests for index endpoint."""

    def test_index_endpoint_exists(self) -> None:
        """Test index endpoint exists and responds."""
        client = TestClient(app)
        response = client.get("/")
        # Response depends on filesystem state - just verify it responds
        assert response.status_code in (200, 404)


# -----------------------------------------------------------------------------
# WebSocket Endpoint Tests
# -----------------------------------------------------------------------------


class TestWebSocketEndpoint:
    """Tests for WebSocket endpoint."""

    def test_websocket_ping_pong(self) -> None:
        """Test ping/pong handling."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "ping"})
            data = websocket.receive_json()
            assert data == {"type": "pong"}

    def test_websocket_disconnect(self) -> None:
        """Test websocket disconnection."""
        client = TestClient(app)

        with client.websocket_connect("/ws"):
            # Just connect and disconnect
            pass


class TestHandleSearch:
    """Tests for handle_search function."""

    @pytest.mark.asyncio
    async def test_handle_search_invalid_request(self) -> None:
        """Test handling invalid search request."""
        mock_ws = AsyncMock()

        await handle_search(mock_ws, {"invalid": "config"})

        # Should send error
        mock_ws.send_json.assert_called()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "error"
        assert "Invalid request" in call_args["data"]["message"]

    @pytest.mark.asyncio
    async def test_handle_search_valid_request(self) -> None:
        """Test handling valid search request."""
        mock_ws = AsyncMock()

        async def mock_run_dts(*_args, **_kwargs):
            yield {"type": "search_started"}
            yield {"type": "search_complete"}

        with patch("backend.api.server.run_dts_session", side_effect=mock_run_dts):
            await handle_search(
                mock_ws,
                {
                    "goal": "Test goal",
                    "first_message": "Test message",
                },
            )

            # Should receive events
            assert mock_ws.send_json.call_count >= 1

    @pytest.mark.asyncio
    async def test_handle_search_error_during_run(self) -> None:
        """Test handling error during search."""
        mock_ws = AsyncMock()

        async def mock_run_dts_error(*_args, **_kwargs):
            raise RuntimeError("Search failed")
            yield  # Make it a generator

        with patch("backend.api.server.run_dts_session", side_effect=mock_run_dts_error):
            await handle_search(
                mock_ws,
                {
                    "goal": "Test goal",
                    "first_message": "Test message",
                },
            )

            # Should send error
            call_args = mock_ws.send_json.call_args[0][0]
            assert call_args["type"] == "error"


# -----------------------------------------------------------------------------
# Factory Function Tests
# -----------------------------------------------------------------------------


class TestCreateApp:
    """Tests for create_app factory."""

    def test_create_app_returns_fastapi(self) -> None:
        """Test factory returns FastAPI instance."""
        from fastapi import FastAPI

        result = create_app()
        assert isinstance(result, FastAPI)
        assert result is app


# -----------------------------------------------------------------------------
# Additional Coverage Tests
# -----------------------------------------------------------------------------


class TestModelsEndpointAdditional:
    """Additional tests for models endpoint coverage."""

    @pytest.mark.asyncio
    async def test_models_filters_non_text_models(self) -> None:
        """Test that models without text modality are filtered out."""
        from backend.api import server

        # Clear cache
        server._models_cache = {"data": None, "timestamp": 0}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "text-model",
                    "name": "Text Model",
                    "context_length": 4096,
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                    "pricing": {"prompt": "0.00001", "completion": "0.00002"},
                    "supported_parameters": [],
                },
                {
                    "id": "image-only-model",
                    "name": "Image Model",
                    "context_length": 0,
                    "architecture": {
                        "input_modalities": ["image"],
                        "output_modalities": ["image"],
                    },
                    "pricing": {},
                    "supported_parameters": [],
                },
                {
                    "id": "embedding-model",
                    "name": "Embedding Model",
                    "context_length": 8192,
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": [],  # No text output
                    },
                    "pricing": {},
                    "supported_parameters": [],
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            client = TestClient(app)
            response = client.get("/api/models")

            assert response.status_code == 200
            data = response.json()
            # Only the text model should be included
            assert len(data["models"]) == 1
            assert data["models"][0]["id"] == "text-model"

        # Clean up
        server._models_cache = {"data": None, "timestamp": 0}

    @pytest.mark.asyncio
    async def test_models_handles_generic_exception(self) -> None:
        """Test models endpoint handles generic exceptions."""
        from backend.api import server

        # Clear cache
        server._models_cache = {"data": None, "timestamp": 0}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = RuntimeError(
                "Network error"
            )

            client = TestClient(app)
            response = client.get("/api/models")

            assert response.status_code == 200
            data = response.json()
            assert data["models"] == []
            assert "Network error" in data["error"]

        # Clean up
        server._models_cache = {"data": None, "timestamp": 0}


class TestWebSocketAdditional:
    """Additional WebSocket tests for coverage."""

    def test_websocket_start_search_triggers_handler(self) -> None:
        """Test that start_search message triggers search handler."""
        client = TestClient(app)

        async def mock_run_dts(*_args, **_kwargs):
            yield {"type": "started", "data": {}}
            yield {"type": "complete", "data": {"result": "done"}}

        with (
            patch("backend.api.server.run_dts_session", side_effect=mock_run_dts),
            client.websocket_connect("/ws") as websocket,
        ):
            websocket.send_json(
                {
                    "type": "start_search",
                    "config": {
                        "goal": "Test goal",
                        "first_message": "Hello",
                    },
                }
            )
            # Receive the events
            data1 = websocket.receive_json()
            assert data1["type"] == "started"
            data2 = websocket.receive_json()
            assert data2["type"] == "complete"

    def test_websocket_unknown_message_type(self) -> None:
        """Test that unknown message types don't crash."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send unknown message type
            websocket.send_json({"type": "unknown_type"})
            # Send ping to verify connection still works
            websocket.send_json({"type": "ping"})
            data = websocket.receive_json()
            assert data == {"type": "pong"}


class TestServeIndexAdditional:
    """Additional tests for serve_index."""

    @pytest.mark.asyncio
    async def test_serve_index_returns_response(self) -> None:
        """Test that serve_index returns some response."""
        from backend.api.server import serve_index

        response = await serve_index()
        # Response type depends on filesystem - just check it's not None
        assert response is not None
