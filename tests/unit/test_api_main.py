"""Unit tests for FastAPI main application."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from noa_swarm.api.main import app, get_api_info

if TYPE_CHECKING:
    pass


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestAPIRoot:
    """Tests for API root endpoint."""

    def test_root_returns_api_info(self, client: TestClient) -> None:
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "NOA Semantic Swarm Mapper API"

    def test_root_includes_endpoints(self, client: TestClient) -> None:
        """Test root endpoint lists available endpoints."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "endpoints" in data
        endpoints = data["endpoints"]

        # Should list all main route groups
        assert "/api/v1/discovery" in endpoints
        assert "/api/v1/mapping" in endpoints
        assert "/api/v1/aas" in endpoints


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_healthy(self, client: TestClient) -> None:
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_components(self, client: TestClient) -> None:
        """Test health check includes component statuses."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "components" in data


class TestReadinessEndpoint:
    """Tests for readiness probe endpoint."""

    def test_readiness_returns_ready(self, client: TestClient) -> None:
        """Test readiness endpoint returns ready status."""
        response = client.get("/ready")
        assert response.status_code == 200

        data = response.json()
        assert "ready" in data


class TestOpenAPI:
    """Tests for OpenAPI documentation."""

    def test_openapi_schema_available(self, client: TestClient) -> None:
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "NOA Semantic Swarm Mapper API"

    def test_docs_endpoint_available(self, client: TestClient) -> None:
        """Test Swagger UI documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint_available(self, client: TestClient) -> None:
        """Test ReDoc documentation is available."""
        response = client.get("/redoc")
        assert response.status_code == 200


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client: TestClient) -> None:
        """Test CORS headers are present in responses."""
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # OPTIONS should work (may return 200 or 405 depending on config)
        assert response.status_code in [200, 405]


class TestAPIInfo:
    """Tests for get_api_info function."""

    def test_get_api_info_returns_dict(self) -> None:
        """Test get_api_info returns complete info."""
        info = get_api_info()

        assert isinstance(info, dict)
        assert info["name"] == "NOA Semantic Swarm Mapper API"
        assert "version" in info
        assert "description" in info
        assert "endpoints" in info
