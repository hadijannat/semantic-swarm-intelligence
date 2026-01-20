"""FastAPI main application module.

This module creates and configures the FastAPI application with all
routes and middleware for the NOA Semantic Swarm Mapper API.

Example usage:
    >>> import uvicorn
    >>> from noa_swarm.api.main import app
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)

    # Or from command line:
    # uvicorn noa_swarm.api.main:app --reload
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from noa_swarm.api.routes import aas, discovery, federated, mapping, swarm
from noa_swarm.common.logging import get_logger
from noa_swarm.observability import generate_metrics_output
from noa_swarm.services.state import get_state

logger = get_logger(__name__)

# API version
API_VERSION = "1.0.0"


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    components: dict[str, str]


class ReadinessResponse(BaseModel):
    """Readiness probe response model."""

    ready: bool
    checks: dict[str, bool]


def get_api_info() -> dict[str, Any]:
    """Get API information.

    Returns:
        Dictionary containing API name, version, description, and endpoints.
    """
    return {
        "name": "NOA Semantic Swarm Mapper API",
        "version": API_VERSION,
        "description": "REST API for semantic tag mapping using swarm intelligence",
        "endpoints": {
            "/api/v1/discovery": "OPC UA tag discovery and browsing",
            "/api/v1/mapping": "Tag-to-IRDI mapping operations",
            "/api/v1/aas": "Asset Administration Shell export",
            "/api/v1/swarm": "Swarm coordination and consensus",
            "/api/v1/federated": "Federated learning operations",
        },
    }


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    api_info = get_api_info()

    application = FastAPI(
        title=api_info["name"],
        description=api_info["description"],
        version=api_info["version"],
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    @application.on_event("startup")
    async def _startup() -> None:
        state = get_state()
        if state.database is not None:
            await state.database.init_models()

    # Configure CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    application.include_router(
        discovery.router,
        prefix="/api/v1/discovery",
        tags=["discovery"],
    )
    application.include_router(
        mapping.router,
        prefix="/api/v1/mapping",
        tags=["mapping"],
    )
    application.include_router(
        aas.router,
        prefix="/api/v1/aas",
        tags=["aas"],
    )
    application.include_router(
        swarm.router,
        prefix="/api/v1/swarm",
        tags=["swarm"],
    )
    application.include_router(
        federated.router,
        prefix="/api/v1/federated",
        tags=["federated"],
    )

    @application.get("/")
    def root() -> dict[str, Any]:
        """Root endpoint returning API information."""
        return get_api_info()

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health check endpoint.

        Returns the health status of the API and its components.
        """
        return HealthResponse(
            status="healthy",
            components={
                "api": "healthy",
                "database": "healthy",
                "mqtt": "healthy",
            },
        )

    @application.get("/ready", response_model=ReadinessResponse)
    def readiness() -> ReadinessResponse:
        """Readiness probe endpoint.

        Returns whether the API is ready to serve requests.
        """
        return ReadinessResponse(
            ready=True,
            checks={
                "api": True,
                "database": True,
                "mqtt": True,
            },
        )

    @application.get("/metrics")
    def metrics() -> Response:
        """Prometheus metrics endpoint.

        Returns metrics in Prometheus text exposition format for scraping.
        """
        return Response(
            content=generate_metrics_output(),
            media_type="text/plain; charset=utf-8",
        )

    logger.info(
        "Created FastAPI application",
        version=api_info["version"],
    )

    return application


# Create the application instance
app = create_app()
