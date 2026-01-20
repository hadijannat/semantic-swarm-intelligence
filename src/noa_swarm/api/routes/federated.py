"""Federated learning routes.

This module provides endpoints for managing federated learning
operations including training rounds and model synchronization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from noa_swarm.api.deps import get_federated_service
from noa_swarm.common.logging import get_logger
from noa_swarm.services.federated import FederatedService

logger = get_logger(__name__)

router = APIRouter()


class FederatedStatus(BaseModel):
    """Model for federated learning status."""

    is_training: bool
    current_round: int
    total_rounds: int
    participating_clients: int
    model_version: str | None = None


class RoundInfo(BaseModel):
    """Model for training round information."""

    round_number: int
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    clients_participated: int
    aggregation_method: str
    metrics: dict[str, float] | None = None


class ClientInfo(BaseModel):
    """Model for federated client information."""

    client_id: str
    status: str
    last_contribution: datetime | None = None
    rounds_participated: int
    local_samples: int


class TrainingConfig(BaseModel):
    """Model for training configuration."""

    num_rounds: int = 10
    min_clients: int = 2
    local_epochs: int = 3
    learning_rate: float = 0.01
    proximal_mu: float = 0.1  # FedProx proximal term
    dp_enabled: bool = False
    dp_epsilon: float | None = None
    dp_delta: float | None = None


class ModelMetrics(BaseModel):
    """Model for model performance metrics."""

    accuracy: float
    f1_score: float
    loss: float
    ece: float | None = None  # Expected Calibration Error


@router.get("/status", response_model=FederatedStatus)
async def get_federated_status(
    service: FederatedService = Depends(get_federated_service),
) -> FederatedStatus:
    """Get the current federated learning status.

    Returns information about ongoing training and model state.
    """
    state = service.status()
    return FederatedStatus(
        is_training=state.is_training,
        current_round=state.current_round,
        total_rounds=state.total_rounds,
        participating_clients=state.participating_clients,
        model_version=state.model_version,
    )


@router.post("/start")
async def start_training(
    config: TrainingConfig,
    service: FederatedService = Depends(get_federated_service),
) -> dict[str, Any]:
    """Start a federated training session.

    Initiates federated learning with the specified configuration.

    Args:
        config: Training configuration options.

    Returns:
        Status of the training session start.
    """
    logger.info(
        "Starting federated training",
        num_rounds=config.num_rounds,
        min_clients=config.min_clients,
        dp_enabled=config.dp_enabled,
    )

    state = service.start(config.num_rounds, config.min_clients)
    return {
        "status": "started",
        "config": config.model_dump(),
        "message": f"Federated training started for {state.total_rounds} rounds",
    }


@router.post("/stop")
async def stop_training(
    service: FederatedService = Depends(get_federated_service),
) -> dict[str, str]:
    """Stop the current federated training session.

    Gracefully stops training after the current round completes.

    Returns:
        Confirmation of training stop.
    """
    service.stop()
    return {
        "status": "stopped",
        "message": "Federated training stopped",
    }


@router.get("/rounds", response_model=list[RoundInfo])
async def list_rounds(
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50, ge=1, le=100, description="Maximum rounds to return"),
    service: FederatedService = Depends(get_federated_service),
) -> list[RoundInfo]:
    """List training rounds.

    Returns the history of federated learning rounds.

    Args:
        offset: Number of rounds to skip.
        limit: Maximum number of rounds to return.

    Returns:
        List of training round information.
    """
    state = service.status()
    if state.total_rounds == 0:
        return []

    return [
        RoundInfo(
            round_number=state.current_round,
            status="running" if state.is_training else "stopped",
            clients_participated=state.participating_clients,
            aggregation_method="fedprox",
        )
    ][offset : offset + limit]


@router.get("/rounds/{round_number}", response_model=RoundInfo)
async def get_round(round_number: int) -> RoundInfo:
    """Get information about a specific training round.

    Args:
        round_number: The round number to retrieve.

    Returns:
        Detailed information about the round.
    """
    return RoundInfo(
        round_number=round_number,
        status="unknown",
        clients_participated=0,
        aggregation_method="fedprox",
    )


@router.get("/clients", response_model=list[ClientInfo])
async def list_clients(
    service: FederatedService = Depends(get_federated_service),
) -> list[ClientInfo]:
    """List federated learning clients.

    Returns information about all registered FL clients.
    """
    return []


@router.get("/clients/{client_id}", response_model=ClientInfo)
async def get_client(client_id: str) -> ClientInfo:
    """Get information about a specific client.

    Args:
        client_id: The ID of the client.

    Returns:
        Detailed information about the client.
    """
    return ClientInfo(
        client_id=client_id,
        status="unknown",
        rounds_participated=0,
        local_samples=0,
    )


@router.get("/model/metrics", response_model=ModelMetrics)
async def get_model_metrics() -> ModelMetrics:
    """Get current model performance metrics.

    Returns the latest metrics from the global model.
    """
    return ModelMetrics(
        accuracy=0.0,
        f1_score=0.0,
        loss=0.0,
        ece=None,
    )


@router.get("/model/version")
async def get_model_version() -> dict[str, Any]:
    """Get the current model version information.

    Returns version and metadata about the current global model.
    """
    return {
        "version": None,
        "created_at": None,
        "rounds_trained": 0,
        "checksum": None,
    }


@router.post("/model/download")
async def download_model() -> dict[str, str]:
    """Download the current global model weights.

    Returns:
        Download URL or status.
    """
    return {
        "status": "not_available",
        "message": "No trained model available",
    }


@router.get("/privacy")
async def get_privacy_budget() -> dict[str, Any]:
    """Get differential privacy budget status.

    Returns the current privacy budget consumption if DP is enabled.
    """
    return {
        "dp_enabled": False,
        "epsilon_budget": None,
        "epsilon_used": None,
        "delta": None,
        "rounds_remaining": None,
    }
