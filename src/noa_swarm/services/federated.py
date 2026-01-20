"""Federated learning service state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class FederatedStatusState:
    """Internal state for federated learning."""

    is_training: bool = False
    current_round: int = 0
    total_rounds: int = 0
    participating_clients: int = 0
    model_version: str | None = None
    started_at: datetime | None = None
    stopped_at: datetime | None = None


class FederatedService:
    """Service that tracks federated learning sessions."""

    def __init__(self) -> None:
        self._state = FederatedStatusState()

    def status(self) -> FederatedStatusState:
        return self._state

    def start(self, total_rounds: int, min_clients: int) -> FederatedStatusState:
        self._state = FederatedStatusState(
            is_training=True,
            current_round=0,
            total_rounds=total_rounds,
            participating_clients=min_clients,
            started_at=datetime.now(timezone.utc),
        )
        return self._state

    def stop(self) -> FederatedStatusState:
        self._state.is_training = False
        self._state.stopped_at = datetime.now(timezone.utc)
        return self._state
