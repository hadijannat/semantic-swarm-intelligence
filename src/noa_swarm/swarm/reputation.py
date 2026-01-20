"""Agent Reliability Scoring for swarm consensus.

This module provides reputation tracking for swarm agents based on their
prediction accuracy. More reliable agents have greater influence in consensus.

Key components:
- **ReputationConfig**: Configuration for scoring parameters
- **AgentOutcome**: Record of a single prediction outcome
- **AgentReputation**: Full reputation record for an agent
- **ReputationTracker**: Main class for tracking and updating reputations

Example usage:
    >>> from noa_swarm.swarm.reputation import ReputationTracker, ReputationConfig
    >>> config = ReputationConfig(window_size=100)
    >>> tracker = ReputationTracker(config)
    >>> tracker.record_outcome("agent-001", "tag-123", "0173-1#01-ABA234#001", "0173-1#01-ABA234#001")
    >>> score = tracker.get_reliability("agent-001")
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from noa_swarm.common.logging import get_logger

logger = get_logger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class ReputationConfig:
    """Configuration for the reputation tracking system.

    Attributes:
        window_size: Number of recent outcomes to track per agent.
        initial_score: Starting reliability score for new agents.
        min_score: Minimum allowed reliability score (floor).
        max_score: Maximum allowed reliability score (ceiling).
        decay_factor: Exponential weight for newer outcomes (0-1).
        agreement_bonus: Score increase for agreeing with consensus.
        disagreement_penalty: Score decrease for disagreeing with consensus.
    """

    window_size: int = 100
    initial_score: float = 0.5
    min_score: float = 0.1
    max_score: float = 0.95
    decay_factor: float = 0.95
    agreement_bonus: float = 0.02
    disagreement_penalty: float = 0.03

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {self.window_size}")
        if not 0.0 <= self.initial_score <= 1.0:
            raise ValueError(
                f"initial_score must be between 0.0 and 1.0, got {self.initial_score}"
            )
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(
                f"min_score must be between 0.0 and 1.0, got {self.min_score}"
            )
        if not 0.0 <= self.max_score <= 1.0:
            raise ValueError(
                f"max_score must be between 0.0 and 1.0, got {self.max_score}"
            )
        if self.min_score > self.max_score:
            raise ValueError(
                f"min_score ({self.min_score}) cannot be greater than "
                f"max_score ({self.max_score})"
            )
        if not 0.0 < self.decay_factor <= 1.0:
            raise ValueError(
                f"decay_factor must be between 0.0 (exclusive) and 1.0, "
                f"got {self.decay_factor}"
            )
        if self.agreement_bonus < 0:
            raise ValueError(
                f"agreement_bonus must be non-negative, got {self.agreement_bonus}"
            )
        if self.disagreement_penalty < 0:
            raise ValueError(
                f"disagreement_penalty must be non-negative, "
                f"got {self.disagreement_penalty}"
            )


@dataclass
class AgentOutcome:
    """Record of a single prediction outcome.

    Attributes:
        agent_id: Identifier of the agent that made the prediction.
        tag_id: Identifier of the tag being predicted.
        predicted_irdi: The IRDI predicted by the agent.
        final_irdi: The IRDI determined by consensus.
        timestamp: When the outcome was recorded.
    """

    agent_id: str
    tag_id: str
    predicted_irdi: str
    final_irdi: str
    timestamp: datetime = field(default_factory=_utc_now)

    @property
    def was_correct(self) -> bool:
        """Return whether the prediction matched the final consensus."""
        return self.predicted_irdi == self.final_irdi

    def __post_init__(self) -> None:
        """Validate outcome fields."""
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
        if not self.tag_id:
            raise ValueError("tag_id cannot be empty")
        if not self.predicted_irdi:
            raise ValueError("predicted_irdi cannot be empty")
        if not self.final_irdi:
            raise ValueError("final_irdi cannot be empty")


@dataclass
class AgentReputation:
    """Full reputation record for an agent.

    Attributes:
        agent_id: Identifier of the agent.
        reliability_score: Current computed reliability score (0-1).
        outcomes: List of recent prediction outcomes.
        total_predictions: Total number of predictions made.
        correct_predictions: Total number of correct predictions.
        created_at: When the reputation record was created.
        updated_at: When the reputation was last updated.
    """

    agent_id: str
    reliability_score: float
    outcomes: list[AgentOutcome] = field(default_factory=list)
    total_predictions: int = 0
    correct_predictions: int = 0
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Validate reputation fields."""
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
        if not 0.0 <= self.reliability_score <= 1.0:
            raise ValueError(
                f"reliability_score must be between 0.0 and 1.0, "
                f"got {self.reliability_score}"
            )

    @property
    def accuracy(self) -> float:
        """Return the overall accuracy (correct/total).

        Returns 0.0 if no predictions have been made.
        """
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max bounds."""
    return max(min_val, min(value, max_val))


def _compute_score(
    outcomes: list[AgentOutcome],
    decay: float,
    initial_score: float,
    min_score: float,
    max_score: float,
) -> float:
    """Compute exponentially weighted accuracy score.

    Args:
        outcomes: List of prediction outcomes (older first).
        decay: Exponential decay factor (0-1).
        initial_score: Score to return if no outcomes.
        min_score: Minimum allowed score.
        max_score: Maximum allowed score.

    Returns:
        Computed reliability score, clamped to [min_score, max_score].
    """
    if not outcomes:
        return initial_score

    weighted_sum = 0.0
    weight_total = 0.0

    # Iterate most recent first (reversed)
    for i, outcome in enumerate(reversed(outcomes)):
        weight = decay**i
        weighted_sum += weight * (1.0 if outcome.was_correct else 0.0)
        weight_total += weight

    raw_score = weighted_sum / weight_total if weight_total > 0 else initial_score
    return _clamp(raw_score, min_score, max_score)


class ReputationTracker:
    """Tracker for agent reliability scores.

    This class maintains reputation records for swarm agents based on their
    prediction accuracy. Scores are computed using exponentially weighted
    accuracy, giving more weight to recent outcomes.

    The tracker is thread-safe for concurrent access.

    Example:
        >>> tracker = ReputationTracker()
        >>> tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")
        >>> tracker.get_reliability("agent-001")
        0.95  # High score for correct prediction
    """

    def __init__(self, config: ReputationConfig | None = None) -> None:
        """Initialize the reputation tracker.

        Args:
            config: Configuration for scoring parameters.
                   Uses default ReputationConfig if not provided.
        """
        self.config = config or ReputationConfig()
        self._reputations: dict[str, AgentReputation] = {}
        self._lock = threading.RLock()

        logger.debug(
            "ReputationTracker initialized",
            window_size=self.config.window_size,
            initial_score=self.config.initial_score,
            decay_factor=self.config.decay_factor,
        )

    def get_reliability(self, agent_id: str) -> float:
        """Get an agent's current reliability score.

        Returns the initial_score for unknown agents.

        Args:
            agent_id: Identifier of the agent.

        Returns:
            The agent's reliability score (0-1).
        """
        with self._lock:
            reputation = self._reputations.get(agent_id)
            if reputation is None:
                return self.config.initial_score
            return reputation.reliability_score

    def get_reputation(self, agent_id: str) -> AgentReputation | None:
        """Get the full reputation record for an agent.

        Args:
            agent_id: Identifier of the agent.

        Returns:
            The agent's reputation record, or None if not found.
        """
        with self._lock:
            return self._reputations.get(agent_id)

    def record_outcome(
        self,
        agent_id: str,
        tag_id: str,
        predicted_irdi: str,
        final_irdi: str,
    ) -> AgentReputation:
        """Record a prediction outcome for an agent.

        This creates or updates the agent's reputation record based on
        whether their prediction matched the final consensus.

        Args:
            agent_id: Identifier of the agent.
            tag_id: Identifier of the tag.
            predicted_irdi: The IRDI predicted by the agent.
            final_irdi: The IRDI determined by consensus.

        Returns:
            The updated AgentReputation record.
        """
        outcome = AgentOutcome(
            agent_id=agent_id,
            tag_id=tag_id,
            predicted_irdi=predicted_irdi,
            final_irdi=final_irdi,
        )

        with self._lock:
            # Get or create reputation
            reputation = self._reputations.get(agent_id)
            if reputation is None:
                reputation = AgentReputation(
                    agent_id=agent_id,
                    reliability_score=self.config.initial_score,
                )
                self._reputations[agent_id] = reputation

            # Add outcome (maintain window size)
            reputation.outcomes.append(outcome)
            if len(reputation.outcomes) > self.config.window_size:
                reputation.outcomes = reputation.outcomes[-self.config.window_size :]

            # Update counters
            reputation.total_predictions += 1
            if outcome.was_correct:
                reputation.correct_predictions += 1

            # Recompute score
            reputation.reliability_score = _compute_score(
                reputation.outcomes,
                self.config.decay_factor,
                self.config.initial_score,
                self.config.min_score,
                self.config.max_score,
            )
            reputation.updated_at = _utc_now()

            logger.debug(
                "Recorded outcome",
                agent_id=agent_id,
                tag_id=tag_id,
                was_correct=outcome.was_correct,
                new_score=reputation.reliability_score,
            )

            return reputation

    def update_scores(self) -> None:
        """Recalculate all scores based on recent outcomes.

        This is useful for batch recomputation after configuration changes
        or data imports.
        """
        with self._lock:
            for _agent_id, reputation in self._reputations.items():
                reputation.reliability_score = _compute_score(
                    reputation.outcomes,
                    self.config.decay_factor,
                    self.config.initial_score,
                    self.config.min_score,
                    self.config.max_score,
                )
                reputation.updated_at = _utc_now()

            logger.debug(
                "Updated scores for all agents",
                agent_count=len(self._reputations),
            )

    def get_all_reputations(self) -> dict[str, AgentReputation]:
        """Get all tracked reputation records.

        Returns:
            Dictionary mapping agent_id to AgentReputation.
        """
        with self._lock:
            return dict(self._reputations)

    def bootstrap_agent(
        self,
        agent_id: str,
        initial_score: float | None = None,
    ) -> AgentReputation:
        """Explicitly create a reputation record for a new agent.

        This allows setting up agents with custom initial scores before
        they make any predictions.

        Args:
            agent_id: Identifier of the agent.
            initial_score: Optional custom initial score. Uses config
                          default if not provided.

        Returns:
            The created AgentReputation record.

        Raises:
            ValueError: If the agent already has a reputation record.
        """
        score = initial_score if initial_score is not None else self.config.initial_score

        # Validate score bounds
        score = _clamp(score, self.config.min_score, self.config.max_score)

        with self._lock:
            if agent_id in self._reputations:
                raise ValueError(f"Agent '{agent_id}' already has a reputation record")

            reputation = AgentReputation(
                agent_id=agent_id,
                reliability_score=score,
            )
            self._reputations[agent_id] = reputation

            logger.info(
                "Bootstrapped agent",
                agent_id=agent_id,
                initial_score=score,
            )

            return reputation

    def prune_old_outcomes(self, max_age_hours: float = 168.0) -> int:
        """Remove outcomes older than the specified age.

        This helps manage memory by removing stale outcome data while
        preserving the aggregate statistics.

        Args:
            max_age_hours: Maximum age in hours for outcomes to retain.
                          Defaults to 168 hours (7 days).

        Returns:
            Number of outcomes pruned across all agents.
        """
        cutoff = _utc_now() - timedelta(hours=max_age_hours)
        total_pruned = 0

        with self._lock:
            for _agent_id, reputation in self._reputations.items():
                original_count = len(reputation.outcomes)
                reputation.outcomes = [
                    o for o in reputation.outcomes if o.timestamp >= cutoff
                ]
                pruned = original_count - len(reputation.outcomes)

                if pruned > 0:
                    # Recompute score after pruning
                    reputation.reliability_score = _compute_score(
                        reputation.outcomes,
                        self.config.decay_factor,
                        self.config.initial_score,
                        self.config.min_score,
                        self.config.max_score,
                    )
                    reputation.updated_at = _utc_now()
                    total_pruned += pruned

            if total_pruned > 0:
                logger.info(
                    "Pruned old outcomes",
                    total_pruned=total_pruned,
                    max_age_hours=max_age_hours,
                )

        return total_pruned

    def clear(self) -> None:
        """Clear all reputation data.

        This removes all tracked agents and their outcomes.
        """
        with self._lock:
            self._reputations.clear()
            logger.info("Cleared all reputation data")
