"""Consensus Engine for swarm-based semantic mapping decisions.

This module provides confidence-weighted voting and quorum logic for reaching
consensus across multiple swarm agents on tag-to-IRDI mappings.

Key components:
- **ConsensusConfig**: Configuration for consensus thresholds and weights
- **WeightedVote**: A vote with computed weights from calibration, reliability, and freshness
- **ConsensusEngine**: Main engine for aggregating votes and determining quorum

Example usage:
    >>> from noa_swarm.swarm.consensus import ConsensusEngine, ConsensusConfig
    >>> config = ConsensusConfig(hard_quorum_threshold=0.8)
    >>> engine = ConsensusEngine(config)
    >>> record = engine.reach_consensus(tag_id, votes, calibration_factors)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from noa_swarm.common.logging import get_logger
from noa_swarm.common.schemas import ConsensusRecord, QuorumType, Vote, utc_now

logger = get_logger(__name__)


@dataclass(frozen=True)
class ConsensusConfig:
    """Configuration for the consensus engine.

    Attributes:
        hard_quorum_threshold: Threshold for unanimous agreement (0.0-1.0).
        soft_quorum_threshold: Threshold for majority agreement (0.0-1.0).
        min_votes: Minimum number of votes required for consensus.
        freshness_decay_hours: Half-life for time decay in hours.
        calibration_weight: Weight of calibration factor in combined weight.
        reliability_weight: Weight of agent reliability in combined weight.
        confidence_weight: Weight of raw confidence in combined weight.
    """

    hard_quorum_threshold: float = 0.8
    soft_quorum_threshold: float = 0.5
    min_votes: int = 2
    freshness_decay_hours: float = 24.0
    calibration_weight: float = 0.3
    reliability_weight: float = 0.5
    confidence_weight: float = 0.2

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.hard_quorum_threshold <= 1.0:
            raise ValueError(
                f"hard_quorum_threshold must be between 0.0 and 1.0, "
                f"got {self.hard_quorum_threshold}"
            )
        if not 0.0 <= self.soft_quorum_threshold <= 1.0:
            raise ValueError(
                f"soft_quorum_threshold must be between 0.0 and 1.0, "
                f"got {self.soft_quorum_threshold}"
            )
        if self.soft_quorum_threshold > self.hard_quorum_threshold:
            raise ValueError(
                f"soft_quorum_threshold ({self.soft_quorum_threshold}) cannot be "
                f"greater than hard_quorum_threshold ({self.hard_quorum_threshold})"
            )
        if self.min_votes < 1:
            raise ValueError(f"min_votes must be at least 1, got {self.min_votes}")
        if self.freshness_decay_hours <= 0:
            raise ValueError(
                f"freshness_decay_hours must be positive, got {self.freshness_decay_hours}"
            )

        # Validate weights sum to approximately 1.0 (with small tolerance for floating point)
        total_weight = self.calibration_weight + self.reliability_weight + self.confidence_weight
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight} "
                f"(calibration={self.calibration_weight}, "
                f"reliability={self.reliability_weight}, "
                f"confidence={self.confidence_weight})"
            )


@dataclass(frozen=True)
class WeightedVote:
    """A vote with computed weights from calibration, reliability, and freshness.

    Attributes:
        vote: The original Vote object.
        calibration_factor: Model calibration quality (e.g., 1 - ECE), 0-1.
        freshness_factor: Time decay factor, 0-1.
        combined_weight: Final computed weight for this vote.
    """

    vote: Vote
    calibration_factor: float
    freshness_factor: float
    combined_weight: float

    def __post_init__(self) -> None:
        """Validate weight values."""
        if not 0.0 <= self.calibration_factor <= 1.0:
            raise ValueError(
                f"calibration_factor must be between 0.0 and 1.0, " f"got {self.calibration_factor}"
            )
        if not 0.0 <= self.freshness_factor <= 1.0:
            raise ValueError(
                f"freshness_factor must be between 0.0 and 1.0, " f"got {self.freshness_factor}"
            )


class ConsensusEngine:
    """Engine for confidence-weighted voting and quorum determination.

    The consensus engine aggregates votes from multiple swarm agents using
    a weighted voting scheme that considers:
    - Model calibration quality (how well-calibrated the confidence scores are)
    - Agent reliability (historical accuracy of the agent)
    - Vote confidence (the agent's confidence in the prediction)
    - Freshness (time since the vote was cast)

    Example:
        >>> engine = ConsensusEngine()
        >>> votes = [vote1, vote2, vote3]
        >>> calibration_factors = {"agent-001": 0.9, "agent-002": 0.85}
        >>> record = engine.reach_consensus("tag-123", votes, calibration_factors)
        >>> print(record.quorum_type)  # 'hard', 'soft', or 'conflict'
    """

    def __init__(self, config: ConsensusConfig | None = None) -> None:
        """Initialize the consensus engine.

        Args:
            config: Configuration for consensus thresholds and weights.
                   Uses default ConsensusConfig if not provided.
        """
        self.config: ConsensusConfig = config or ConsensusConfig()
        logger.debug(
            "ConsensusEngine initialized",
            hard_quorum=self.config.hard_quorum_threshold,
            soft_quorum=self.config.soft_quorum_threshold,
            min_votes=self.config.min_votes,
        )

    def compute_freshness_factor(
        self,
        vote_timestamp: datetime,
        reference_time: datetime | None = None,
    ) -> float:
        """Compute the freshness decay factor for a vote.

        Uses exponential decay with configurable half-life:
        freshness = 0.5 ** (hours_old / half_life)

        Args:
            vote_timestamp: When the vote was cast.
            reference_time: Reference time for computing age. Defaults to now.

        Returns:
            Freshness factor between 0 and 1 (1 = fresh, 0 = very old).
        """
        if reference_time is None:
            reference_time = utc_now()

        # Ensure both timestamps are timezone-aware
        if vote_timestamp.tzinfo is None:
            vote_timestamp = vote_timestamp.replace(tzinfo=UTC)
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=UTC)

        # Calculate age in hours
        age_seconds = (reference_time - vote_timestamp).total_seconds()
        hours_old = max(0.0, age_seconds / 3600.0)  # Clamp to non-negative

        # Exponential decay with half-life
        decay_hours = float(self.config.freshness_decay_hours)
        freshness = float(0.5 ** (hours_old / decay_hours))

        return freshness

    def compute_weighted_vote(
        self,
        vote: Vote,
        calibration_factor: float,
        reference_time: datetime | None = None,
    ) -> WeightedVote:
        """Compute weighted vote with calibration, reliability, and freshness.

        The combined weight formula is:
        combined_weight = (
            calibration_weight * calibration_factor +
            reliability_weight * reliability_score +
            confidence_weight * confidence
        ) * freshness_factor

        Args:
            vote: The vote to weight.
            calibration_factor: Model calibration quality (e.g., 1 - ECE).
            reference_time: Reference time for freshness computation.

        Returns:
            WeightedVote with computed weights.
        """
        freshness_factor = self.compute_freshness_factor(vote.timestamp, reference_time)

        # Compute base weight from calibration, reliability, and confidence
        base_weight = (
            self.config.calibration_weight * calibration_factor
            + self.config.reliability_weight * vote.reliability_score
            + self.config.confidence_weight * vote.confidence
        )

        # Apply freshness decay
        combined_weight = base_weight * freshness_factor

        return WeightedVote(
            vote=vote,
            calibration_factor=calibration_factor,
            freshness_factor=freshness_factor,
            combined_weight=combined_weight,
        )

    def aggregate_votes(
        self,
        votes: list[Vote],
        calibration_factors: dict[str, float],
        reference_time: datetime | None = None,
    ) -> tuple[dict[str, float], float, list[WeightedVote]]:
        """Aggregate votes into weighted scores per IRDI.

        Args:
            votes: List of votes to aggregate.
            calibration_factors: Mapping of agent_id to calibration factor.
                               Agents not in this dict default to 0.5.
            reference_time: Reference time for freshness computation.

        Returns:
            Tuple of:
            - Dict mapping IRDI to aggregated weighted score
            - Total weight across all votes
            - List of WeightedVote objects
        """
        irdi_scores: dict[str, float] = {}
        total_weight = 0.0
        weighted_votes: list[WeightedVote] = []

        for vote in votes:
            # Get calibration factor for this agent (default to 0.5 if unknown)
            calibration = calibration_factors.get(vote.agent_id, 0.5)

            weighted_vote = self.compute_weighted_vote(vote, calibration, reference_time)
            weighted_votes.append(weighted_vote)

            # Aggregate by IRDI
            irdi = vote.candidate_irdi
            if irdi not in irdi_scores:
                irdi_scores[irdi] = 0.0
            irdi_scores[irdi] += weighted_vote.combined_weight
            total_weight += weighted_vote.combined_weight

        return irdi_scores, total_weight, weighted_votes

    def determine_quorum(
        self,
        aggregated_scores: dict[str, float],
        total_weight: float,
        votes: list[Vote],
    ) -> tuple[str | None, QuorumType, float]:
        """Determine quorum type from aggregated scores.

        Quorum rules:
        - hard: Single IRDI has >= hard_quorum_threshold of total weighted votes
        - soft: Single IRDI has >= soft_quorum_threshold but < hard_quorum_threshold
        - conflict: No IRDI reaches soft_quorum_threshold, or tie between top

        Args:
            aggregated_scores: Dict mapping IRDI to aggregated weighted score.
            total_weight: Total weight across all votes.
            votes: Original votes (for minimum vote check).

        Returns:
            Tuple of (winning_irdi, quorum_type, consensus_confidence).
            winning_irdi may be None for conflict with no clear winner.
        """
        # Check minimum votes requirement
        if len(votes) < self.config.min_votes:
            logger.debug(
                "Insufficient votes for consensus",
                vote_count=len(votes),
                min_required=self.config.min_votes,
            )
            # Return the top candidate but mark as conflict
            if aggregated_scores:
                top_irdi = max(aggregated_scores, key=lambda k: aggregated_scores[k])
                confidence = aggregated_scores[top_irdi] / total_weight if total_weight > 0 else 0
                return top_irdi, "conflict", confidence
            return None, "conflict", 0.0

        # Handle edge case: no votes or zero total weight
        if not aggregated_scores or total_weight == 0:
            return None, "conflict", 0.0

        # Sort IRDIs by score (descending)
        sorted_irdis = sorted(
            aggregated_scores.keys(),
            key=lambda k: aggregated_scores[k],
            reverse=True,
        )

        top_irdi = sorted_irdis[0]
        top_score = aggregated_scores[top_irdi]
        top_proportion = top_score / total_weight

        # Check for tie with second place
        if len(sorted_irdis) > 1:
            second_score = aggregated_scores[sorted_irdis[1]]
            # Consider it a tie if scores are very close (within 1%)
            if abs(top_score - second_score) / total_weight < 0.01:
                logger.debug(
                    "Tie detected between top candidates",
                    top_irdi=top_irdi,
                    top_score=top_score,
                    second_irdi=sorted_irdis[1],
                    second_score=second_score,
                )
                return top_irdi, "conflict", top_proportion

        # Determine quorum type based on proportion
        if top_proportion >= self.config.hard_quorum_threshold:
            quorum_type: QuorumType = "hard"
        elif top_proportion >= self.config.soft_quorum_threshold:
            quorum_type = "soft"
        else:
            quorum_type = "conflict"

        logger.debug(
            "Quorum determined",
            irdi=top_irdi,
            proportion=top_proportion,
            quorum_type=quorum_type,
        )

        return top_irdi, quorum_type, top_proportion

    def reach_consensus(
        self,
        tag_id: str,
        votes: list[Vote],
        calibration_factors: dict[str, float],
        reference_time: datetime | None = None,
    ) -> ConsensusRecord:
        """Reach consensus on a tag mapping through weighted voting.

        This is the main entry point for the consensus process. It:
        1. Computes weighted votes using calibration, reliability, and freshness
        2. Aggregates votes by IRDI
        3. Determines quorum type (hard, soft, or conflict)
        4. Creates a ConsensusRecord with audit trail

        Args:
            tag_id: Unique identifier for the tag being mapped.
            votes: List of votes from swarm agents.
            calibration_factors: Mapping of agent_id to calibration factor.
            reference_time: Reference time for freshness computation.

        Returns:
            ConsensusRecord with the consensus decision and audit trail.

        Raises:
            ValueError: If no votes are provided.
        """
        if not votes:
            raise ValueError("Cannot reach consensus with no votes")

        logger.info(
            "Starting consensus process",
            tag_id=tag_id,
            vote_count=len(votes),
            unique_agents=len({v.agent_id for v in votes}),
        )

        # Build initial audit trail
        audit_trail: list[str] = []
        timestamp = utc_now()
        audit_trail.append(
            f"{timestamp.isoformat()}: Consensus process started with {len(votes)} votes"
        )

        # Aggregate votes
        aggregated_scores, total_weight, weighted_votes = self.aggregate_votes(
            votes, calibration_factors, reference_time
        )

        # Log aggregation details
        for irdi, score in sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True):
            proportion = score / total_weight if total_weight > 0 else 0
            audit_trail.append(
                f"{timestamp.isoformat()}: IRDI {irdi} received "
                f"{score:.4f} weighted score ({proportion:.1%} of total)"
            )

        # Determine quorum
        winning_irdi, quorum_type, consensus_confidence = self.determine_quorum(
            aggregated_scores, total_weight, votes
        )

        # Handle edge case where no winner is determined
        if winning_irdi is None:
            # Use a placeholder IRDI for conflict with no votes
            # This should rarely happen in practice
            winning_irdi = votes[0].candidate_irdi if votes else "0173-1#00-UNKNOWN#000"

        audit_trail.append(
            f"{timestamp.isoformat()}: Quorum type '{quorum_type}' achieved "
            f"with confidence {consensus_confidence:.1%}"
        )

        # Create consensus record
        record = ConsensusRecord(
            tag_id=tag_id,
            agreed_irdi=winning_irdi,
            consensus_confidence=consensus_confidence,
            votes=votes,
            quorum_type=quorum_type,
            audit_trail=audit_trail,
            created_at=timestamp,
            updated_at=timestamp,
        )

        logger.info(
            "Consensus reached",
            tag_id=tag_id,
            agreed_irdi=winning_irdi,
            quorum_type=quorum_type,
            confidence=consensus_confidence,
        )

        return record

    def add_audit_entry(self, record: ConsensusRecord, entry: str) -> ConsensusRecord:
        """Add an audit entry to a consensus record.

        This is a convenience wrapper around ConsensusRecord.add_audit_entry
        that also logs the action.

        Args:
            record: The consensus record to update.
            entry: The audit entry to add.

        Returns:
            New ConsensusRecord with the entry added.
        """
        logger.debug(
            "Adding audit entry",
            tag_id=record.tag_id,
            entry=entry,
        )
        return record.add_audit_entry(entry)


# Custom exceptions for consensus-related errors
class ConsensusError(Exception):
    """Base exception for consensus-related errors."""

    pass


class InsufficientVotesError(ConsensusError):
    """Raised when there are not enough votes to reach consensus."""

    def __init__(self, vote_count: int, min_required: int) -> None:
        self.vote_count = vote_count
        self.min_required = min_required
        super().__init__(f"Insufficient votes: {vote_count} provided, {min_required} required")


class NoConsensusError(ConsensusError):
    """Raised when consensus cannot be reached due to conflicting votes."""

    def __init__(self, tag_id: str, top_candidates: list[tuple[str, float]]) -> None:
        self.tag_id = tag_id
        self.top_candidates = top_candidates
        super().__init__(
            f"No consensus reached for tag {tag_id}. " f"Top candidates: {top_candidates}"
        )
