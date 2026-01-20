"""Unit tests for the consensus engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from noa_swarm.common.schemas import ConsensusRecord, Vote
from noa_swarm.swarm.consensus import (
    ConsensusConfig,
    ConsensusEngine,
    ConsensusError,
    InsufficientVotesError,
    NoConsensusError,
    WeightedVote,
)


class TestConsensusConfig:
    """Tests for ConsensusConfig dataclass."""

    def test_default_values(self) -> None:
        """Test ConsensusConfig has expected default values."""
        config = ConsensusConfig()

        assert config.hard_quorum_threshold == 0.8
        assert config.soft_quorum_threshold == 0.5
        assert config.min_votes == 2
        assert config.freshness_decay_hours == 24.0
        assert config.calibration_weight == 0.3
        assert config.reliability_weight == 0.5
        assert config.confidence_weight == 0.2

    def test_custom_values(self) -> None:
        """Test ConsensusConfig accepts custom values."""
        config = ConsensusConfig(
            hard_quorum_threshold=0.9,
            soft_quorum_threshold=0.6,
            min_votes=3,
            freshness_decay_hours=12.0,
            calibration_weight=0.4,
            reliability_weight=0.4,
            confidence_weight=0.2,
        )

        assert config.hard_quorum_threshold == 0.9
        assert config.soft_quorum_threshold == 0.6
        assert config.min_votes == 3
        assert config.freshness_decay_hours == 12.0
        assert config.calibration_weight == 0.4
        assert config.reliability_weight == 0.4
        assert config.confidence_weight == 0.2

    def test_hard_quorum_threshold_validation(self) -> None:
        """Test hard_quorum_threshold must be between 0 and 1."""
        with pytest.raises(ValueError, match="hard_quorum_threshold"):
            ConsensusConfig(hard_quorum_threshold=1.5)

        with pytest.raises(ValueError, match="hard_quorum_threshold"):
            ConsensusConfig(hard_quorum_threshold=-0.1)

    def test_soft_quorum_threshold_validation(self) -> None:
        """Test soft_quorum_threshold must be between 0 and 1."""
        with pytest.raises(ValueError, match="soft_quorum_threshold"):
            ConsensusConfig(soft_quorum_threshold=1.5)

        with pytest.raises(ValueError, match="soft_quorum_threshold"):
            ConsensusConfig(soft_quorum_threshold=-0.1)

    def test_soft_cannot_exceed_hard_threshold(self) -> None:
        """Test soft_quorum_threshold cannot exceed hard_quorum_threshold."""
        with pytest.raises(ValueError, match="soft_quorum_threshold.*cannot be greater"):
            ConsensusConfig(
                hard_quorum_threshold=0.5,
                soft_quorum_threshold=0.6,
            )

    def test_min_votes_validation(self) -> None:
        """Test min_votes must be at least 1."""
        with pytest.raises(ValueError, match="min_votes"):
            ConsensusConfig(min_votes=0)

        with pytest.raises(ValueError, match="min_votes"):
            ConsensusConfig(min_votes=-1)

    def test_freshness_decay_hours_validation(self) -> None:
        """Test freshness_decay_hours must be positive."""
        with pytest.raises(ValueError, match="freshness_decay_hours"):
            ConsensusConfig(freshness_decay_hours=0)

        with pytest.raises(ValueError, match="freshness_decay_hours"):
            ConsensusConfig(freshness_decay_hours=-1)

    def test_weights_must_sum_to_one(self) -> None:
        """Test that calibration, reliability, and confidence weights sum to 1."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ConsensusConfig(
                calibration_weight=0.5,
                reliability_weight=0.5,
                confidence_weight=0.5,
            )

    def test_config_is_frozen(self) -> None:
        """Test ConsensusConfig is immutable."""
        config = ConsensusConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.hard_quorum_threshold = 0.9  # type: ignore[misc]


class TestWeightedVote:
    """Tests for WeightedVote dataclass."""

    @pytest.fixture
    def sample_vote(self) -> Vote:
        """Create a sample vote for testing."""
        return Vote(
            agent_id="agent-001",
            candidate_irdi="0173-1#01-ABA234#001",
            confidence=0.85,
            reliability_score=0.9,
        )

    def test_create_weighted_vote(self, sample_vote: Vote) -> None:
        """Test creating a WeightedVote."""
        weighted = WeightedVote(
            vote=sample_vote,
            calibration_factor=0.95,
            freshness_factor=0.8,
            combined_weight=0.7,
        )

        assert weighted.vote == sample_vote
        assert weighted.calibration_factor == 0.95
        assert weighted.freshness_factor == 0.8
        assert weighted.combined_weight == 0.7

    def test_calibration_factor_validation(self, sample_vote: Vote) -> None:
        """Test calibration_factor must be between 0 and 1."""
        with pytest.raises(ValueError, match="calibration_factor"):
            WeightedVote(
                vote=sample_vote,
                calibration_factor=1.5,
                freshness_factor=0.8,
                combined_weight=0.7,
            )

        with pytest.raises(ValueError, match="calibration_factor"):
            WeightedVote(
                vote=sample_vote,
                calibration_factor=-0.1,
                freshness_factor=0.8,
                combined_weight=0.7,
            )

    def test_freshness_factor_validation(self, sample_vote: Vote) -> None:
        """Test freshness_factor must be between 0 and 1."""
        with pytest.raises(ValueError, match="freshness_factor"):
            WeightedVote(
                vote=sample_vote,
                calibration_factor=0.9,
                freshness_factor=1.5,
                combined_weight=0.7,
            )

        with pytest.raises(ValueError, match="freshness_factor"):
            WeightedVote(
                vote=sample_vote,
                calibration_factor=0.9,
                freshness_factor=-0.1,
                combined_weight=0.7,
            )

    def test_weighted_vote_is_frozen(self, sample_vote: Vote) -> None:
        """Test WeightedVote is immutable."""
        weighted = WeightedVote(
            vote=sample_vote,
            calibration_factor=0.95,
            freshness_factor=0.8,
            combined_weight=0.7,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            weighted.combined_weight = 0.5  # type: ignore[misc]


class TestConsensusEngineFreshness:
    """Tests for freshness decay calculation."""

    @pytest.fixture
    def engine(self) -> ConsensusEngine:
        """Create a consensus engine with default config."""
        return ConsensusEngine()

    def test_fresh_vote_has_high_freshness(self, engine: ConsensusEngine) -> None:
        """Test that a fresh vote (just cast) has freshness close to 1."""
        now = datetime.now(UTC)
        vote_time = now

        freshness = engine.compute_freshness_factor(vote_time, now)

        assert freshness == pytest.approx(1.0, rel=0.01)

    def test_one_half_life_old_vote(self, engine: ConsensusEngine) -> None:
        """Test that a vote one half-life old has freshness of 0.5."""
        # Default half-life is 24 hours
        now = datetime.now(UTC)
        vote_time = now - timedelta(hours=24)

        freshness = engine.compute_freshness_factor(vote_time, now)

        assert freshness == pytest.approx(0.5, rel=0.01)

    def test_two_half_lives_old_vote(self, engine: ConsensusEngine) -> None:
        """Test that a vote two half-lives old has freshness of 0.25."""
        now = datetime.now(UTC)
        vote_time = now - timedelta(hours=48)

        freshness = engine.compute_freshness_factor(vote_time, now)

        assert freshness == pytest.approx(0.25, rel=0.01)

    def test_custom_half_life(self) -> None:
        """Test freshness with custom half-life."""
        config = ConsensusConfig(freshness_decay_hours=12.0)
        engine = ConsensusEngine(config)

        now = datetime.now(UTC)
        vote_time = now - timedelta(hours=12)

        freshness = engine.compute_freshness_factor(vote_time, now)

        assert freshness == pytest.approx(0.5, rel=0.01)

    def test_future_vote_has_freshness_one(self, engine: ConsensusEngine) -> None:
        """Test that future timestamps are clamped to freshness 1."""
        now = datetime.now(UTC)
        future_time = now + timedelta(hours=1)

        freshness = engine.compute_freshness_factor(future_time, now)

        assert freshness == pytest.approx(1.0, rel=0.01)


class TestConsensusEngineWeightedVote:
    """Tests for weighted vote computation."""

    @pytest.fixture
    def engine(self) -> ConsensusEngine:
        """Create a consensus engine with default config."""
        return ConsensusEngine()

    @pytest.fixture
    def fresh_vote(self) -> Vote:
        """Create a fresh vote."""
        return Vote(
            agent_id="agent-001",
            candidate_irdi="0173-1#01-ABA234#001",
            confidence=0.8,
            reliability_score=0.9,
            timestamp=datetime.now(UTC),
        )

    def test_compute_weighted_vote(self, engine: ConsensusEngine, fresh_vote: Vote) -> None:
        """Test computing weighted vote with default config."""
        calibration = 0.95
        reference_time = fresh_vote.timestamp

        weighted = engine.compute_weighted_vote(fresh_vote, calibration, reference_time)

        assert weighted.vote == fresh_vote
        assert weighted.calibration_factor == calibration
        assert weighted.freshness_factor == pytest.approx(1.0, rel=0.01)

        # Verify combined weight calculation:
        # (0.3 * 0.95) + (0.5 * 0.9) + (0.2 * 0.8) = 0.285 + 0.45 + 0.16 = 0.895
        expected_weight = (
            0.3 * calibration  # calibration_weight * calibration_factor
            + 0.5 * 0.9  # reliability_weight * reliability_score
            + 0.2 * 0.8  # confidence_weight * confidence
        )
        assert weighted.combined_weight == pytest.approx(expected_weight, rel=0.01)

    def test_weighted_vote_with_decay(self, engine: ConsensusEngine) -> None:
        """Test weighted vote with time decay applied."""
        old_vote = Vote(
            agent_id="agent-001",
            candidate_irdi="0173-1#01-ABA234#001",
            confidence=0.8,
            reliability_score=0.9,
            timestamp=datetime.now(UTC) - timedelta(hours=24),
        )

        calibration = 0.95
        reference_time = datetime.now(UTC)

        weighted = engine.compute_weighted_vote(old_vote, calibration, reference_time)

        assert weighted.freshness_factor == pytest.approx(0.5, rel=0.01)

        # Combined weight should be halved due to freshness decay
        base_weight = 0.3 * 0.95 + 0.5 * 0.9 + 0.2 * 0.8
        expected_weight = base_weight * 0.5
        assert weighted.combined_weight == pytest.approx(expected_weight, rel=0.01)


class TestConsensusEngineAggregation:
    """Tests for vote aggregation."""

    @pytest.fixture
    def engine(self) -> ConsensusEngine:
        """Create a consensus engine with default config."""
        return ConsensusEngine()

    @pytest.fixture
    def reference_time(self) -> datetime:
        """Create a reference time for testing."""
        return datetime.now(UTC)

    def test_aggregate_single_vote(self, engine: ConsensusEngine, reference_time: datetime) -> None:
        """Test aggregating a single vote."""
        vote = Vote(
            agent_id="agent-001",
            candidate_irdi="0173-1#01-ABA234#001",
            confidence=0.8,
            reliability_score=0.9,
            timestamp=reference_time,
        )

        calibration_factors = {"agent-001": 0.95}

        scores, total, weighted = engine.aggregate_votes(
            [vote], calibration_factors, reference_time
        )

        assert len(scores) == 1
        assert "0173-1#01-ABA234#001" in scores
        assert len(weighted) == 1
        assert total == scores["0173-1#01-ABA234#001"]

    def test_aggregate_multiple_votes_same_irdi(
        self, engine: ConsensusEngine, reference_time: datetime
    ) -> None:
        """Test aggregating multiple votes for the same IRDI."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.8,
                reliability_score=0.9,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.85,
                reliability_score=0.88,
                timestamp=reference_time,
            ),
        ]

        calibration_factors = {"agent-001": 0.95, "agent-002": 0.90}

        scores, total, weighted = engine.aggregate_votes(votes, calibration_factors, reference_time)

        assert len(scores) == 1
        assert "0173-1#01-ABA234#001" in scores
        assert len(weighted) == 2
        assert scores["0173-1#01-ABA234#001"] == total

    def test_aggregate_multiple_votes_different_irdis(
        self, engine: ConsensusEngine, reference_time: datetime
    ) -> None:
        """Test aggregating votes for different IRDIs."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.8,
                reliability_score=0.9,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-XYZ789#001",
                confidence=0.85,
                reliability_score=0.88,
                timestamp=reference_time,
            ),
        ]

        calibration_factors = {"agent-001": 0.95, "agent-002": 0.90}

        scores, total, weighted = engine.aggregate_votes(votes, calibration_factors, reference_time)

        assert len(scores) == 2
        assert "0173-1#01-ABA234#001" in scores
        assert "0173-1#01-XYZ789#001" in scores
        assert len(weighted) == 2
        assert sum(scores.values()) == pytest.approx(total, rel=0.01)

    def test_aggregate_with_default_calibration(
        self, engine: ConsensusEngine, reference_time: datetime
    ) -> None:
        """Test that unknown agents get default calibration of 0.5."""
        vote = Vote(
            agent_id="unknown-agent",
            candidate_irdi="0173-1#01-ABA234#001",
            confidence=0.8,
            reliability_score=0.9,
            timestamp=reference_time,
        )

        calibration_factors = {}  # No calibration data

        scores, total, weighted = engine.aggregate_votes(
            [vote], calibration_factors, reference_time
        )

        assert weighted[0].calibration_factor == 0.5

    def test_aggregate_empty_votes(self, engine: ConsensusEngine, reference_time: datetime) -> None:
        """Test aggregating empty vote list."""
        scores, total, weighted = engine.aggregate_votes([], {}, reference_time)

        assert len(scores) == 0
        assert total == 0.0
        assert len(weighted) == 0


class TestConsensusEngineQuorum:
    """Tests for quorum determination."""

    @pytest.fixture
    def engine(self) -> ConsensusEngine:
        """Create a consensus engine with default config."""
        return ConsensusEngine()

    @pytest.fixture
    def reference_time(self) -> datetime:
        """Create a reference time for testing."""
        return datetime.now(UTC)

    def test_hard_quorum(self, engine: ConsensusEngine, reference_time: datetime) -> None:
        """Test hard quorum is achieved when proportion >= 0.8."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-003",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
        ]

        # All votes for same IRDI = 100% proportion
        scores = {"0173-1#01-ABA234#001": 1.0}
        total = 1.0

        irdi, quorum_type, confidence = engine.determine_quorum(scores, total, votes)

        assert irdi == "0173-1#01-ABA234#001"
        assert quorum_type == "hard"
        assert confidence == 1.0

    def test_soft_quorum(self, engine: ConsensusEngine, reference_time: datetime) -> None:
        """Test soft quorum is achieved when 0.5 <= proportion < 0.8."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-XYZ789#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
        ]

        # 60% for IRDI A, 40% for IRDI B
        scores = {"0173-1#01-ABA234#001": 0.6, "0173-1#01-XYZ789#001": 0.4}
        total = 1.0

        irdi, quorum_type, confidence = engine.determine_quorum(scores, total, votes)

        assert irdi == "0173-1#01-ABA234#001"
        assert quorum_type == "soft"
        assert confidence == pytest.approx(0.6, rel=0.01)

    def test_conflict_quorum(self, engine: ConsensusEngine, reference_time: datetime) -> None:
        """Test conflict quorum when no IRDI reaches 0.5."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-XYZ789#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-003",
                candidate_irdi="0173-1#01-QRS456#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
        ]

        # Three-way split: 40%, 35%, 25%
        scores = {
            "0173-1#01-ABA234#001": 0.40,
            "0173-1#01-XYZ789#001": 0.35,
            "0173-1#01-QRS456#001": 0.25,
        }
        total = 1.0

        irdi, quorum_type, confidence = engine.determine_quorum(scores, total, votes)

        assert irdi == "0173-1#01-ABA234#001"  # Still returns top candidate
        assert quorum_type == "conflict"
        assert confidence == pytest.approx(0.4, rel=0.01)

    def test_tie_detection(self, engine: ConsensusEngine, reference_time: datetime) -> None:
        """Test that ties are detected and marked as conflict."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-XYZ789#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
        ]

        # Exact tie: 50% each
        scores = {"0173-1#01-ABA234#001": 0.5, "0173-1#01-XYZ789#001": 0.5}
        total = 1.0

        irdi, quorum_type, confidence = engine.determine_quorum(scores, total, votes)

        assert quorum_type == "conflict"

    def test_insufficient_votes(self, engine: ConsensusEngine, reference_time: datetime) -> None:
        """Test conflict when below minimum votes."""
        # Default min_votes is 2, provide only 1
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
        ]

        scores = {"0173-1#01-ABA234#001": 1.0}
        total = 1.0

        irdi, quorum_type, confidence = engine.determine_quorum(scores, total, votes)

        assert quorum_type == "conflict"  # Marked as conflict due to insufficient votes

    def test_empty_scores(self, engine: ConsensusEngine, reference_time: datetime) -> None:
        """Test handling empty aggregated scores."""
        irdi, quorum_type, confidence = engine.determine_quorum({}, 0.0, [])

        assert irdi is None
        assert quorum_type == "conflict"
        assert confidence == 0.0


class TestConsensusEngineReachConsensus:
    """Tests for the full consensus process."""

    @pytest.fixture
    def engine(self) -> ConsensusEngine:
        """Create a consensus engine with default config."""
        return ConsensusEngine()

    @pytest.fixture
    def reference_time(self) -> datetime:
        """Create a reference time for testing."""
        return datetime.now(UTC)

    def test_reach_consensus_unanimous(
        self, engine: ConsensusEngine, reference_time: datetime
    ) -> None:
        """Test reaching unanimous consensus."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.85,
                reliability_score=0.90,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-003",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.88,
                reliability_score=0.92,
                timestamp=reference_time,
            ),
        ]

        calibration_factors = {
            "agent-001": 0.95,
            "agent-002": 0.90,
            "agent-003": 0.88,
        }

        record = engine.reach_consensus("tag-123", votes, calibration_factors, reference_time)

        assert record.tag_id == "tag-123"
        assert record.agreed_irdi == "0173-1#01-ABA234#001"
        assert record.quorum_type == "hard"
        assert record.consensus_confidence == 1.0
        assert len(record.votes) == 3
        assert record.is_unanimous

    def test_reach_consensus_soft_quorum(
        self, engine: ConsensusEngine, reference_time: datetime
    ) -> None:
        """Test reaching soft quorum consensus."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.85,
                reliability_score=0.90,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-003",
                candidate_irdi="0173-1#01-XYZ789#001",
                confidence=0.88,
                reliability_score=0.92,
                timestamp=reference_time,
            ),
        ]

        calibration_factors = {
            "agent-001": 0.95,
            "agent-002": 0.90,
            "agent-003": 0.88,
        }

        record = engine.reach_consensus("tag-123", votes, calibration_factors, reference_time)

        # 2 votes for ABA234, 1 for XYZ789 - should be soft quorum
        assert record.agreed_irdi == "0173-1#01-ABA234#001"
        assert record.quorum_type == "soft"
        assert 0.5 < record.consensus_confidence < 0.8
        assert not record.is_unanimous

    def test_reach_consensus_conflict(
        self, engine: ConsensusEngine, reference_time: datetime
    ) -> None:
        """Test reaching conflict consensus."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-XYZ789#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
        ]

        calibration_factors = {
            "agent-001": 0.95,
            "agent-002": 0.95,
        }

        record = engine.reach_consensus("tag-123", votes, calibration_factors, reference_time)

        # Equal votes = tie = conflict
        assert record.quorum_type == "conflict"
        assert not record.is_unanimous

    def test_reach_consensus_no_votes_raises(self, engine: ConsensusEngine) -> None:
        """Test that empty votes raises ValueError."""
        with pytest.raises(ValueError, match="no votes"):
            engine.reach_consensus("tag-123", [], {})

    def test_reach_consensus_audit_trail(
        self, engine: ConsensusEngine, reference_time: datetime
    ) -> None:
        """Test that consensus record has audit trail."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.85,
                reliability_score=0.90,
                timestamp=reference_time,
            ),
        ]

        calibration_factors = {"agent-001": 0.95, "agent-002": 0.90}

        record = engine.reach_consensus("tag-123", votes, calibration_factors, reference_time)

        assert len(record.audit_trail) > 0
        assert "Consensus process started" in record.audit_trail[0]
        assert any("Quorum type" in entry for entry in record.audit_trail)

    def test_reach_consensus_with_old_votes(self, engine: ConsensusEngine) -> None:
        """Test consensus with votes of different ages."""
        now = datetime.now(UTC)

        votes = [
            # Fresh vote
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=now,
            ),
            # Old vote (24 hours old = half freshness)
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-XYZ789#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=now - timedelta(hours=24),
            ),
        ]

        calibration_factors = {"agent-001": 0.95, "agent-002": 0.95}

        record = engine.reach_consensus("tag-123", votes, calibration_factors, now)

        # Fresh vote should have more weight
        # The fresh vote for ABA234 should win due to higher freshness
        assert record.agreed_irdi == "0173-1#01-ABA234#001"


class TestConsensusEngineAuditTrail:
    """Tests for audit trail functionality."""

    @pytest.fixture
    def engine(self) -> ConsensusEngine:
        """Create a consensus engine with default config."""
        return ConsensusEngine()

    @pytest.fixture
    def sample_record(self) -> ConsensusRecord:
        """Create a sample consensus record."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.85,
                reliability_score=0.90,
            ),
        ]

        return ConsensusRecord(
            tag_id="tag-123",
            agreed_irdi="0173-1#01-ABA234#001",
            consensus_confidence=0.88,
            votes=votes,
            quorum_type="hard",
        )

    def test_add_audit_entry(self, engine: ConsensusEngine, sample_record: ConsensusRecord) -> None:
        """Test adding an audit entry."""
        updated = engine.add_audit_entry(sample_record, "Manual review completed")

        assert len(updated.audit_trail) == 1
        assert "Manual review completed" in updated.audit_trail[0]
        # Original unchanged
        assert len(sample_record.audit_trail) == 0

    def test_multiple_audit_entries(
        self, engine: ConsensusEngine, sample_record: ConsensusRecord
    ) -> None:
        """Test adding multiple audit entries."""
        updated = engine.add_audit_entry(sample_record, "Entry 1")
        updated = engine.add_audit_entry(updated, "Entry 2")
        updated = engine.add_audit_entry(updated, "Entry 3")

        assert len(updated.audit_trail) == 3


class TestConsensusEngineEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def engine(self) -> ConsensusEngine:
        """Create a consensus engine with default config."""
        return ConsensusEngine()

    @pytest.fixture
    def reference_time(self) -> datetime:
        """Create a reference time for testing."""
        return datetime.now(UTC)

    def test_single_vote_below_min(self, engine: ConsensusEngine, reference_time: datetime) -> None:
        """Test consensus with single vote (below min_votes=2)."""
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            ),
        ]

        calibration_factors = {"agent-001": 0.95}

        record = engine.reach_consensus("tag-123", votes, calibration_factors, reference_time)

        # Should return conflict due to insufficient votes
        assert record.quorum_type == "conflict"
        assert record.agreed_irdi == "0173-1#01-ABA234#001"

    def test_all_same_irdi(self, engine: ConsensusEngine, reference_time: datetime) -> None:
        """Test consensus when all votes are for the same IRDI."""
        votes = [
            Vote(
                agent_id=f"agent-{i:03d}",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.8 + i * 0.01,
                reliability_score=0.9,
                timestamp=reference_time,
            )
            for i in range(5)
        ]

        calibration_factors = {f"agent-{i:03d}": 0.9 for i in range(5)}

        record = engine.reach_consensus("tag-123", votes, calibration_factors, reference_time)

        assert record.quorum_type == "hard"
        assert record.consensus_confidence == 1.0
        assert record.is_unanimous

    def test_min_votes_config(self, reference_time: datetime) -> None:
        """Test custom min_votes configuration."""
        config = ConsensusConfig(min_votes=5)
        engine = ConsensusEngine(config)

        votes = [
            Vote(
                agent_id=f"agent-{i:03d}",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=reference_time,
            )
            for i in range(3)
        ]

        calibration_factors = {f"agent-{i:03d}": 0.95 for i in range(3)}

        record = engine.reach_consensus("tag-123", votes, calibration_factors, reference_time)

        # 3 votes but 5 required -> conflict
        assert record.quorum_type == "conflict"

    def test_very_old_votes(self, engine: ConsensusEngine) -> None:
        """Test consensus with very old votes (low freshness)."""
        now = datetime.now(UTC)
        old_time = now - timedelta(days=30)  # 30 days old

        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.9,
                reliability_score=0.95,
                timestamp=old_time,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.85,
                reliability_score=0.90,
                timestamp=old_time,
            ),
        ]

        calibration_factors = {"agent-001": 0.95, "agent-002": 0.90}

        record = engine.reach_consensus("tag-123", votes, calibration_factors, now)

        # Should still reach consensus, just with lower effective weights
        assert record.agreed_irdi == "0173-1#01-ABA234#001"
        # After 30 days (~1.25 half-lives), freshness is very low
        # But all votes are equally old, so still unanimous
        assert record.quorum_type == "hard"


class TestConsensusExceptions:
    """Tests for custom consensus exceptions."""

    def test_insufficient_votes_error(self) -> None:
        """Test InsufficientVotesError exception."""
        error = InsufficientVotesError(vote_count=1, min_required=3)

        assert error.vote_count == 1
        assert error.min_required == 3
        assert "1 provided" in str(error)
        assert "3 required" in str(error)
        assert isinstance(error, ConsensusError)

    def test_no_consensus_error(self) -> None:
        """Test NoConsensusError exception."""
        error = NoConsensusError(
            tag_id="tag-123",
            top_candidates=[("0173-1#01-ABA234#001", 0.45), ("0173-1#01-XYZ789#001", 0.40)],
        )

        assert error.tag_id == "tag-123"
        assert len(error.top_candidates) == 2
        assert "tag-123" in str(error)
        assert isinstance(error, ConsensusError)


class TestConsensusEngineIntegration:
    """Integration tests for complete consensus workflows."""

    def test_full_workflow(self) -> None:
        """Test complete consensus workflow from config to record."""
        # 1. Create config
        config = ConsensusConfig(
            hard_quorum_threshold=0.75,
            soft_quorum_threshold=0.5,
            min_votes=2,
        )

        # 2. Create engine
        engine = ConsensusEngine(config)

        # 3. Create votes
        now = datetime.now(UTC)
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.92,
                reliability_score=0.95,
                timestamp=now,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.88,
                reliability_score=0.90,
                timestamp=now,
            ),
            Vote(
                agent_id="agent-003",
                candidate_irdi="0173-1#01-XYZ789#001",
                confidence=0.85,
                reliability_score=0.88,
                timestamp=now,
            ),
        ]

        # 4. Set calibration factors
        calibration_factors = {
            "agent-001": 0.95,  # Well-calibrated
            "agent-002": 0.90,
            "agent-003": 0.85,
        }

        # 5. Reach consensus
        record = engine.reach_consensus("tag-reactor-temp", votes, calibration_factors, now)

        # 6. Verify record
        assert record.tag_id == "tag-reactor-temp"
        assert record.agreed_irdi == "0173-1#01-ABA234#001"
        assert record.vote_count == 3
        assert len(record.audit_trail) > 0

        # 7. Add audit entry
        updated = engine.add_audit_entry(record, "Reviewed by system admin")
        assert len(updated.audit_trail) == len(record.audit_trail) + 1

        # 8. Mark as validated
        validated = updated.mark_validated("Confirmed by domain expert")
        assert validated.human_validated is True
        assert "domain expert" in validated.validation_notes or ""

    def test_swarm_import(self) -> None:
        """Test that consensus classes can be imported from swarm module."""
        from noa_swarm.swarm import (
            ConsensusConfig,
            ConsensusEngine,
            ConsensusError,
            InsufficientVotesError,
            NoConsensusError,
            WeightedVote,
        )

        # Verify imports work
        assert ConsensusConfig is not None
        assert ConsensusEngine is not None
        assert WeightedVote is not None
        assert ConsensusError is not None
        assert InsufficientVotesError is not None
        assert NoConsensusError is not None
