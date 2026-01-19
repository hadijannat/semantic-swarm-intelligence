"""Unit tests for the reputation tracking system."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from noa_swarm.swarm.reputation import (
    AgentOutcome,
    AgentReputation,
    ReputationConfig,
    ReputationTracker,
    _clamp,
    _compute_score,
)


class TestReputationConfig:
    """Tests for ReputationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test ReputationConfig has expected default values."""
        config = ReputationConfig()

        assert config.window_size == 100
        assert config.initial_score == 0.5
        assert config.min_score == 0.1
        assert config.max_score == 0.95
        assert config.decay_factor == 0.95
        assert config.agreement_bonus == 0.02
        assert config.disagreement_penalty == 0.03

    def test_custom_values(self) -> None:
        """Test ReputationConfig accepts custom values."""
        config = ReputationConfig(
            window_size=50,
            initial_score=0.6,
            min_score=0.2,
            max_score=0.9,
            decay_factor=0.9,
            agreement_bonus=0.05,
            disagreement_penalty=0.04,
        )

        assert config.window_size == 50
        assert config.initial_score == 0.6
        assert config.min_score == 0.2
        assert config.max_score == 0.9
        assert config.decay_factor == 0.9
        assert config.agreement_bonus == 0.05
        assert config.disagreement_penalty == 0.04

    def test_window_size_validation(self) -> None:
        """Test window_size must be at least 1."""
        with pytest.raises(ValueError, match="window_size"):
            ReputationConfig(window_size=0)

        with pytest.raises(ValueError, match="window_size"):
            ReputationConfig(window_size=-1)

    def test_initial_score_validation(self) -> None:
        """Test initial_score must be between 0 and 1."""
        with pytest.raises(ValueError, match="initial_score"):
            ReputationConfig(initial_score=1.5)

        with pytest.raises(ValueError, match="initial_score"):
            ReputationConfig(initial_score=-0.1)

    def test_min_score_validation(self) -> None:
        """Test min_score must be between 0 and 1."""
        with pytest.raises(ValueError, match="min_score"):
            ReputationConfig(min_score=1.5)

        with pytest.raises(ValueError, match="min_score"):
            ReputationConfig(min_score=-0.1)

    def test_max_score_validation(self) -> None:
        """Test max_score must be between 0 and 1."""
        with pytest.raises(ValueError, match="max_score"):
            ReputationConfig(max_score=1.5)

        with pytest.raises(ValueError, match="max_score"):
            ReputationConfig(max_score=-0.1)

    def test_min_cannot_exceed_max(self) -> None:
        """Test min_score cannot be greater than max_score."""
        with pytest.raises(ValueError, match="min_score.*cannot be greater"):
            ReputationConfig(min_score=0.9, max_score=0.5)

    def test_decay_factor_validation(self) -> None:
        """Test decay_factor must be between 0 (exclusive) and 1."""
        with pytest.raises(ValueError, match="decay_factor"):
            ReputationConfig(decay_factor=0.0)

        with pytest.raises(ValueError, match="decay_factor"):
            ReputationConfig(decay_factor=-0.1)

        with pytest.raises(ValueError, match="decay_factor"):
            ReputationConfig(decay_factor=1.5)

        # 1.0 should be valid
        config = ReputationConfig(decay_factor=1.0)
        assert config.decay_factor == 1.0

    def test_agreement_bonus_validation(self) -> None:
        """Test agreement_bonus must be non-negative."""
        with pytest.raises(ValueError, match="agreement_bonus"):
            ReputationConfig(agreement_bonus=-0.01)

        # Zero should be valid
        config = ReputationConfig(agreement_bonus=0.0)
        assert config.agreement_bonus == 0.0

    def test_disagreement_penalty_validation(self) -> None:
        """Test disagreement_penalty must be non-negative."""
        with pytest.raises(ValueError, match="disagreement_penalty"):
            ReputationConfig(disagreement_penalty=-0.01)

        # Zero should be valid
        config = ReputationConfig(disagreement_penalty=0.0)
        assert config.disagreement_penalty == 0.0

    def test_config_is_frozen(self) -> None:
        """Test ReputationConfig is immutable."""
        config = ReputationConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.window_size = 50  # type: ignore[misc]


class TestAgentOutcome:
    """Tests for AgentOutcome dataclass."""

    def test_create_outcome(self) -> None:
        """Test creating an AgentOutcome."""
        outcome = AgentOutcome(
            agent_id="agent-001",
            tag_id="tag-123",
            predicted_irdi="0173-1#01-ABA234#001",
            final_irdi="0173-1#01-ABA234#001",
        )

        assert outcome.agent_id == "agent-001"
        assert outcome.tag_id == "tag-123"
        assert outcome.predicted_irdi == "0173-1#01-ABA234#001"
        assert outcome.final_irdi == "0173-1#01-ABA234#001"
        assert outcome.timestamp is not None

    def test_was_correct_property_true(self) -> None:
        """Test was_correct returns True when prediction matches."""
        outcome = AgentOutcome(
            agent_id="agent-001",
            tag_id="tag-123",
            predicted_irdi="0173-1#01-ABA234#001",
            final_irdi="0173-1#01-ABA234#001",
        )

        assert outcome.was_correct is True

    def test_was_correct_property_false(self) -> None:
        """Test was_correct returns False when prediction differs."""
        outcome = AgentOutcome(
            agent_id="agent-001",
            tag_id="tag-123",
            predicted_irdi="0173-1#01-ABA234#001",
            final_irdi="0173-1#01-XYZ789#001",
        )

        assert outcome.was_correct is False

    def test_custom_timestamp(self) -> None:
        """Test creating outcome with custom timestamp."""
        custom_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        outcome = AgentOutcome(
            agent_id="agent-001",
            tag_id="tag-123",
            predicted_irdi="irdi-a",
            final_irdi="irdi-a",
            timestamp=custom_time,
        )

        assert outcome.timestamp == custom_time

    def test_empty_agent_id_validation(self) -> None:
        """Test validation rejects empty agent_id."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            AgentOutcome(
                agent_id="",
                tag_id="tag-123",
                predicted_irdi="irdi-a",
                final_irdi="irdi-a",
            )

    def test_empty_tag_id_validation(self) -> None:
        """Test validation rejects empty tag_id."""
        with pytest.raises(ValueError, match="tag_id cannot be empty"):
            AgentOutcome(
                agent_id="agent-001",
                tag_id="",
                predicted_irdi="irdi-a",
                final_irdi="irdi-a",
            )

    def test_empty_predicted_irdi_validation(self) -> None:
        """Test validation rejects empty predicted_irdi."""
        with pytest.raises(ValueError, match="predicted_irdi cannot be empty"):
            AgentOutcome(
                agent_id="agent-001",
                tag_id="tag-123",
                predicted_irdi="",
                final_irdi="irdi-a",
            )

    def test_empty_final_irdi_validation(self) -> None:
        """Test validation rejects empty final_irdi."""
        with pytest.raises(ValueError, match="final_irdi cannot be empty"):
            AgentOutcome(
                agent_id="agent-001",
                tag_id="tag-123",
                predicted_irdi="irdi-a",
                final_irdi="",
            )


class TestAgentReputation:
    """Tests for AgentReputation dataclass."""

    def test_create_reputation(self) -> None:
        """Test creating an AgentReputation."""
        reputation = AgentReputation(
            agent_id="agent-001",
            reliability_score=0.75,
        )

        assert reputation.agent_id == "agent-001"
        assert reputation.reliability_score == 0.75
        assert reputation.outcomes == []
        assert reputation.total_predictions == 0
        assert reputation.correct_predictions == 0
        assert reputation.created_at is not None
        assert reputation.updated_at is not None

    def test_create_reputation_with_outcomes(self) -> None:
        """Test creating reputation with existing outcomes."""
        outcomes = [
            AgentOutcome(
                agent_id="agent-001",
                tag_id="tag-1",
                predicted_irdi="irdi-a",
                final_irdi="irdi-a",
            ),
            AgentOutcome(
                agent_id="agent-001",
                tag_id="tag-2",
                predicted_irdi="irdi-b",
                final_irdi="irdi-c",
            ),
        ]

        reputation = AgentReputation(
            agent_id="agent-001",
            reliability_score=0.5,
            outcomes=outcomes,
            total_predictions=2,
            correct_predictions=1,
        )

        assert len(reputation.outcomes) == 2
        assert reputation.total_predictions == 2
        assert reputation.correct_predictions == 1

    def test_accuracy_property(self) -> None:
        """Test accuracy property calculates correctly."""
        reputation = AgentReputation(
            agent_id="agent-001",
            reliability_score=0.5,
            total_predictions=10,
            correct_predictions=7,
        )

        assert reputation.accuracy == pytest.approx(0.7)

    def test_accuracy_property_no_predictions(self) -> None:
        """Test accuracy returns 0.0 when no predictions."""
        reputation = AgentReputation(
            agent_id="agent-001",
            reliability_score=0.5,
        )

        assert reputation.accuracy == 0.0

    def test_empty_agent_id_validation(self) -> None:
        """Test validation rejects empty agent_id."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            AgentReputation(
                agent_id="",
                reliability_score=0.5,
            )

    def test_reliability_score_validation(self) -> None:
        """Test reliability_score must be between 0 and 1."""
        with pytest.raises(ValueError, match="reliability_score"):
            AgentReputation(
                agent_id="agent-001",
                reliability_score=1.5,
            )

        with pytest.raises(ValueError, match="reliability_score"):
            AgentReputation(
                agent_id="agent-001",
                reliability_score=-0.1,
            )


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_clamp_within_bounds(self) -> None:
        """Test clamp returns value when within bounds."""
        assert _clamp(0.5, 0.0, 1.0) == 0.5
        assert _clamp(0.3, 0.1, 0.9) == 0.3

    def test_clamp_below_minimum(self) -> None:
        """Test clamp returns minimum when value is below."""
        assert _clamp(0.05, 0.1, 0.9) == 0.1
        assert _clamp(-0.5, 0.0, 1.0) == 0.0

    def test_clamp_above_maximum(self) -> None:
        """Test clamp returns maximum when value is above."""
        assert _clamp(0.95, 0.1, 0.9) == 0.9
        assert _clamp(1.5, 0.0, 1.0) == 1.0

    def test_compute_score_empty_outcomes(self) -> None:
        """Test compute_score returns initial_score when no outcomes."""
        score = _compute_score([], 0.95, 0.5, 0.1, 0.95)
        assert score == 0.5

    def test_compute_score_all_correct(self) -> None:
        """Test compute_score with all correct predictions."""
        outcomes = [
            AgentOutcome(
                agent_id="agent-001",
                tag_id=f"tag-{i}",
                predicted_irdi="irdi-a",
                final_irdi="irdi-a",
            )
            for i in range(5)
        ]

        score = _compute_score(outcomes, 0.95, 0.5, 0.1, 0.95)
        assert score == 0.95  # Clamped to max

    def test_compute_score_all_wrong(self) -> None:
        """Test compute_score with all wrong predictions."""
        outcomes = [
            AgentOutcome(
                agent_id="agent-001",
                tag_id=f"tag-{i}",
                predicted_irdi="irdi-a",
                final_irdi="irdi-b",
            )
            for i in range(5)
        ]

        score = _compute_score(outcomes, 0.95, 0.5, 0.1, 0.95)
        assert score == 0.1  # Clamped to min

    def test_compute_score_mixed_outcomes(self) -> None:
        """Test compute_score with mixed correct/wrong predictions."""
        outcomes = [
            AgentOutcome(
                agent_id="agent-001",
                tag_id="tag-1",
                predicted_irdi="irdi-a",
                final_irdi="irdi-a",  # correct
            ),
            AgentOutcome(
                agent_id="agent-001",
                tag_id="tag-2",
                predicted_irdi="irdi-a",
                final_irdi="irdi-b",  # wrong
            ),
        ]

        score = _compute_score(outcomes, 0.95, 0.5, 0.1, 0.95)
        # Most recent (wrong) has more weight than older (correct)
        # With decay=0.95: wrong weight=1.0, correct weight=0.95
        # weighted_sum = 0.95 * 1.0 + 1.0 * 0.0 = 0.95
        # weight_total = 0.95 + 1.0 = 1.95
        # raw_score = 0.95 / 1.95 = ~0.487
        assert 0.4 < score < 0.6

    def test_compute_score_decay_effect(self) -> None:
        """Test that decay gives more weight to recent outcomes."""
        # Older correct, newer wrong
        outcomes_old_correct = [
            AgentOutcome(
                agent_id="agent-001",
                tag_id="tag-1",
                predicted_irdi="irdi-a",
                final_irdi="irdi-a",  # correct (older)
            ),
            AgentOutcome(
                agent_id="agent-001",
                tag_id="tag-2",
                predicted_irdi="irdi-a",
                final_irdi="irdi-b",  # wrong (newer)
            ),
        ]

        # Older wrong, newer correct
        outcomes_old_wrong = [
            AgentOutcome(
                agent_id="agent-001",
                tag_id="tag-1",
                predicted_irdi="irdi-a",
                final_irdi="irdi-b",  # wrong (older)
            ),
            AgentOutcome(
                agent_id="agent-001",
                tag_id="tag-2",
                predicted_irdi="irdi-a",
                final_irdi="irdi-a",  # correct (newer)
            ),
        ]

        score_old_correct = _compute_score(outcomes_old_correct, 0.95, 0.5, 0.1, 0.95)
        score_old_wrong = _compute_score(outcomes_old_wrong, 0.95, 0.5, 0.1, 0.95)

        # Newer outcomes should dominate, so old_wrong (new correct) > old_correct (new wrong)
        assert score_old_wrong > score_old_correct


class TestReputationTrackerInit:
    """Tests for ReputationTracker initialization."""

    def test_default_config(self) -> None:
        """Test ReputationTracker with default config."""
        tracker = ReputationTracker()

        assert tracker.config.window_size == 100
        assert tracker.config.initial_score == 0.5

    def test_custom_config(self) -> None:
        """Test ReputationTracker with custom config."""
        config = ReputationConfig(window_size=50, initial_score=0.6)
        tracker = ReputationTracker(config)

        assert tracker.config.window_size == 50
        assert tracker.config.initial_score == 0.6


class TestReputationTrackerGetReliability:
    """Tests for ReputationTracker.get_reliability."""

    @pytest.fixture
    def tracker(self) -> ReputationTracker:
        """Create a ReputationTracker with default config."""
        return ReputationTracker()

    def test_get_reliability_unknown_agent(self, tracker: ReputationTracker) -> None:
        """Test get_reliability returns initial_score for unknown agent."""
        score = tracker.get_reliability("unknown-agent")
        assert score == 0.5

    def test_get_reliability_after_record(self, tracker: ReputationTracker) -> None:
        """Test get_reliability returns correct score after recording."""
        tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")
        score = tracker.get_reliability("agent-001")

        assert score > 0.5  # Should be higher after correct prediction


class TestReputationTrackerGetReputation:
    """Tests for ReputationTracker.get_reputation."""

    @pytest.fixture
    def tracker(self) -> ReputationTracker:
        """Create a ReputationTracker with default config."""
        return ReputationTracker()

    def test_get_reputation_unknown_agent(self, tracker: ReputationTracker) -> None:
        """Test get_reputation returns None for unknown agent."""
        reputation = tracker.get_reputation("unknown-agent")
        assert reputation is None

    def test_get_reputation_after_record(self, tracker: ReputationTracker) -> None:
        """Test get_reputation returns record after recording."""
        tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")
        reputation = tracker.get_reputation("agent-001")

        assert reputation is not None
        assert reputation.agent_id == "agent-001"
        assert reputation.total_predictions == 1
        assert reputation.correct_predictions == 1


class TestReputationTrackerRecordOutcome:
    """Tests for ReputationTracker.record_outcome."""

    @pytest.fixture
    def tracker(self) -> ReputationTracker:
        """Create a ReputationTracker with default config."""
        return ReputationTracker()

    def test_record_first_correct_outcome(self, tracker: ReputationTracker) -> None:
        """Test recording first correct outcome for new agent."""
        reputation = tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")

        assert reputation.agent_id == "agent-001"
        assert reputation.total_predictions == 1
        assert reputation.correct_predictions == 1
        assert len(reputation.outcomes) == 1
        assert reputation.outcomes[0].was_correct is True

    def test_record_first_wrong_outcome(self, tracker: ReputationTracker) -> None:
        """Test recording first wrong outcome for new agent."""
        reputation = tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-b")

        assert reputation.total_predictions == 1
        assert reputation.correct_predictions == 0
        assert reputation.outcomes[0].was_correct is False

    def test_record_multiple_outcomes(self, tracker: ReputationTracker) -> None:
        """Test recording multiple outcomes."""
        tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")  # correct
        tracker.record_outcome("agent-001", "tag-2", "irdi-b", "irdi-c")  # wrong
        reputation = tracker.record_outcome(
            "agent-001", "tag-3", "irdi-d", "irdi-d"
        )  # correct

        assert reputation.total_predictions == 3
        assert reputation.correct_predictions == 2
        assert len(reputation.outcomes) == 3

    def test_record_outcome_window_size(self) -> None:
        """Test that outcomes are trimmed to window_size."""
        config = ReputationConfig(window_size=3)
        tracker = ReputationTracker(config)

        # Record more outcomes than window_size
        for i in range(5):
            tracker.record_outcome("agent-001", f"tag-{i}", "irdi-a", "irdi-a")

        reputation = tracker.get_reputation("agent-001")
        assert reputation is not None
        assert len(reputation.outcomes) == 3  # Trimmed to window_size
        assert reputation.total_predictions == 5  # Total still tracked

    def test_record_outcome_updates_score(self, tracker: ReputationTracker) -> None:
        """Test that recording outcomes updates the reliability score."""
        reputation1 = tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")
        score1 = reputation1.reliability_score

        reputation2 = tracker.record_outcome("agent-001", "tag-2", "irdi-b", "irdi-b")
        score2 = reputation2.reliability_score

        # Score should remain high (all correct)
        assert score1 > 0.5
        assert score2 > 0.5

    def test_record_outcome_different_agents(self, tracker: ReputationTracker) -> None:
        """Test recording outcomes for different agents."""
        tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")
        tracker.record_outcome("agent-002", "tag-1", "irdi-a", "irdi-b")

        rep1 = tracker.get_reputation("agent-001")
        rep2 = tracker.get_reputation("agent-002")

        assert rep1 is not None
        assert rep2 is not None
        assert rep1.reliability_score > rep2.reliability_score  # agent-001 was correct


class TestReputationTrackerUpdateScores:
    """Tests for ReputationTracker.update_scores."""

    def test_update_scores_recalculates(self) -> None:
        """Test that update_scores recalculates all scores."""
        tracker = ReputationTracker()

        # Record some outcomes
        tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")
        tracker.record_outcome("agent-002", "tag-1", "irdi-a", "irdi-b")

        # Get initial scores
        score1_before = tracker.get_reliability("agent-001")
        score2_before = tracker.get_reliability("agent-002")

        # Update scores (should not change much, just recalculate)
        tracker.update_scores()

        score1_after = tracker.get_reliability("agent-001")
        score2_after = tracker.get_reliability("agent-002")

        # Scores should be approximately the same (small float differences ok)
        assert score1_after == pytest.approx(score1_before, rel=0.01)
        assert score2_after == pytest.approx(score2_before, rel=0.01)


class TestReputationTrackerGetAllReputations:
    """Tests for ReputationTracker.get_all_reputations."""

    @pytest.fixture
    def tracker(self) -> ReputationTracker:
        """Create a ReputationTracker with default config."""
        return ReputationTracker()

    def test_get_all_reputations_empty(self, tracker: ReputationTracker) -> None:
        """Test get_all_reputations when empty."""
        reputations = tracker.get_all_reputations()
        assert reputations == {}

    def test_get_all_reputations_multiple_agents(
        self, tracker: ReputationTracker
    ) -> None:
        """Test get_all_reputations with multiple agents."""
        tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")
        tracker.record_outcome("agent-002", "tag-1", "irdi-b", "irdi-b")
        tracker.record_outcome("agent-003", "tag-1", "irdi-c", "irdi-d")

        reputations = tracker.get_all_reputations()

        assert len(reputations) == 3
        assert "agent-001" in reputations
        assert "agent-002" in reputations
        assert "agent-003" in reputations


class TestReputationTrackerBootstrapAgent:
    """Tests for ReputationTracker.bootstrap_agent."""

    @pytest.fixture
    def tracker(self) -> ReputationTracker:
        """Create a ReputationTracker with default config."""
        return ReputationTracker()

    def test_bootstrap_new_agent(self, tracker: ReputationTracker) -> None:
        """Test bootstrapping a new agent."""
        reputation = tracker.bootstrap_agent("agent-001")

        assert reputation.agent_id == "agent-001"
        assert reputation.reliability_score == 0.5  # default initial_score
        assert reputation.total_predictions == 0
        assert reputation.outcomes == []

    def test_bootstrap_with_custom_score(self, tracker: ReputationTracker) -> None:
        """Test bootstrapping with custom initial score."""
        reputation = tracker.bootstrap_agent("agent-001", initial_score=0.8)

        assert reputation.reliability_score == 0.8

    def test_bootstrap_score_clamped_to_bounds(self) -> None:
        """Test that bootstrap score is clamped to config bounds."""
        config = ReputationConfig(min_score=0.2, max_score=0.9)
        tracker = ReputationTracker(config)

        rep_low = tracker.bootstrap_agent("agent-low", initial_score=0.05)
        assert rep_low.reliability_score == 0.2  # Clamped to min

        rep_high = tracker.bootstrap_agent("agent-high", initial_score=0.99)
        assert rep_high.reliability_score == 0.9  # Clamped to max

    def test_bootstrap_existing_agent_raises(self, tracker: ReputationTracker) -> None:
        """Test that bootstrapping existing agent raises error."""
        tracker.bootstrap_agent("agent-001")

        with pytest.raises(ValueError, match="already has a reputation"):
            tracker.bootstrap_agent("agent-001")

    def test_bootstrap_after_record_raises(self, tracker: ReputationTracker) -> None:
        """Test that bootstrapping after recording raises error."""
        tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")

        with pytest.raises(ValueError, match="already has a reputation"):
            tracker.bootstrap_agent("agent-001")


class TestReputationTrackerPruneOldOutcomes:
    """Tests for ReputationTracker.prune_old_outcomes."""

    def test_prune_removes_old_outcomes(self) -> None:
        """Test that prune removes outcomes older than max_age."""
        tracker = ReputationTracker()

        # Add outcomes with old timestamps
        old_time = datetime.now(timezone.utc) - timedelta(hours=200)
        new_time = datetime.now(timezone.utc)

        reputation = tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")

        # Manually add an old outcome
        old_outcome = AgentOutcome(
            agent_id="agent-001",
            tag_id="tag-0",
            predicted_irdi="irdi-old",
            final_irdi="irdi-old",
            timestamp=old_time,
        )
        reputation.outcomes.insert(0, old_outcome)

        # Verify we have 2 outcomes
        assert len(reputation.outcomes) == 2

        # Prune with 168 hours (7 days) max age
        pruned = tracker.prune_old_outcomes(max_age_hours=168.0)

        # Should have pruned 1 outcome
        assert pruned == 1
        reputation = tracker.get_reputation("agent-001")
        assert reputation is not None
        assert len(reputation.outcomes) == 1

    def test_prune_no_old_outcomes(self) -> None:
        """Test prune when all outcomes are recent."""
        tracker = ReputationTracker()
        tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")
        tracker.record_outcome("agent-001", "tag-2", "irdi-b", "irdi-b")

        pruned = tracker.prune_old_outcomes(max_age_hours=168.0)

        assert pruned == 0

    def test_prune_empty_tracker(self) -> None:
        """Test prune on empty tracker."""
        tracker = ReputationTracker()
        pruned = tracker.prune_old_outcomes()
        assert pruned == 0

    def test_prune_updates_scores(self) -> None:
        """Test that prune recalculates scores after removing outcomes."""
        tracker = ReputationTracker()

        # Record an outcome
        reputation = tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")

        # Add an old wrong outcome
        old_time = datetime.now(timezone.utc) - timedelta(hours=200)
        old_outcome = AgentOutcome(
            agent_id="agent-001",
            tag_id="tag-0",
            predicted_irdi="irdi-old",
            final_irdi="irdi-wrong",  # wrong
            timestamp=old_time,
        )
        reputation.outcomes.insert(0, old_outcome)

        # Recalculate with the wrong outcome
        tracker.update_scores()
        score_before = tracker.get_reliability("agent-001")

        # Prune the old wrong outcome
        tracker.prune_old_outcomes(max_age_hours=168.0)
        score_after = tracker.get_reliability("agent-001")

        # Score should be higher without the wrong outcome
        assert score_after >= score_before


class TestReputationTrackerClear:
    """Tests for ReputationTracker.clear."""

    def test_clear_removes_all_data(self) -> None:
        """Test that clear removes all reputation data."""
        tracker = ReputationTracker()

        tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")
        tracker.record_outcome("agent-002", "tag-1", "irdi-b", "irdi-b")

        assert len(tracker.get_all_reputations()) == 2

        tracker.clear()

        assert len(tracker.get_all_reputations()) == 0
        assert tracker.get_reliability("agent-001") == 0.5  # Returns initial_score


class TestReputationTrackerScoreCalculation:
    """Tests for score calculation edge cases."""

    def test_all_correct_predictions_max_score(self) -> None:
        """Test that all correct predictions give max score."""
        config = ReputationConfig(max_score=0.95)
        tracker = ReputationTracker(config)

        for i in range(10):
            tracker.record_outcome("agent-001", f"tag-{i}", "irdi-a", "irdi-a")

        score = tracker.get_reliability("agent-001")
        assert score == 0.95  # Clamped to max

    def test_all_wrong_predictions_min_score(self) -> None:
        """Test that all wrong predictions give min score."""
        config = ReputationConfig(min_score=0.1)
        tracker = ReputationTracker(config)

        for i in range(10):
            tracker.record_outcome("agent-001", f"tag-{i}", "irdi-a", "irdi-b")

        score = tracker.get_reliability("agent-001")
        assert score == 0.1  # Clamped to min

    def test_decay_factor_effect(self) -> None:
        """Test that decay factor gives more weight to recent outcomes."""
        tracker = ReputationTracker()

        # Record older correct, newer wrong
        tracker.record_outcome("agent-001", "tag-1", "irdi-a", "irdi-a")  # correct
        tracker.record_outcome("agent-001", "tag-2", "irdi-a", "irdi-b")  # wrong

        score_newer_wrong = tracker.get_reliability("agent-001")

        # Now test with opposite order (same outcomes but different timing)
        tracker2 = ReputationTracker()
        tracker2.record_outcome("agent-002", "tag-1", "irdi-a", "irdi-b")  # wrong
        tracker2.record_outcome("agent-002", "tag-2", "irdi-a", "irdi-a")  # correct

        score_newer_correct = tracker2.get_reliability("agent-002")

        # Agent with newer correct prediction should have higher score
        assert score_newer_correct > score_newer_wrong


class TestReputationTrackerThreadSafety:
    """Tests for thread safety of ReputationTracker."""

    def test_concurrent_record_outcomes(self) -> None:
        """Test that concurrent record_outcome calls are thread-safe."""
        tracker = ReputationTracker()
        num_threads = 10
        outcomes_per_thread = 100

        def record_outcomes(agent_id: str) -> None:
            for i in range(outcomes_per_thread):
                tracker.record_outcome(
                    agent_id,
                    f"tag-{i}",
                    "irdi-a",
                    "irdi-a" if i % 2 == 0 else "irdi-b",
                )

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=record_outcomes, args=(f"agent-{i}",))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All agents should have recorded outcomes
        reputations = tracker.get_all_reputations()
        assert len(reputations) == num_threads

        for agent_id, reputation in reputations.items():
            assert reputation.total_predictions == outcomes_per_thread

    def test_concurrent_read_and_write(self) -> None:
        """Test concurrent read and write operations."""
        tracker = ReputationTracker()
        num_iterations = 100

        def writer() -> None:
            for i in range(num_iterations):
                tracker.record_outcome("agent-001", f"tag-{i}", "irdi-a", "irdi-a")

        def reader() -> None:
            for _ in range(num_iterations):
                tracker.get_reliability("agent-001")
                tracker.get_reputation("agent-001")
                tracker.get_all_reputations()

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        # Should complete without errors
        reputation = tracker.get_reputation("agent-001")
        assert reputation is not None
        assert reputation.total_predictions == num_iterations


class TestReputationTrackerIntegration:
    """Integration tests for ReputationTracker."""

    def test_full_workflow(self) -> None:
        """Test complete workflow from bootstrap to pruning."""
        # 1. Create tracker with custom config
        config = ReputationConfig(
            window_size=10,
            initial_score=0.5,
            min_score=0.1,
            max_score=0.95,
        )
        tracker = ReputationTracker(config)

        # 2. Bootstrap some agents
        tracker.bootstrap_agent("agent-001", initial_score=0.7)
        tracker.bootstrap_agent("agent-002", initial_score=0.6)

        # 3. Record outcomes
        for i in range(5):
            # agent-001 is always correct
            tracker.record_outcome("agent-001", f"tag-{i}", "irdi-a", "irdi-a")
            # agent-002 alternates
            final = "irdi-a" if i % 2 == 0 else "irdi-b"
            tracker.record_outcome("agent-002", f"tag-{i}", "irdi-a", final)

        # 4. Check scores
        score_001 = tracker.get_reliability("agent-001")
        score_002 = tracker.get_reliability("agent-002")

        assert score_001 > score_002  # agent-001 more reliable

        # 5. Get all reputations
        reputations = tracker.get_all_reputations()
        assert len(reputations) == 2

        # 6. Verify accuracy
        rep_001 = reputations["agent-001"]
        rep_002 = reputations["agent-002"]

        assert rep_001.accuracy == 1.0  # All correct
        assert 0.4 < rep_002.accuracy < 0.7  # Some wrong

    def test_swarm_import(self) -> None:
        """Test that reputation classes can be imported from swarm module."""
        from noa_swarm.swarm import (
            AgentOutcome,
            AgentReputation,
            ReputationConfig,
            ReputationTracker,
        )

        # Verify imports work
        assert ReputationConfig is not None
        assert ReputationTracker is not None
        assert AgentOutcome is not None
        assert AgentReputation is not None

        # Create instances
        config = ReputationConfig()
        tracker = ReputationTracker(config)
        outcome = AgentOutcome(
            agent_id="test",
            tag_id="tag",
            predicted_irdi="a",
            final_irdi="a",
        )
        reputation = AgentReputation(agent_id="test", reliability_score=0.5)

        assert tracker is not None
        assert outcome.was_correct is True
        assert reputation.accuracy == 0.0
