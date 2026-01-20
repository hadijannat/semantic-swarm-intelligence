"""Integration tests for swarm coordination using current domain models."""

from __future__ import annotations

import pytest

from noa_swarm.common.schemas import TagRecord, Vote
from noa_swarm.swarm.consensus import ConsensusConfig, ConsensusEngine
from noa_swarm.swarm.reputation import ReputationConfig, ReputationTracker


class TestSwarmConsensusIntegration:
    """Tests for swarm consensus reaching agreement."""

    @pytest.fixture
    def consensus_engine(self) -> ConsensusEngine:
        return ConsensusEngine(
            ConsensusConfig(
                min_votes=2,
                hard_quorum_threshold=0.7,
                soft_quorum_threshold=0.5,
            )
        )

    @pytest.fixture
    def sample_tag(self) -> TagRecord:
        return TagRecord(
            node_id="ns=2;s=TIC-101.PV",
            browse_name="TIC-101.PV",
            source_server="opc.tcp://localhost:4840",
        )

    def test_consensus_with_unanimous_agreement(
        self,
        consensus_engine: ConsensusEngine,
        sample_tag: TagRecord,
    ) -> None:
        irdi = "0173-1#01-AAA001#001"
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi=irdi,
                confidence=0.95,
                reliability_score=0.9,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi=irdi,
                confidence=0.92,
                reliability_score=0.85,
            ),
            Vote(
                agent_id="agent-003",
                candidate_irdi=irdi,
                confidence=0.88,
                reliability_score=0.8,
            ),
        ]

        record = consensus_engine.reach_consensus(
            sample_tag.tag_id,
            votes,
            calibration_factors={"agent-001": 0.98, "agent-002": 0.95, "agent-003": 0.92},
        )

        assert record.quorum_type == "hard"
        assert record.agreed_irdi == irdi
        assert record.consensus_confidence >= 0.7

    def test_consensus_with_disagreement(
        self,
        consensus_engine: ConsensusEngine,
        sample_tag: TagRecord,
    ) -> None:
        irdi_1 = "0173-1#01-AAA001#001"
        irdi_2 = "0173-1#01-AAA002#001"

        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi=irdi_1,
                confidence=0.9,
                reliability_score=0.9,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi=irdi_1,
                confidence=0.85,
                reliability_score=0.85,
            ),
            Vote(
                agent_id="agent-003",
                candidate_irdi=irdi_2,
                confidence=0.95,
                reliability_score=0.8,
            ),
        ]

        record = consensus_engine.reach_consensus(
            sample_tag.tag_id,
            votes,
            calibration_factors={"agent-001": 0.9, "agent-002": 0.9, "agent-003": 0.9},
        )

        assert record.quorum_type in {"hard", "soft"}
        assert record.agreed_irdi == irdi_1

    def test_consensus_below_quorum(
        self,
        consensus_engine: ConsensusEngine,
        sample_tag: TagRecord,
    ) -> None:
        irdi = "0173-1#01-AAA001#001"

        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi=irdi,
                confidence=0.95,
                reliability_score=0.9,
            )
        ]

        record = consensus_engine.reach_consensus(
            sample_tag.tag_id,
            votes,
            calibration_factors={"agent-001": 0.9},
        )

        assert record.quorum_type == "conflict"


class TestReputationTracker:
    """Tests for agent reliability scoring."""

    @pytest.fixture
    def tracker(self) -> ReputationTracker:
        return ReputationTracker(
            ReputationConfig(
                window_size=20,
                initial_score=0.5,
                decay_factor=0.9,
            )
        )

    def test_reputation_updates_after_outcomes(self, tracker: ReputationTracker) -> None:
        agent_id = "agent-001"
        for _ in range(10):
            tracker.record_outcome(agent_id, "tag-1", "irdi-a", "irdi-a")

        assert tracker.get_reliability(agent_id) > 0.5

        for _ in range(5):
            tracker.record_outcome(agent_id, "tag-1", "irdi-a", "irdi-b")

        assert tracker.get_reliability(agent_id) < 0.95
