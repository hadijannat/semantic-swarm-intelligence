"""Unit tests for core data schemas."""

from __future__ import annotations

from datetime import UTC

import pytest
from pydantic import ValidationError

from noa_swarm.common.schemas import (
    Candidate,
    ConsensusRecord,
    Hypothesis,
    TagRecord,
    Vote,
)


class TestTagRecord:
    """Tests for TagRecord schema."""

    def test_create_minimal_tag_record(self) -> None:
        """Test creating TagRecord with required fields only."""
        tag = TagRecord(
            node_id="ns=2;s=Temperature.PV",
            browse_name="Temperature",
            source_server="opc.tcp://localhost:4840",
        )

        assert tag.node_id == "ns=2;s=Temperature.PV"
        assert tag.browse_name == "Temperature"
        assert tag.source_server == "opc.tcp://localhost:4840"
        assert tag.display_name is None
        assert tag.data_type is None
        assert tag.description is None
        assert tag.parent_path == []

    def test_create_full_tag_record(self) -> None:
        """Test creating TagRecord with all fields."""
        tag = TagRecord(
            node_id="ns=2;s=Reactor1.Temperature.PV",
            browse_name="Temperature",
            display_name="Reactor 1 Temperature",
            data_type="Double",
            description="Process temperature measurement",
            parent_path=["Objects", "Reactor1", "Sensors"],
            source_server="opc.tcp://192.168.1.100:4840",
            engineering_unit="degC",
            access_level=3,
        )

        assert tag.display_name == "Reactor 1 Temperature"
        assert tag.data_type == "Double"
        assert tag.description == "Process temperature measurement"
        assert tag.parent_path == ["Objects", "Reactor1", "Sensors"]
        assert tag.engineering_unit == "degC"
        assert tag.access_level == 3

    def test_tag_record_full_path(self) -> None:
        """Test full_path property."""
        tag = TagRecord(
            node_id="ns=2;s=Temperature.PV",
            browse_name="Temperature",
            parent_path=["Objects", "Reactor1"],
            source_server="opc.tcp://localhost:4840",
        )

        assert tag.full_path == "Objects/Reactor1/Temperature"

    def test_tag_record_full_path_no_parent(self) -> None:
        """Test full_path property with no parent path."""
        tag = TagRecord(
            node_id="ns=2;s=Temperature.PV",
            browse_name="Temperature",
            source_server="opc.tcp://localhost:4840",
        )

        assert tag.full_path == "Temperature"

    def test_tag_record_tag_id(self) -> None:
        """Test tag_id property combines server and node_id."""
        tag = TagRecord(
            node_id="ns=2;s=Temperature.PV",
            browse_name="Temperature",
            source_server="opc.tcp://localhost:4840",
        )

        assert tag.tag_id == "opc.tcp://localhost:4840|ns=2;s=Temperature.PV"

    def test_tag_record_discovered_at_default(self) -> None:
        """Test discovered_at has default value."""
        tag = TagRecord(
            node_id="ns=2;s=Temperature.PV",
            browse_name="Temperature",
            source_server="opc.tcp://localhost:4840",
        )

        assert tag.discovered_at is not None
        assert tag.discovered_at.tzinfo == UTC

    def test_tag_record_immutable(self) -> None:
        """Test TagRecord is immutable."""
        tag = TagRecord(
            node_id="ns=2;s=Temperature.PV",
            browse_name="Temperature",
            source_server="opc.tcp://localhost:4840",
        )

        with pytest.raises(ValidationError):
            tag.browse_name = "NewName"  # type: ignore[misc]

    def test_tag_record_validation_empty_node_id(self) -> None:
        """Test validation rejects empty node_id."""
        with pytest.raises(ValidationError, match="node_id"):
            TagRecord(
                node_id="",
                browse_name="Temperature",
                source_server="opc.tcp://localhost:4840",
            )


class TestCandidate:
    """Tests for Candidate schema."""

    def test_create_candidate(self) -> None:
        """Test creating a Candidate."""
        candidate = Candidate(
            irdi="0173-1#01-ABA234#001",
            confidence=0.85,
            source_model="semantic-v1",
        )

        assert candidate.irdi == "0173-1#01-ABA234#001"
        assert candidate.confidence == 0.85
        assert candidate.source_model == "semantic-v1"
        assert candidate.features is None
        assert candidate.reasoning is None

    def test_candidate_with_features(self) -> None:
        """Test creating Candidate with features."""
        candidate = Candidate(
            irdi="0173-1#01-ABA234#001",
            confidence=0.9,
            source_model="embedding-v2",
            features={"embedding": [0.1, 0.2, 0.3], "similarity": 0.95},
            reasoning="High semantic similarity with temperature property",
        )

        assert candidate.features == {"embedding": [0.1, 0.2, 0.3], "similarity": 0.95}
        assert candidate.reasoning == "High semantic similarity with temperature property"

    def test_candidate_irdi_validation(self) -> None:
        """Test that invalid IRDI is rejected."""
        with pytest.raises(ValidationError, match="Invalid IRDI format"):
            Candidate(
                irdi="invalid-irdi",
                confidence=0.85,
                source_model="semantic-v1",
            )

    def test_candidate_irdi_normalization(self) -> None:
        """Test that IRDI is normalized to canonical form."""
        candidate = Candidate(
            irdi="0173-1#01-aba234#001",  # lowercase
            confidence=0.85,
            source_model="semantic-v1",
        )

        assert candidate.irdi == "0173-1#01-ABA234#001"  # uppercase

    def test_candidate_confidence_bounds(self) -> None:
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValidationError, match="confidence"):
            Candidate(
                irdi="0173-1#01-ABA234#001",
                confidence=1.5,  # Invalid
                source_model="semantic-v1",
            )

        with pytest.raises(ValidationError, match="confidence"):
            Candidate(
                irdi="0173-1#01-ABA234#001",
                confidence=-0.1,  # Invalid
                source_model="semantic-v1",
            )

    def test_candidate_parsed_irdi(self) -> None:
        """Test parsed_irdi property."""
        candidate = Candidate(
            irdi="0173-1#01-ABA234#001",
            confidence=0.85,
            source_model="semantic-v1",
        )

        irdi = candidate.parsed_irdi
        assert irdi.org_code == "0173-1"
        assert irdi.item_code == "ABA234"


class TestHypothesis:
    """Tests for Hypothesis schema."""

    def test_create_hypothesis(self) -> None:
        """Test creating a Hypothesis."""
        candidates = [
            Candidate(irdi="0173-1#01-ABA234#001", confidence=0.9, source_model="v1"),
            Candidate(irdi="0173-1#01-XYZ789#001", confidence=0.7, source_model="v1"),
        ]

        hypothesis = Hypothesis(
            tag_id="server1|ns=2;s=Temp",
            candidates=candidates,
            agent_id="agent-001",
        )

        assert hypothesis.tag_id == "server1|ns=2;s=Temp"
        assert hypothesis.agent_id == "agent-001"
        assert len(hypothesis.candidates) == 2

    def test_hypothesis_candidates_sorted_by_confidence(self) -> None:
        """Test that candidates are sorted by confidence (highest first)."""
        candidates = [
            Candidate(irdi="0173-1#01-LOW111#001", confidence=0.3, source_model="v1"),
            Candidate(irdi="0173-1#01-HIGH99#001", confidence=0.95, source_model="v1"),
            Candidate(irdi="0173-1#01-MID555#001", confidence=0.6, source_model="v1"),
        ]

        hypothesis = Hypothesis(
            tag_id="server1|ns=2;s=Temp",
            candidates=candidates,
            agent_id="agent-001",
        )

        assert hypothesis.candidates[0].confidence == 0.95
        assert hypothesis.candidates[1].confidence == 0.6
        assert hypothesis.candidates[2].confidence == 0.3

    def test_hypothesis_top_candidate(self) -> None:
        """Test top_candidate property."""
        candidates = [
            Candidate(irdi="0173-1#01-ABA234#001", confidence=0.9, source_model="v1"),
            Candidate(irdi="0173-1#01-XYZ789#001", confidence=0.7, source_model="v1"),
        ]

        hypothesis = Hypothesis(
            tag_id="server1|ns=2;s=Temp",
            candidates=candidates,
            agent_id="agent-001",
        )

        assert hypothesis.top_candidate is not None
        assert hypothesis.top_candidate.confidence == 0.9

    def test_hypothesis_top_irdi(self) -> None:
        """Test top_irdi property."""
        candidates = [
            Candidate(irdi="0173-1#01-ABA234#001", confidence=0.9, source_model="v1"),
        ]

        hypothesis = Hypothesis(
            tag_id="server1|ns=2;s=Temp",
            candidates=candidates,
            agent_id="agent-001",
        )

        assert hypothesis.top_irdi == "0173-1#01-ABA234#001"

    def test_hypothesis_requires_at_least_one_candidate(self) -> None:
        """Test that at least one candidate is required."""
        with pytest.raises(ValidationError, match="candidates"):
            Hypothesis(
                tag_id="server1|ns=2;s=Temp",
                candidates=[],
                agent_id="agent-001",
            )


class TestVote:
    """Tests for Vote schema."""

    def test_create_vote(self) -> None:
        """Test creating a Vote."""
        vote = Vote(
            agent_id="agent-001",
            candidate_irdi="0173-1#01-ABA234#001",
            confidence=0.85,
            reliability_score=0.92,
        )

        assert vote.agent_id == "agent-001"
        assert vote.candidate_irdi == "0173-1#01-ABA234#001"
        assert vote.confidence == 0.85
        assert vote.reliability_score == 0.92

    def test_vote_irdi_normalization(self) -> None:
        """Test IRDI is normalized."""
        vote = Vote(
            agent_id="agent-001",
            candidate_irdi="0173-1#01-aba234#001",
            confidence=0.85,
            reliability_score=0.92,
        )

        assert vote.candidate_irdi == "0173-1#01-ABA234#001"

    def test_vote_weighted_confidence(self) -> None:
        """Test weighted_confidence property."""
        vote = Vote(
            agent_id="agent-001",
            candidate_irdi="0173-1#01-ABA234#001",
            confidence=0.8,
            reliability_score=0.5,
        )

        assert vote.weighted_confidence == 0.4  # 0.8 * 0.5

    def test_vote_immutable(self) -> None:
        """Test Vote is immutable."""
        vote = Vote(
            agent_id="agent-001",
            candidate_irdi="0173-1#01-ABA234#001",
            confidence=0.85,
            reliability_score=0.92,
        )

        with pytest.raises(ValidationError):
            vote.confidence = 0.5  # type: ignore[misc]

    def test_vote_timestamp_default(self) -> None:
        """Test timestamp has default value."""
        vote = Vote(
            agent_id="agent-001",
            candidate_irdi="0173-1#01-ABA234#001",
            confidence=0.85,
            reliability_score=0.92,
        )

        assert vote.timestamp is not None
        assert vote.timestamp.tzinfo == UTC


class TestConsensusRecord:
    """Tests for ConsensusRecord schema."""

    def create_sample_votes(self) -> list[Vote]:
        """Create sample votes for testing."""
        return [
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
                reliability_score=0.88,
            ),
            Vote(
                agent_id="agent-003",
                candidate_irdi="0173-1#01-ABA234#001",
                confidence=0.8,
                reliability_score=0.92,
            ),
        ]

    def test_create_consensus_record(self) -> None:
        """Test creating a ConsensusRecord."""
        votes = self.create_sample_votes()

        record = ConsensusRecord(
            tag_id="server1|ns=2;s=Temp",
            agreed_irdi="0173-1#01-ABA234#001",
            consensus_confidence=0.88,
            votes=votes,
            quorum_type="hard",
        )

        assert record.tag_id == "server1|ns=2;s=Temp"
        assert record.agreed_irdi == "0173-1#01-ABA234#001"
        assert record.consensus_confidence == 0.88
        assert record.quorum_type == "hard"
        assert record.human_validated is False
        assert len(record.votes) == 3

    def test_consensus_record_quorum_types(self) -> None:
        """Test different quorum types."""
        votes = self.create_sample_votes()

        for quorum_type in ["hard", "soft", "conflict"]:
            record = ConsensusRecord(
                tag_id="server1|ns=2;s=Temp",
                agreed_irdi="0173-1#01-ABA234#001",
                consensus_confidence=0.88,
                votes=votes,
                quorum_type=quorum_type,  # type: ignore[arg-type]
            )
            assert record.quorum_type == quorum_type

    def test_consensus_record_invalid_quorum_type(self) -> None:
        """Test invalid quorum type is rejected."""
        votes = self.create_sample_votes()

        with pytest.raises(ValidationError, match="quorum_type"):
            ConsensusRecord(
                tag_id="server1|ns=2;s=Temp",
                agreed_irdi="0173-1#01-ABA234#001",
                consensus_confidence=0.88,
                votes=votes,
                quorum_type="invalid",  # type: ignore[arg-type]
            )

    def test_consensus_record_vote_count(self) -> None:
        """Test vote_count property."""
        votes = self.create_sample_votes()

        record = ConsensusRecord(
            tag_id="server1|ns=2;s=Temp",
            agreed_irdi="0173-1#01-ABA234#001",
            consensus_confidence=0.88,
            votes=votes,
            quorum_type="hard",
        )

        assert record.vote_count == 3

    def test_consensus_record_unique_voters(self) -> None:
        """Test unique_voters property."""
        votes = self.create_sample_votes()

        record = ConsensusRecord(
            tag_id="server1|ns=2;s=Temp",
            agreed_irdi="0173-1#01-ABA234#001",
            consensus_confidence=0.88,
            votes=votes,
            quorum_type="hard",
        )

        assert record.unique_voters == {"agent-001", "agent-002", "agent-003"}

    def test_consensus_record_is_unanimous(self) -> None:
        """Test is_unanimous property."""
        votes = self.create_sample_votes()

        # Unanimous case
        record = ConsensusRecord(
            tag_id="server1|ns=2;s=Temp",
            agreed_irdi="0173-1#01-ABA234#001",
            consensus_confidence=0.88,
            votes=votes,
            quorum_type="hard",
        )
        assert record.is_unanimous is True

        # Non-unanimous case
        dissenting_votes = votes + [
            Vote(
                agent_id="agent-004",
                candidate_irdi="0173-1#01-XYZ789#001",  # Different IRDI
                confidence=0.7,
                reliability_score=0.8,
            )
        ]
        record2 = ConsensusRecord(
            tag_id="server1|ns=2;s=Temp",
            agreed_irdi="0173-1#01-ABA234#001",
            consensus_confidence=0.75,
            votes=dissenting_votes,
            quorum_type="soft",
        )
        assert record2.is_unanimous is False

    def test_consensus_record_add_audit_entry(self) -> None:
        """Test add_audit_entry creates new record with entry."""
        votes = self.create_sample_votes()

        record = ConsensusRecord(
            tag_id="server1|ns=2;s=Temp",
            agreed_irdi="0173-1#01-ABA234#001",
            consensus_confidence=0.88,
            votes=votes,
            quorum_type="hard",
        )

        new_record = record.add_audit_entry("Consensus reached with 3 votes")

        assert len(new_record.audit_trail) == 1
        assert "Consensus reached with 3 votes" in new_record.audit_trail[0]
        assert len(record.audit_trail) == 0  # Original unchanged

    def test_consensus_record_mark_validated(self) -> None:
        """Test mark_validated creates new validated record."""
        votes = self.create_sample_votes()

        record = ConsensusRecord(
            tag_id="server1|ns=2;s=Temp",
            agreed_irdi="0173-1#01-ABA234#001",
            consensus_confidence=0.88,
            votes=votes,
            quorum_type="hard",
        )

        validated = record.mark_validated("Confirmed by domain expert")

        assert validated.human_validated is True
        assert validated.validation_notes == "Confirmed by domain expert"
        assert "Human validated" in validated.audit_trail[-1]
        assert record.human_validated is False  # Original unchanged

    def test_consensus_record_requires_at_least_one_vote(self) -> None:
        """Test that at least one vote is required."""
        with pytest.raises(ValidationError, match="votes"):
            ConsensusRecord(
                tag_id="server1|ns=2;s=Temp",
                agreed_irdi="0173-1#01-ABA234#001",
                consensus_confidence=0.88,
                votes=[],
                quorum_type="hard",
            )

    def test_consensus_record_parsed_irdi(self) -> None:
        """Test parsed_irdi property."""
        votes = self.create_sample_votes()

        record = ConsensusRecord(
            tag_id="server1|ns=2;s=Temp",
            agreed_irdi="0173-1#01-ABA234#001",
            consensus_confidence=0.88,
            votes=votes,
            quorum_type="hard",
        )

        irdi = record.parsed_irdi
        assert irdi.org_code == "0173-1"
        assert irdi.item_code == "ABA234"


class TestSchemaIntegration:
    """Integration tests for schema interactions."""

    def test_full_workflow(self) -> None:
        """Test complete workflow from tag discovery to consensus."""
        # 1. Discover a tag
        tag = TagRecord(
            node_id="ns=2;s=Reactor1.Temperature.PV",
            browse_name="Temperature",
            display_name="Reactor 1 Temperature",
            data_type="Double",
            parent_path=["Objects", "Reactor1"],
            source_server="opc.tcp://localhost:4840",
        )

        # 2. Generate candidates
        candidates = [
            Candidate(
                irdi="0173-1#02-AAA001#001",
                confidence=0.92,
                source_model="semantic-v1",
                reasoning="High match with temperature property",
            ),
            Candidate(
                irdi="0173-1#02-BBB002#001",
                confidence=0.75,
                source_model="semantic-v1",
            ),
        ]

        # 3. Create hypothesis
        hypothesis = Hypothesis(
            tag_id=tag.tag_id,
            candidates=candidates,
            agent_id="agent-001",
        )

        assert hypothesis.top_irdi == "0173-1#02-AAA001#001"

        # 4. Cast votes
        votes = [
            Vote(
                agent_id="agent-001",
                candidate_irdi="0173-1#02-AAA001#001",
                confidence=0.92,
                reliability_score=0.95,
            ),
            Vote(
                agent_id="agent-002",
                candidate_irdi="0173-1#02-AAA001#001",
                confidence=0.88,
                reliability_score=0.90,
            ),
        ]

        # 5. Reach consensus
        consensus = ConsensusRecord(
            tag_id=tag.tag_id,
            agreed_irdi="0173-1#02-AAA001#001",
            consensus_confidence=0.9,
            votes=votes,
            quorum_type="hard",
            audit_trail=["Initial consensus achieved"],
        )

        assert consensus.is_unanimous
        assert consensus.vote_count == 2

        # 6. Validate
        validated = consensus.mark_validated("Confirmed correct by engineer")
        assert validated.human_validated
