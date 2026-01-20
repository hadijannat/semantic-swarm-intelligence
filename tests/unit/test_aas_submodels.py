"""Unit tests for AAS submodel templates."""

from __future__ import annotations

import pytest

from noa_swarm.aas.submodels import (
    ConsensusInfo,
    DiscoveredTag,
    MappingStatistics,
    MappingStatus,
    TagMappingSubmodel,
)


class TestMappingStatus:
    """Tests for MappingStatus enum."""

    def test_all_statuses_defined(self) -> None:
        """Test all expected statuses are defined."""
        assert hasattr(MappingStatus, "PENDING")
        assert hasattr(MappingStatus, "MAPPED")
        assert hasattr(MappingStatus, "VERIFIED")
        assert hasattr(MappingStatus, "REJECTED")
        assert hasattr(MappingStatus, "CONFLICT")


class TestConsensusInfo:
    """Tests for ConsensusInfo dataclass."""

    def test_create_consensus_info(self) -> None:
        """Test creating consensus info."""
        info = ConsensusInfo(
            confidence=0.95,
            agreement_ratio=0.8,
            participating_agents=5,
            voting_round=3,
        )

        assert info.confidence == 0.95
        assert info.agreement_ratio == 0.8
        assert info.participating_agents == 5
        assert info.voting_round == 3

    def test_consensus_info_defaults(self) -> None:
        """Test consensus info has sensible defaults."""
        info = ConsensusInfo()

        assert info.confidence == 0.0
        assert info.agreement_ratio == 0.0
        assert info.participating_agents == 0
        assert info.voting_round == 0

    def test_invalid_confidence_raises(self) -> None:
        """Test confidence outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be"):
            ConsensusInfo(confidence=1.5)
        with pytest.raises(ValueError, match="confidence must be"):
            ConsensusInfo(confidence=-0.1)

    def test_invalid_agreement_ratio_raises(self) -> None:
        """Test agreement_ratio outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="agreement_ratio must be"):
            ConsensusInfo(agreement_ratio=1.5)


class TestDiscoveredTag:
    """Tests for DiscoveredTag dataclass."""

    def test_create_discovered_tag(self) -> None:
        """Test creating a discovered tag."""
        tag = DiscoveredTag(
            tag_name="TIC-101.PV",
            browse_path="/Objects/Plant/Area1/TIC-101/PV",
            irdi="0173-1#02-AAB663#001",
            preferred_name="Temperature",
            status=MappingStatus.MAPPED,
        )

        assert tag.tag_name == "TIC-101.PV"
        assert tag.browse_path == "/Objects/Plant/Area1/TIC-101/PV"
        assert tag.irdi == "0173-1#02-AAB663#001"
        assert tag.preferred_name == "Temperature"
        assert tag.status == MappingStatus.MAPPED

    def test_discovered_tag_defaults(self) -> None:
        """Test discovered tag has default values."""
        tag = DiscoveredTag(
            tag_name="TEST-001",
            browse_path="/Objects/Test",
        )

        assert tag.irdi is None
        assert tag.preferred_name is None
        assert tag.status == MappingStatus.PENDING
        assert tag.consensus is None
        assert tag.discovered_at is not None  # Auto-set

    def test_discovered_tag_with_consensus(self) -> None:
        """Test discovered tag with consensus info."""
        consensus = ConsensusInfo(confidence=0.9, participating_agents=3)
        tag = DiscoveredTag(
            tag_name="FIC-200.SP",
            browse_path="/Objects/Plant/FIC-200/SP",
            irdi="0173-1#02-AAD001#001",
            consensus=consensus,
        )

        assert tag.consensus is not None
        assert tag.consensus.confidence == 0.9

    def test_empty_tag_name_raises(self) -> None:
        """Test empty tag_name raises ValueError."""
        with pytest.raises(ValueError, match="tag_name cannot be empty"):
            DiscoveredTag(tag_name="", browse_path="/test")

    def test_empty_browse_path_raises(self) -> None:
        """Test empty browse_path raises ValueError."""
        with pytest.raises(ValueError, match="browse_path cannot be empty"):
            DiscoveredTag(tag_name="TEST", browse_path="")


class TestMappingStatistics:
    """Tests for MappingStatistics dataclass."""

    def test_create_statistics(self) -> None:
        """Test creating mapping statistics."""
        stats = MappingStatistics(
            total_tags=100,
            mapped_tags=80,
            verified_tags=50,
            conflict_tags=5,
            pending_tags=15,
        )

        assert stats.total_tags == 100
        assert stats.mapped_tags == 80
        assert stats.verified_tags == 50
        assert stats.conflict_tags == 5
        assert stats.pending_tags == 15

    def test_statistics_defaults(self) -> None:
        """Test statistics has default values of 0."""
        stats = MappingStatistics()

        assert stats.total_tags == 0
        assert stats.mapped_tags == 0
        assert stats.verified_tags == 0
        assert stats.conflict_tags == 0
        assert stats.pending_tags == 0

    def test_mapping_rate_property(self) -> None:
        """Test mapping_rate calculation."""
        stats = MappingStatistics(total_tags=100, mapped_tags=80)
        assert stats.mapping_rate == 0.8

    def test_mapping_rate_zero_total(self) -> None:
        """Test mapping_rate is 0 when total_tags is 0."""
        stats = MappingStatistics(total_tags=0)
        assert stats.mapping_rate == 0.0

    def test_negative_values_raise(self) -> None:
        """Test negative values raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            MappingStatistics(total_tags=-1)
        with pytest.raises(ValueError, match="cannot be negative"):
            MappingStatistics(mapped_tags=-1)


class TestTagMappingSubmodel:
    """Tests for TagMappingSubmodel class."""

    def test_create_submodel(self) -> None:
        """Test creating a tag mapping submodel."""
        submodel = TagMappingSubmodel(
            submodel_id="urn:noa:submodel:tagmapping:plant001",
            semantic_id="urn:noa:sm:TagMapping:1.0",
        )

        assert submodel.submodel_id == "urn:noa:submodel:tagmapping:plant001"
        assert submodel.semantic_id == "urn:noa:sm:TagMapping:1.0"
        assert len(submodel.tags) == 0

    def test_add_tag(self) -> None:
        """Test adding a tag to the submodel."""
        submodel = TagMappingSubmodel()
        tag = DiscoveredTag(
            tag_name="TIC-101.PV",
            browse_path="/Objects/TIC-101/PV",
        )

        submodel.add_tag(tag)

        assert len(submodel.tags) == 1
        assert submodel.tags[0].tag_name == "TIC-101.PV"

    def test_add_multiple_tags(self) -> None:
        """Test adding multiple tags."""
        submodel = TagMappingSubmodel()

        for i in range(10):
            tag = DiscoveredTag(
                tag_name=f"TAG-{i:03d}",
                browse_path=f"/Objects/TAG-{i:03d}",
            )
            submodel.add_tag(tag)

        assert len(submodel.tags) == 10

    def test_get_tag_by_name(self) -> None:
        """Test retrieving a tag by name."""
        submodel = TagMappingSubmodel()
        tag = DiscoveredTag(
            tag_name="FIC-200.SP",
            browse_path="/Objects/FIC-200/SP",
        )
        submodel.add_tag(tag)

        found = submodel.get_tag("FIC-200.SP")
        assert found is not None
        assert found.tag_name == "FIC-200.SP"

    def test_get_tag_not_found(self) -> None:
        """Test getting nonexistent tag returns None."""
        submodel = TagMappingSubmodel()
        found = submodel.get_tag("nonexistent")
        assert found is None

    def test_update_tag(self) -> None:
        """Test updating an existing tag."""
        submodel = TagMappingSubmodel()
        tag = DiscoveredTag(
            tag_name="TIC-101.PV",
            browse_path="/Objects/TIC-101/PV",
        )
        submodel.add_tag(tag)

        updated = DiscoveredTag(
            tag_name="TIC-101.PV",
            browse_path="/Objects/TIC-101/PV",
            irdi="0173-1#02-AAB663#001",
            status=MappingStatus.MAPPED,
        )
        submodel.update_tag(updated)

        found = submodel.get_tag("TIC-101.PV")
        assert found is not None
        assert found.irdi == "0173-1#02-AAB663#001"
        assert found.status == MappingStatus.MAPPED

    def test_remove_tag(self) -> None:
        """Test removing a tag."""
        submodel = TagMappingSubmodel()
        tag = DiscoveredTag(
            tag_name="TIC-101.PV",
            browse_path="/Objects/TIC-101/PV",
        )
        submodel.add_tag(tag)

        result = submodel.remove_tag("TIC-101.PV")
        assert result is True
        assert len(submodel.tags) == 0

    def test_remove_nonexistent_tag(self) -> None:
        """Test removing nonexistent tag returns False."""
        submodel = TagMappingSubmodel()
        result = submodel.remove_tag("nonexistent")
        assert result is False

    def test_get_statistics(self) -> None:
        """Test computing statistics from tags."""
        submodel = TagMappingSubmodel()

        # Add various tags with different statuses
        submodel.add_tag(
            DiscoveredTag(tag_name="TAG-001", browse_path="/a", status=MappingStatus.PENDING)
        )
        submodel.add_tag(
            DiscoveredTag(tag_name="TAG-002", browse_path="/b", status=MappingStatus.MAPPED)
        )
        submodel.add_tag(
            DiscoveredTag(tag_name="TAG-003", browse_path="/c", status=MappingStatus.MAPPED)
        )
        submodel.add_tag(
            DiscoveredTag(tag_name="TAG-004", browse_path="/d", status=MappingStatus.VERIFIED)
        )
        submodel.add_tag(
            DiscoveredTag(tag_name="TAG-005", browse_path="/e", status=MappingStatus.CONFLICT)
        )

        stats = submodel.get_statistics()

        assert stats.total_tags == 5
        assert stats.pending_tags == 1
        assert stats.mapped_tags == 2
        assert stats.verified_tags == 1
        assert stats.conflict_tags == 1

    def test_filter_by_status(self) -> None:
        """Test filtering tags by status."""
        submodel = TagMappingSubmodel()

        submodel.add_tag(
            DiscoveredTag(tag_name="TAG-001", browse_path="/a", status=MappingStatus.PENDING)
        )
        submodel.add_tag(
            DiscoveredTag(tag_name="TAG-002", browse_path="/b", status=MappingStatus.MAPPED)
        )
        submodel.add_tag(
            DiscoveredTag(tag_name="TAG-003", browse_path="/c", status=MappingStatus.MAPPED)
        )

        mapped = submodel.filter_by_status(MappingStatus.MAPPED)
        assert len(mapped) == 2

        pending = submodel.filter_by_status(MappingStatus.PENDING)
        assert len(pending) == 1

    def test_to_basyx_submodel(self) -> None:
        """Test converting to BaSyx submodel object."""
        submodel = TagMappingSubmodel(
            submodel_id="urn:test:submodel:1",
        )
        submodel.add_tag(
            DiscoveredTag(
                tag_name="TIC-101.PV",
                browse_path="/Objects/TIC-101/PV",
                irdi="0173-1#02-AAB663#001",
                status=MappingStatus.MAPPED,
            )
        )

        basyx_sm = submodel.to_basyx_submodel()

        # Should return a BaSyx Submodel object
        from basyx.aas.model import Submodel

        assert isinstance(basyx_sm, Submodel)
        assert str(basyx_sm.id) == "urn:test:submodel:1"
