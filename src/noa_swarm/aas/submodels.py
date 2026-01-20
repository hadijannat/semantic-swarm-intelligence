"""AAS submodel templates for tag mapping.

This module defines the TagMapping submodel structure for the Asset
Administration Shell (AAS). The submodel contains:

- **DiscoveredTags**: Collection of discovered OPC UA tags
- **Statistics**: Mapping progress and quality metrics
- **ConsensusInfo**: Swarm consensus details per tag

The submodel follows the AAS metamodel specification and can be
exported to AASX packages or JSON/XML formats using BaSyx SDK.

Example usage:
    >>> from noa_swarm.aas.submodels import TagMappingSubmodel, DiscoveredTag
    >>> submodel = TagMappingSubmodel(submodel_id="urn:noa:sm:tagmapping:1")
    >>> submodel.add_tag(DiscoveredTag(
    ...     tag_name="TIC-101.PV",
    ...     browse_path="/Objects/TIC-101/PV",
    ...     irdi="0173-1#02-AAB663#001",
    ... ))
    >>> basyx_sm = submodel.to_basyx_submodel()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from basyx.aas import model

from noa_swarm.common.logging import get_logger

logger = get_logger(__name__)


class MappingStatus(str, Enum):
    """Status of a tag mapping.

    Attributes:
        PENDING: Tag discovered but not yet mapped.
        MAPPED: Tag has been mapped to an IRDI (automated).
        VERIFIED: Mapping has been verified by human/validation.
        REJECTED: Mapping was rejected as incorrect.
        CONFLICT: Multiple conflicting mappings exist.
    """

    PENDING = "pending"
    MAPPED = "mapped"
    VERIFIED = "verified"
    REJECTED = "rejected"
    CONFLICT = "conflict"


@dataclass
class ConsensusInfo:
    """Consensus information for a tag mapping.

    Contains details about the swarm consensus process that
    produced the mapping.

    Attributes:
        confidence: Calibrated confidence score (0.0 to 1.0).
        agreement_ratio: Proportion of agents that agreed (0.0 to 1.0).
        participating_agents: Number of agents that voted.
        voting_round: The consensus round number.
    """

    confidence: float = 0.0
    agreement_ratio: float = 0.0
    participating_agents: int = 0
    voting_round: int = 0

    def __post_init__(self) -> None:
        """Validate consensus info fields."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if not (0.0 <= self.agreement_ratio <= 1.0):
            raise ValueError(
                f"agreement_ratio must be between 0.0 and 1.0, "
                f"got {self.agreement_ratio}"
            )


@dataclass
class DiscoveredTag:
    """A discovered tag with its semantic mapping.

    Represents an OPC UA node that has been discovered and potentially
    mapped to an IEC 61987 IRDI.

    Attributes:
        tag_name: The tag identifier (e.g., "TIC-101.PV").
        browse_path: OPC UA browse path to the node.
        irdi: Mapped IEC 61987 IRDI (if mapped).
        preferred_name: Human-readable name from dictionary.
        status: Current mapping status.
        consensus: Consensus information (if from swarm).
        discovered_at: Timestamp when tag was discovered.
        updated_at: Timestamp of last update.
        data_type: OPC UA data type of the tag.
        unit: Unit of measure (from dictionary or OPC UA).
        node_id: OPC UA NodeId string.
    """

    tag_name: str
    browse_path: str
    irdi: str | None = None
    preferred_name: str | None = None
    status: MappingStatus = MappingStatus.PENDING
    consensus: ConsensusInfo | None = None
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None
    data_type: str | None = None
    unit: str | None = None
    node_id: str | None = None

    def __post_init__(self) -> None:
        """Validate tag fields."""
        if not self.tag_name:
            raise ValueError("tag_name cannot be empty")
        if not self.browse_path:
            raise ValueError("browse_path cannot be empty")


@dataclass
class MappingStatistics:
    """Statistics for tag mapping progress.

    Aggregated statistics computed from the collection of discovered tags.

    Attributes:
        total_tags: Total number of discovered tags.
        mapped_tags: Tags with MAPPED status.
        verified_tags: Tags with VERIFIED status.
        conflict_tags: Tags with CONFLICT status.
        pending_tags: Tags with PENDING status.
        rejected_tags: Tags with REJECTED status.
        average_confidence: Mean confidence across mapped tags.
        last_updated: Timestamp of last statistics update.
    """

    total_tags: int = 0
    mapped_tags: int = 0
    verified_tags: int = 0
    conflict_tags: int = 0
    pending_tags: int = 0
    rejected_tags: int = 0
    average_confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate statistics fields."""
        for field_name in [
            "total_tags",
            "mapped_tags",
            "verified_tags",
            "conflict_tags",
            "pending_tags",
            "rejected_tags",
        ]:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative, got {value}")

    @property
    def mapping_rate(self) -> float:
        """Calculate the mapping rate.

        Returns:
            Fraction of tags that have been mapped (0.0 to 1.0).
        """
        if self.total_tags == 0:
            return 0.0
        return self.mapped_tags / self.total_tags


class TagMappingSubmodel:
    """Submodel template for tag mapping data.

    This class manages a collection of discovered tags and provides
    methods for adding, updating, and querying tags. It can be
    converted to a BaSyx Submodel object for AAS export.

    The submodel structure follows PA-DIM conventions:
    - DiscoveredTags: SubmodelElementCollection containing tag entries
    - Statistics: SubmodelElementCollection with mapping metrics

    Attributes:
        submodel_id: Unique identifier for the submodel.
        semantic_id: Semantic identifier (typically a URI).
        tags: List of discovered tags.
    """

    DEFAULT_SEMANTIC_ID = "urn:noa:sm:TagMapping:1.0"

    def __init__(
        self,
        submodel_id: str = "urn:noa:submodel:tagmapping:default",
        semantic_id: str | None = None,
    ) -> None:
        """Initialize the tag mapping submodel.

        Args:
            submodel_id: Unique identifier for this submodel instance.
            semantic_id: Semantic identifier URI.
        """
        self.submodel_id = submodel_id
        self.semantic_id = semantic_id or self.DEFAULT_SEMANTIC_ID
        self._tags: dict[str, DiscoveredTag] = {}

        logger.debug(
            "Created TagMappingSubmodel",
            submodel_id=submodel_id,
            semantic_id=self.semantic_id,
        )

    @property
    def tags(self) -> list[DiscoveredTag]:
        """Return list of all discovered tags."""
        return list(self._tags.values())

    def add_tag(self, tag: DiscoveredTag) -> None:
        """Add a discovered tag to the submodel.

        Args:
            tag: The tag to add.
        """
        self._tags[tag.tag_name] = tag
        logger.debug("Added tag to submodel", tag_name=tag.tag_name)

    def get_tag(self, tag_name: str) -> DiscoveredTag | None:
        """Get a tag by its name.

        Args:
            tag_name: Name of the tag to retrieve.

        Returns:
            The tag if found, None otherwise.
        """
        return self._tags.get(tag_name)

    def update_tag(self, tag: DiscoveredTag) -> bool:
        """Update an existing tag.

        Args:
            tag: The updated tag (matched by tag_name).

        Returns:
            True if tag was updated, False if not found.
        """
        if tag.tag_name not in self._tags:
            return False

        tag.updated_at = datetime.utcnow()
        self._tags[tag.tag_name] = tag
        logger.debug("Updated tag in submodel", tag_name=tag.tag_name)
        return True

    def remove_tag(self, tag_name: str) -> bool:
        """Remove a tag from the submodel.

        Args:
            tag_name: Name of the tag to remove.

        Returns:
            True if tag was removed, False if not found.
        """
        if tag_name in self._tags:
            del self._tags[tag_name]
            logger.debug("Removed tag from submodel", tag_name=tag_name)
            return True
        return False

    def filter_by_status(self, status: MappingStatus) -> list[DiscoveredTag]:
        """Get all tags with a specific status.

        Args:
            status: The status to filter by.

        Returns:
            List of tags with the specified status.
        """
        return [tag for tag in self._tags.values() if tag.status == status]

    def get_statistics(self) -> MappingStatistics:
        """Compute statistics from current tags.

        Returns:
            MappingStatistics with current metrics.
        """
        tags = list(self._tags.values())
        total = len(tags)

        pending = sum(1 for t in tags if t.status == MappingStatus.PENDING)
        mapped = sum(1 for t in tags if t.status == MappingStatus.MAPPED)
        verified = sum(1 for t in tags if t.status == MappingStatus.VERIFIED)
        conflict = sum(1 for t in tags if t.status == MappingStatus.CONFLICT)
        rejected = sum(1 for t in tags if t.status == MappingStatus.REJECTED)

        # Calculate average confidence for mapped/verified tags
        confidences = [
            t.consensus.confidence
            for t in tags
            if t.consensus is not None
            and t.status in (MappingStatus.MAPPED, MappingStatus.VERIFIED)
        ]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return MappingStatistics(
            total_tags=total,
            mapped_tags=mapped,
            verified_tags=verified,
            conflict_tags=conflict,
            pending_tags=pending,
            rejected_tags=rejected,
            average_confidence=avg_conf,
        )

    def to_basyx_submodel(self) -> model.Submodel:
        """Convert to a BaSyx Submodel object.

        Creates an AAS-compliant Submodel with:
        - DiscoveredTags collection containing all tags
        - Statistics collection with mapping metrics

        Returns:
            BaSyx Submodel object ready for export.
        """
        # Create semantic ID reference
        semantic_id = model.ExternalReference(
            (model.Key(model.KeyTypes.GLOBAL_REFERENCE, self.semantic_id),)
        )

        # Create the submodel
        submodel = model.Submodel(
            id_=self.submodel_id,
            semantic_id=semantic_id,
        )

        # Add DiscoveredTags collection
        tags_collection = self._create_tags_collection()
        submodel.submodel_element.add(tags_collection)

        # Add Statistics collection
        stats_collection = self._create_statistics_collection()
        submodel.submodel_element.add(stats_collection)

        logger.info(
            "Created BaSyx submodel",
            submodel_id=self.submodel_id,
            tag_count=len(self._tags),
        )

        return submodel

    def _create_tags_collection(self) -> model.SubmodelElementCollection:
        """Create the DiscoveredTags SubmodelElementCollection.

        Returns:
            Collection containing all tag entries.
        """
        collection = model.SubmodelElementCollection(
            id_short="DiscoveredTags",
            semantic_id=model.ExternalReference(
                (
                    model.Key(
                        model.KeyTypes.GLOBAL_REFERENCE,
                        "urn:noa:sme:DiscoveredTags:1.0",
                    ),
                )
            ),
        )

        for tag in self._tags.values():
            tag_smc = self._create_tag_element(tag)
            collection.value.add(tag_smc)

        return collection

    def _create_tag_element(
        self, tag: DiscoveredTag
    ) -> model.SubmodelElementCollection:
        """Create a SubmodelElementCollection for a single tag.

        Args:
            tag: The tag to convert.

        Returns:
            SubmodelElementCollection representing the tag.
        """
        # Use sanitized tag name as id_short
        id_short = self._sanitize_id_short(tag.tag_name)

        tag_smc = model.SubmodelElementCollection(id_short=id_short)

        # Add properties
        tag_smc.value.add(
            model.Property(
                id_short="TagName",
                value_type=model.datatypes.String,
                value=tag.tag_name,
            )
        )

        tag_smc.value.add(
            model.Property(
                id_short="BrowsePath",
                value_type=model.datatypes.String,
                value=tag.browse_path,
            )
        )

        tag_smc.value.add(
            model.Property(
                id_short="Status",
                value_type=model.datatypes.String,
                value=tag.status.value,
            )
        )

        if tag.irdi:
            tag_smc.value.add(
                model.Property(
                    id_short="IRDI",
                    value_type=model.datatypes.String,
                    value=tag.irdi,
                )
            )

        if tag.preferred_name:
            tag_smc.value.add(
                model.Property(
                    id_short="PreferredName",
                    value_type=model.datatypes.String,
                    value=tag.preferred_name,
                )
            )

        if tag.consensus:
            tag_smc.value.add(
                model.Property(
                    id_short="Confidence",
                    value_type=model.datatypes.Float,
                    value=tag.consensus.confidence,
                )
            )

        tag_smc.value.add(
            model.Property(
                id_short="DiscoveredAt",
                value_type=model.datatypes.String,
                value=tag.discovered_at.isoformat(),
            )
        )

        return tag_smc

    def _create_statistics_collection(self) -> model.SubmodelElementCollection:
        """Create the Statistics SubmodelElementCollection.

        Returns:
            Collection containing mapping statistics.
        """
        stats = self.get_statistics()

        collection = model.SubmodelElementCollection(
            id_short="Statistics",
            semantic_id=model.ExternalReference(
                (
                    model.Key(
                        model.KeyTypes.GLOBAL_REFERENCE,
                        "urn:noa:sme:MappingStatistics:1.0",
                    ),
                )
            ),
        )

        collection.value.add(
            model.Property(
                id_short="TotalTags",
                value_type=model.datatypes.Int,
                value=stats.total_tags,
            )
        )

        collection.value.add(
            model.Property(
                id_short="MappedTags",
                value_type=model.datatypes.Int,
                value=stats.mapped_tags,
            )
        )

        collection.value.add(
            model.Property(
                id_short="VerifiedTags",
                value_type=model.datatypes.Int,
                value=stats.verified_tags,
            )
        )

        collection.value.add(
            model.Property(
                id_short="ConflictTags",
                value_type=model.datatypes.Int,
                value=stats.conflict_tags,
            )
        )

        collection.value.add(
            model.Property(
                id_short="PendingTags",
                value_type=model.datatypes.Int,
                value=stats.pending_tags,
            )
        )

        collection.value.add(
            model.Property(
                id_short="MappingRate",
                value_type=model.datatypes.Float,
                value=stats.mapping_rate,
            )
        )

        collection.value.add(
            model.Property(
                id_short="AverageConfidence",
                value_type=model.datatypes.Float,
                value=stats.average_confidence,
            )
        )

        return collection

    @staticmethod
    def _sanitize_id_short(name: str) -> str:
        """Sanitize a name for use as AAS id_short.

        AAS id_short must match [a-zA-Z][a-zA-Z0-9_]*

        Args:
            name: The name to sanitize.

        Returns:
            Sanitized id_short string.
        """
        # Replace common separators with underscore
        result = name.replace(".", "_").replace("-", "_").replace(" ", "_")

        # Ensure starts with letter
        if result and not result[0].isalpha():
            result = "Tag_" + result

        # Remove any remaining invalid characters
        result = "".join(c if c.isalnum() or c == "_" else "_" for c in result)

        return result or "Unknown"
