"""Swarm intelligence module for NOA Semantic Swarm Mapper.

This module provides the distributed membership and consensus infrastructure
using the SWIM (Scalable Weakly-consistent Infection-style Membership) protocol
and MQTT-based gossip communication.

Key components:
- **SwarmMember**: Dataclass representing a member in the swarm
- **SwarmMembership**: Main class for managing swarm membership and failure detection
- **HypothesisGossip**: MQTT-based gossip protocol for hypothesis/vote/consensus sharing
- **GossipMessage**: Message format for gossip communication
- **ConsensusConfig**: Configuration for consensus thresholds and weights
- **WeightedVote**: A vote with computed weights from calibration, reliability, and freshness
- **ConsensusEngine**: Main engine for aggregating votes and determining quorum

Example usage:
    >>> from noa_swarm.swarm import SwarmMembership, SwarmMember
    >>> membership = SwarmMembership(
    ...     agent_id="agent-001",
    ...     host="192.168.1.100",
    ...     port=7946,
    ... )
    >>> await membership.start()
    >>> members = membership.get_alive_members()

    >>> from noa_swarm.swarm import HypothesisGossip
    >>> gossip = HypothesisGossip(mqtt_client, "agent-001")
    >>> await gossip.start()
    >>> await gossip.broadcast_hypothesis(tag_id, hypothesis)

    >>> from noa_swarm.swarm import ConsensusEngine, ConsensusConfig
    >>> config = ConsensusConfig(hard_quorum_threshold=0.8)
    >>> engine = ConsensusEngine(config)
    >>> record = engine.reach_consensus(tag_id, votes, calibration_factors)
"""

from noa_swarm.swarm.consensus import (
    ConsensusConfig,
    ConsensusEngine,
    ConsensusError,
    InsufficientVotesError,
    NoConsensusError,
    WeightedVote,
)
from noa_swarm.swarm.gossip import (
    TOPIC_AGENTS_PREFIX,
    TOPIC_SYSTEM_PREFIX,
    TOPIC_TAGS_PREFIX,
    GossipMessage,
    HypothesisGossip,
    HypothesisGossipError,
    agent_hypothesis_topic,
    agent_status_topic,
    tag_candidates_topic,
    tag_consensus_topic,
)
from noa_swarm.swarm.membership import (
    SwarmAlreadyRunningError,
    SwarmMember,
    SwarmMembership,
    SwarmMembershipError,
    SwarmNotStartedError,
)

__all__ = [
    # Membership
    "SwarmAlreadyRunningError",
    "SwarmMember",
    "SwarmMembership",
    "SwarmMembershipError",
    "SwarmNotStartedError",
    # Gossip
    "GossipMessage",
    "HypothesisGossip",
    "HypothesisGossipError",
    # Topic helpers
    "TOPIC_AGENTS_PREFIX",
    "TOPIC_TAGS_PREFIX",
    "TOPIC_SYSTEM_PREFIX",
    "agent_status_topic",
    "agent_hypothesis_topic",
    "tag_candidates_topic",
    "tag_consensus_topic",
    # Consensus
    "ConsensusConfig",
    "ConsensusEngine",
    "ConsensusError",
    "InsufficientVotesError",
    "NoConsensusError",
    "WeightedVote",
]
