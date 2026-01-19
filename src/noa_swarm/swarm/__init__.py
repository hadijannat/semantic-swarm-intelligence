"""Swarm intelligence module for NOA Semantic Swarm Mapper.

This module provides the distributed membership and consensus infrastructure
using the SWIM (Scalable Weakly-consistent Infection-style Membership) protocol.

Key components:
- **SwarmMember**: Dataclass representing a member in the swarm
- **SwarmMembership**: Main class for managing swarm membership and failure detection

Example usage:
    >>> from noa_swarm.swarm import SwarmMembership, SwarmMember
    >>> membership = SwarmMembership(
    ...     agent_id="agent-001",
    ...     host="192.168.1.100",
    ...     port=7946,
    ... )
    >>> await membership.start()
    >>> members = membership.get_alive_members()
"""

from noa_swarm.swarm.membership import (
    SwarmAlreadyRunningError,
    SwarmMember,
    SwarmMembership,
    SwarmMembershipError,
    SwarmNotStartedError,
)

__all__ = [
    "SwarmAlreadyRunningError",
    "SwarmMember",
    "SwarmMembership",
    "SwarmMembershipError",
    "SwarmNotStartedError",
]
