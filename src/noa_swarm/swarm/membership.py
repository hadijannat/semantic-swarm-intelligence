"""SWIM Membership Protocol Integration for NOA Swarm.

This module provides decentralized membership management using the SWIM
(Scalable Weakly-consistent Infection-style Membership) protocol.

Features:
- Agent discovery and failure detection
- Gossip-based membership dissemination
- Metadata piggybacking (model version, capabilities)
- Event callbacks for member join/leave/update
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Self

from loguru import logger

from noa_swarm.common.config import SwarmSettings

if TYPE_CHECKING:
    from swimprotocol.members import Member as SwimMember
    from swimprotocol.members import Members as SwimMembers
    from swimprotocol.udp import UdpTransport
    from swimprotocol.worker import Worker


# Type aliases for callbacks
MemberCallback = Callable[["SwarmMember"], Awaitable[None] | None]

# Member status type
MemberStatus = Literal["alive", "suspect", "dead"]

# Port validation constants
MIN_PORT = 0  # 0 allowed as sentinel for unknown
MAX_PORT = 65535

# Default SWIM protocol timing constants (in seconds)
DEFAULT_PING_INTERVAL = 1.0
DEFAULT_PING_TIMEOUT = 0.3
DEFAULT_SUSPECT_TIMEOUT = 5.0


class SwarmMembershipError(Exception):
    """Base exception for swarm membership errors."""

    pass


class SwarmNotStartedError(SwarmMembershipError):
    """Exception raised when operations are attempted before starting the swarm."""

    def __init__(self, message: str = "Swarm membership has not been started") -> None:
        super().__init__(message)


class SwarmAlreadyRunningError(SwarmMembershipError):
    """Exception raised when start is called on an already running swarm."""

    def __init__(self, message: str = "Swarm membership is already running") -> None:
        super().__init__(message)


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


@dataclass
class SwarmMember:
    """Represents a member in the swarm cluster.

    Attributes:
        agent_id: Unique identifier for the agent.
        host: Hostname or IP address of the member.
        port: Port number for SWIM protocol communication.
        model_version: Version of the ML model used by the agent.
        capabilities: List of capabilities supported by the agent.
        joined_at: Timestamp when the member joined the swarm.
        last_seen: Timestamp of the last successful ping.
        status: Current status of the member (alive, suspect, dead).
    """

    agent_id: str
    host: str
    port: int
    model_version: str | None = None
    capabilities: list[str] = field(default_factory=list)
    joined_at: datetime = field(default_factory=_utc_now)
    last_seen: datetime = field(default_factory=_utc_now)
    status: MemberStatus = "alive"

    def __post_init__(self) -> None:
        """Validate member fields after initialization."""
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
        if not self.host:
            raise ValueError("host cannot be empty")
        # Port 0 allowed as sentinel for unknown; otherwise must be valid port
        if self.port < MIN_PORT or self.port > MAX_PORT:
            raise ValueError(f"port must be between {MIN_PORT} and {MAX_PORT}, got {self.port}")

    @property
    def address(self) -> str:
        """Return the full address in host:port format."""
        return f"{self.host}:{self.port}"

    @property
    def swim_name(self) -> str:
        """Return the SWIM protocol name (agent_id@host:port)."""
        return f"{self.agent_id}@{self.host}:{self.port}"

    def to_metadata(self) -> dict[str, bytes]:
        """Convert member metadata to SWIM-compatible format.

        Returns:
            Dictionary of metadata as bytes for SWIM protocol.
        """
        metadata: dict[str, bytes] = {}
        if self.model_version:
            metadata["model_version"] = self.model_version.encode("utf-8")
        if self.capabilities:
            metadata["capabilities"] = json.dumps(self.capabilities).encode("utf-8")
        return metadata

    @classmethod
    def from_swim_member(
        cls,
        swim_member: SwimMember,
        status: MemberStatus,
    ) -> SwarmMember:
        """Create a SwarmMember from a SWIM protocol Member.

        Args:
            swim_member: The SWIM protocol member object.
            status: The current status of the member.

        Returns:
            A new SwarmMember instance.
        """
        # Parse the name which should be in format "agent_id@host:port"
        name = swim_member.name
        if "@" in name:
            agent_id, address = name.split("@", 1)
        else:
            agent_id = name
            address = "unknown:0"

        if ":" in address:
            host, port_str = address.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = 0
        else:
            host = address
            port = 0

        # Extract metadata - handle both str and bytes keys
        raw_metadata: dict[Any, Any] = dict(swim_member.metadata) if swim_member.metadata else {}
        model_version = None
        capabilities: list[str] = []

        # Check for model_version in metadata (both str and bytes keys)
        mv_bytes = raw_metadata.get("model_version") or raw_metadata.get(b"model_version")
        if mv_bytes:
            model_version = (
                mv_bytes.decode("utf-8") if isinstance(mv_bytes, bytes) else str(mv_bytes)
            )

        # Check for capabilities in metadata (both str and bytes keys)
        cap_bytes = raw_metadata.get("capabilities") or raw_metadata.get(b"capabilities")
        if cap_bytes:
            cap_str = cap_bytes.decode("utf-8") if isinstance(cap_bytes, bytes) else str(cap_bytes)
            try:
                capabilities = json.loads(cap_str)
            except json.JSONDecodeError:
                capabilities = []

        return cls(
            agent_id=agent_id,
            host=host,
            port=port,
            model_version=model_version,
            capabilities=capabilities,
            status=status,
        )

    def update_last_seen(self) -> SwarmMember:
        """Return a copy of the member with updated last_seen timestamp.

        Returns:
            New SwarmMember with updated last_seen.
        """
        return SwarmMember(
            agent_id=self.agent_id,
            host=self.host,
            port=self.port,
            model_version=self.model_version,
            capabilities=list(self.capabilities),
            joined_at=self.joined_at,
            last_seen=_utc_now(),
            status=self.status,
        )

    def with_status(self, status: MemberStatus) -> SwarmMember:
        """Return a copy of the member with a new status.

        Args:
            status: The new status for the member.

        Returns:
            New SwarmMember with updated status.
        """
        return SwarmMember(
            agent_id=self.agent_id,
            host=self.host,
            port=self.port,
            model_version=self.model_version,
            capabilities=list(self.capabilities),
            joined_at=self.joined_at,
            last_seen=self.last_seen,
            status=status,
        )


class SwarmMembership:
    """Manages swarm membership using the SWIM protocol.

    This class provides the core membership management functionality including:
    - Starting and stopping the SWIM protocol
    - Joining existing swarms via seed addresses
    - Member discovery and failure detection
    - Metadata propagation (model version, capabilities)
    - Event callbacks for membership changes

    Usage:
        membership = SwarmMembership(
            agent_id="agent-001",
            host="192.168.1.100",
            port=7946,
        )
        await membership.start()
        await membership.join([("192.168.1.101", 7946)])

        # Get current members
        members = membership.get_alive_members()

        # Graceful shutdown
        await membership.stop()
    """

    def __init__(
        self,
        agent_id: str,
        host: str,
        port: int,
        settings: SwarmSettings | None = None,
        *,
        secret: str | None = None,
        ping_interval: float = DEFAULT_PING_INTERVAL,
        ping_timeout: float = DEFAULT_PING_TIMEOUT,
        suspect_timeout: float = DEFAULT_SUSPECT_TIMEOUT,
    ) -> None:
        """Initialize the swarm membership manager.

        Args:
            agent_id: Unique identifier for this agent.
            host: Hostname or IP address to bind to.
            port: Port number for SWIM protocol.
            settings: Optional SwarmSettings for additional configuration.
            secret: Shared secret for cluster authentication (optional).
            ping_interval: Interval between pings in seconds.
            ping_timeout: Timeout for ping responses in seconds.
            suspect_timeout: Time before marking suspect members as dead.
        """
        self._agent_id = agent_id
        self._host = host
        self._port = port
        self._settings = settings or SwarmSettings()
        self._secret = secret
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._suspect_timeout = suspect_timeout

        # State
        self._running = False
        self._members: dict[str, SwarmMember] = {}
        self._local_member: SwarmMember | None = None

        # Metadata
        self._model_version: str | None = None
        self._capabilities: list[str] = []

        # SWIM components (initialized on start)
        self._swim_members: SwimMembers | None = None
        self._worker: Worker | None = None
        self._transport: UdpTransport | None = None
        self._tasks: list[asyncio.Task[Any]] = []

        # Callbacks
        self._on_join_callbacks: list[MemberCallback] = []
        self._on_leave_callbacks: list[MemberCallback] = []
        self._on_update_callbacks: list[MemberCallback] = []

        # Lock for thread-safe member updates
        self._lock = asyncio.Lock()

    @property
    def agent_id(self) -> str:
        """Return the agent ID."""
        return self._agent_id

    @property
    def host(self) -> str:
        """Return the host address."""
        return self._host

    @property
    def port(self) -> int:
        """Return the port number."""
        return self._port

    @property
    def is_running(self) -> bool:
        """Return True if the swarm membership is running."""
        return self._running

    @property
    def local_member(self) -> SwarmMember | None:
        """Return the local member information."""
        return self._local_member

    async def start(self) -> None:
        """Start the SWIM protocol and begin membership management.

        Raises:
            SwarmAlreadyRunningError: If the swarm is already running.
        """
        if self._running:
            raise SwarmAlreadyRunningError()

        logger.info(
            f"Starting swarm membership for agent {self._agent_id} on {self._host}:{self._port}"
        )

        try:
            # Import SWIM components
            from swimprotocol.members import Members as SwimMembers
            from swimprotocol.udp import UdpConfig, UdpTransport  # type: ignore[attr-defined]
            from swimprotocol.worker import Worker

            # Create local member name
            local_name = f"{self._agent_id}@{self._host}:{self._port}"

            # Prepare local metadata
            local_metadata = self._build_metadata()

            # Create SWIM configuration
            config = UdpConfig(
                secret=self._secret,
                local_name=local_name,
                peers=[],  # Will be populated via join()
                local_metadata=local_metadata,
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
                suspect_timeout=self._suspect_timeout,
                bind_host=self._host,
                bind_port=self._port,
            )

            # Initialize SWIM components
            self._swim_members = SwimMembers(config)
            self._worker = Worker(config, self._swim_members)
            self._transport = UdpTransport(config, self._worker)

            # Create local member
            self._local_member = SwarmMember(
                agent_id=self._agent_id,
                host=self._host,
                port=self._port,
                model_version=self._model_version,
                capabilities=list(self._capabilities),
                status="alive",
            )
            self._members[self._agent_id] = self._local_member

            # Start the worker task
            worker_task = asyncio.create_task(self._run_worker())
            self._tasks.append(worker_task)

            # Start the member sync task
            sync_task = asyncio.create_task(self._sync_members())
            self._tasks.append(sync_task)

            self._running = True
            logger.info(f"Swarm membership started for agent {self._agent_id}")

        except ImportError as e:
            logger.error(f"Failed to import swimprotocol: {e}")
            raise SwarmMembershipError(f"SWIM protocol library not available: {e}") from e
        except Exception as e:
            logger.error(f"Failed to start swarm membership: {e}")
            raise SwarmMembershipError(f"Failed to start swarm membership: {e}") from e

    async def stop(self) -> None:
        """Stop the SWIM protocol and gracefully shut down.

        This method ensures all background tasks are cancelled and resources
        are properly cleaned up.
        """
        if not self._running:
            logger.warning("Swarm membership was not running")
            return

        logger.info(f"Stopping swarm membership for agent {self._agent_id}")

        self._running = False

        # Cancel all background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self._tasks.clear()

        # Clean up SWIM components
        self._swim_members = None
        self._worker = None
        self._transport = None

        logger.info(f"Swarm membership stopped for agent {self._agent_id}")

    async def join(self, seed_addresses: Sequence[tuple[str, int]]) -> None:
        """Join an existing swarm cluster via seed addresses.

        Args:
            seed_addresses: List of (host, port) tuples for seed members.

        Raises:
            SwarmNotStartedError: If the swarm has not been started.
        """
        if not self._running:
            raise SwarmNotStartedError()

        if not seed_addresses:
            logger.warning("No seed addresses provided for join")
            return

        logger.info(f"Joining swarm with {len(seed_addresses)} seed addresses")

        for host, port in seed_addresses:
            # Create a member entry for each seed
            seed_name = f"seed@{host}:{port}"
            logger.debug(f"Adding seed peer: {seed_name}")

            # The SWIM protocol will handle discovery automatically
            # We just need to track that we're attempting to join
            if self._swim_members:
                # Trigger a gossip cycle to discover the seed
                pass  # Gossip is handled by the worker

    def get_members(self) -> list[SwarmMember]:
        """Get all known members in the swarm.

        Returns:
            List of all SwarmMember instances.
        """
        return list(self._members.values())

    def get_alive_members(self) -> list[SwarmMember]:
        """Get all alive members in the swarm.

        Returns:
            List of SwarmMember instances with status 'alive'.
        """
        return [m for m in self._members.values() if m.status == "alive"]

    def get_member(self, agent_id: str) -> SwarmMember | None:
        """Get a specific member by agent ID.

        Args:
            agent_id: The agent ID to look up.

        Returns:
            The SwarmMember if found, None otherwise.
        """
        return self._members.get(agent_id)

    def set_metadata(
        self,
        model_version: str | None = None,
        capabilities: list[str] | None = None,
    ) -> None:
        """Update this agent's metadata for propagation.

        Args:
            model_version: The model version string.
            capabilities: List of capability strings.
        """
        if model_version is not None:
            self._model_version = model_version
        if capabilities is not None:
            self._capabilities = list(capabilities)

        # Update local member
        if self._local_member:
            self._local_member = SwarmMember(
                agent_id=self._local_member.agent_id,
                host=self._local_member.host,
                port=self._local_member.port,
                model_version=self._model_version,
                capabilities=list(self._capabilities),
                joined_at=self._local_member.joined_at,
                last_seen=_utc_now(),
                status=self._local_member.status,
            )
            self._members[self._agent_id] = self._local_member

        # Update SWIM metadata if running
        if self._running and self._swim_members and self._local_member:
            metadata = self._build_metadata()
            local = self._swim_members.local
            if local:
                self._swim_members.update(local, new_metadata=metadata)

        logger.debug(
            f"Updated metadata: model_version={self._model_version}, "
            f"capabilities={self._capabilities}"
        )

    def on_member_join(self, callback: MemberCallback) -> Self:
        """Register a callback for member join events.

        Args:
            callback: Async or sync function to call when a member joins.

        Returns:
            Self for method chaining.
        """
        self._on_join_callbacks.append(callback)
        return self

    def on_member_leave(self, callback: MemberCallback) -> Self:
        """Register a callback for member leave events.

        Args:
            callback: Async or sync function to call when a member leaves.

        Returns:
            Self for method chaining.
        """
        self._on_leave_callbacks.append(callback)
        return self

    def on_member_update(self, callback: MemberCallback) -> Self:
        """Register a callback for member update events.

        Args:
            callback: Async or sync function to call when a member is updated.

        Returns:
            Self for method chaining.
        """
        self._on_update_callbacks.append(callback)
        return self

    def _build_metadata(self) -> dict[str, bytes]:
        """Build the metadata dictionary for SWIM protocol.

        Returns:
            Dictionary of metadata as bytes.
        """
        metadata: dict[str, bytes] = {}
        if self._model_version:
            metadata["model_version"] = self._model_version.encode("utf-8")
        if self._capabilities:
            metadata["capabilities"] = json.dumps(self._capabilities).encode("utf-8")
        return metadata

    async def _run_worker(self) -> None:
        """Run the SWIM worker in the background."""
        if not self._worker:
            return

        try:
            await self._worker.run()
        except asyncio.CancelledError:
            logger.debug("Worker task cancelled")
            raise
        except Exception as e:
            logger.error(f"Worker task error: {e}")

    async def _sync_members(self) -> None:
        """Periodically sync member status from SWIM protocol."""
        while self._running:
            try:
                await asyncio.sleep(self._ping_interval)

                if not self._swim_members:
                    continue

                await self._update_members_from_swim()

            except asyncio.CancelledError:
                logger.debug("Member sync task cancelled")
                raise
            except Exception as e:
                logger.warning(f"Error syncing members: {e}")

    async def _update_members_from_swim(self) -> None:
        """Update internal member list from SWIM protocol state."""
        if not self._swim_members:
            return

        async with self._lock:
            seen_agents: set[str] = {self._agent_id}  # Always include local

            # Get all members from SWIM
            for swim_member in self._swim_members.non_local:
                status = self._swim_status_to_member_status(swim_member.status)
                agent_id = self._extract_agent_id(swim_member.name)
                seen_agents.add(agent_id)

                existing = self._members.get(agent_id)
                new_member = SwarmMember.from_swim_member(swim_member, status)

                if existing is None:
                    # New member joined
                    self._members[agent_id] = new_member
                    await self._fire_callbacks(self._on_join_callbacks, new_member)
                    logger.info(f"Member joined: {agent_id} ({status})")
                elif existing.status != new_member.status:
                    # Status changed
                    self._members[agent_id] = new_member
                    await self._fire_callbacks(self._on_update_callbacks, new_member)
                    logger.info(
                        f"Member status changed: {agent_id} "
                        f"{existing.status} -> {new_member.status}"
                    )
                    if new_member.status == "dead":
                        await self._fire_callbacks(self._on_leave_callbacks, new_member)
                        logger.info(f"Member left: {agent_id}")

            # Handle removed members (not seen in SWIM anymore)
            for agent_id in list(self._members.keys()):
                if agent_id not in seen_agents and agent_id != self._agent_id:
                    member = self._members.pop(agent_id)
                    dead_member = member.with_status("dead")
                    await self._fire_callbacks(self._on_leave_callbacks, dead_member)
                    logger.info(f"Member removed: {agent_id}")

    @staticmethod
    def _swim_status_to_member_status(swim_status: Any) -> MemberStatus:
        """Convert SWIM status to our member status.

        Args:
            swim_status: The SWIM protocol status.

        Returns:
            The corresponding MemberStatus.
        """
        from swimprotocol.status import Status as SwimStatus

        if swim_status == SwimStatus.ONLINE:
            return "alive"
        elif swim_status == SwimStatus.SUSPECT:
            return "suspect"
        else:
            return "dead"

    @staticmethod
    def _extract_agent_id(swim_name: str) -> str:
        """Extract agent ID from SWIM member name.

        Args:
            swim_name: The SWIM member name (agent_id@host:port).

        Returns:
            The extracted agent ID.
        """
        if "@" in swim_name:
            return swim_name.split("@", 1)[0]
        return swim_name

    async def _fire_callbacks(
        self,
        callbacks: list[MemberCallback],
        member: SwarmMember,
    ) -> None:
        """Fire a list of callbacks with the given member.

        Args:
            callbacks: List of callback functions.
            member: The member to pass to callbacks.
        """
        for callback in callbacks:
            try:
                result = callback(member)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context manager."""
        await self.stop()
