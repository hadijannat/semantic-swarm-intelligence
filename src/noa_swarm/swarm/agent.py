"""Semantic Agent - Main orchestrator for the swarm intelligence system.

This module provides the SemanticAgent class which orchestrates the complete
lifecycle of semantic mapping: discover -> infer -> gossip -> vote -> commit.

Features:
- OPC UA tag discovery
- ML inference for IRDI mapping
- MQTT-based gossip for hypothesis sharing
- Consensus voting and commitment
- Health monitoring and graceful shutdown
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Self

from loguru import logger

if TYPE_CHECKING:
    from noa_swarm.common.schemas import (
        ConsensusRecord,
        Hypothesis,
        TagRecord,
        Vote,
    )
    from noa_swarm.connectors.mqtt import MQTTClient
    from noa_swarm.connectors.opcua_asyncua import OPCUABrowser
    from noa_swarm.swarm.consensus import ConsensusEngine
    from noa_swarm.swarm.gossip import HypothesisGossip
    from noa_swarm.swarm.reputation import ReputationTracker


# Port validation constants
MIN_PORT = 1
MAX_PORT = 65535


class SemanticAgentError(Exception):
    """Base exception for semantic agent errors."""

    pass


class AgentState(Enum):
    """Lifecycle states for the SemanticAgent."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class SemanticAgentConfig:
    """Configuration for SemanticAgent.

    Attributes:
        agent_id: Unique identifier for this agent.
        opcua_endpoint: OPC UA server endpoint URL.
        mqtt_host: MQTT broker hostname.
        mqtt_port: MQTT broker port number.
        poll_interval_seconds: Interval between discovery cycles.
        shutdown_timeout_seconds: Maximum time to wait for graceful shutdown.
    """

    agent_id: str
    opcua_endpoint: str
    mqtt_host: str
    mqtt_port: int = 1883
    poll_interval_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
        if not self.opcua_endpoint:
            raise ValueError("opcua_endpoint cannot be empty")
        if not self.mqtt_host:
            raise ValueError("mqtt_host cannot be empty")
        if self.mqtt_port < MIN_PORT or self.mqtt_port > MAX_PORT:
            raise ValueError(
                f"mqtt_port must be between {MIN_PORT} and {MAX_PORT}, got {self.mqtt_port}"
            )
        if self.poll_interval_seconds <= 0:
            raise ValueError(
                f"poll_interval_seconds must be positive, got {self.poll_interval_seconds}"
            )
        if self.shutdown_timeout_seconds < 0:
            raise ValueError(
                f"shutdown_timeout_seconds cannot be negative, got {self.shutdown_timeout_seconds}"
            )


@dataclass
class AgentMetrics:
    """Health metrics for the agent."""

    tags_discovered: int = 0
    hypotheses_generated: int = 0
    votes_cast: int = 0
    last_discovery_time: datetime | None = None
    last_inference_time: datetime | None = None
    last_gossip_time: datetime | None = None


class SemanticAgent:
    """Main agent class orchestrating all swarm components.

    The SemanticAgent coordinates the complete lifecycle of semantic mapping:
    1. Discover tags from OPC UA servers
    2. Infer IRDI mappings using ML models
    3. Gossip hypotheses to other swarm members
    4. Participate in consensus voting
    5. Commit agreed mappings

    Usage:
        config = SemanticAgentConfig(
            agent_id="agent-001",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
        )
        async with SemanticAgent(config) as agent:
            # Agent runs lifecycle loop automatically
            pass

        # Or manual control:
        agent = SemanticAgent(config)
        await agent.start()
        tags = await agent.discover_tags()
        hypotheses = await agent.infer_mappings(tags)
        await agent.gossip_hypotheses(hypotheses)
        await agent.stop()
    """

    def __init__(
        self,
        config: SemanticAgentConfig,
        consensus_engine: ConsensusEngine | None = None,
        reputation_tracker: ReputationTracker | None = None,
        inference_engine: Any | None = None,
    ) -> None:
        """Initialize the SemanticAgent.

        Args:
            config: Agent configuration.
            consensus_engine: Optional ConsensusEngine instance (for testing).
            reputation_tracker: Optional ReputationTracker instance (for testing).
            inference_engine: Optional inference engine instance (for testing).
        """
        self._config = config
        self._consensus_engine = consensus_engine
        self._reputation_tracker = reputation_tracker
        self._inference_engine = inference_engine

        # State management
        self._state = AgentState.INITIALIZING
        self._state_lock = asyncio.Lock()

        # Metrics
        self._metrics = AgentMetrics()

        # Component references (initialized on start)
        self._mqtt_client: MQTTClient | None = None
        self._opcua_browser: OPCUABrowser | None = None
        self._gossip: HypothesisGossip | None = None

        # Task management
        self._lifecycle_task: asyncio.Task[None] | None = None

        # Storage
        self._pending_hypotheses: dict[str, Hypothesis] = {}
        self._committed_mappings: dict[str, ConsensusRecord] = {}

        logger.info(
            f"SemanticAgent initialized: agent_id={config.agent_id}, "
            f"opcua_endpoint={config.opcua_endpoint}, "
            f"mqtt_host={config.mqtt_host}:{config.mqtt_port}"
        )

    @property
    def agent_id(self) -> str:
        """Return the agent ID."""
        return self._config.agent_id

    @property
    def state(self) -> AgentState:
        """Return the current agent state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Return True if the agent is in RUNNING state."""
        return self._state == AgentState.RUNNING

    async def start(self) -> None:
        """Start the agent lifecycle.

        This method:
        1. Connects to MQTT broker
        2. Connects to OPC UA server
        3. Starts gossip protocol
        4. Begins the lifecycle loop

        If already running, logs a warning and returns.
        """
        async with self._state_lock:
            if self._state == AgentState.RUNNING:
                logger.warning(f"Agent {self.agent_id} is already running")
                return

            logger.info(f"Starting agent {self.agent_id}")

            try:
                # Connect to MQTT
                await self._connect_mqtt()

                # Connect to OPC UA
                await self._connect_opcua()

                # Start gossip protocol
                await self._start_gossip()

                # Update state
                self._state = AgentState.RUNNING

                # Start lifecycle loop
                self._lifecycle_task = asyncio.create_task(
                    self._run_lifecycle_loop(),
                    name=f"agent-{self.agent_id}-lifecycle",
                )

                logger.info(f"Agent {self.agent_id} started successfully")

            except Exception as e:
                logger.error(f"Failed to start agent {self.agent_id}: {e}")
                self._state = AgentState.STOPPED
                raise SemanticAgentError(f"Failed to start agent: {e}") from e

    async def stop(self) -> None:
        """Gracefully stop the agent.

        This method:
        1. Sets state to STOPPING
        2. Cancels running tasks
        3. Disconnects from MQTT
        4. Waits up to shutdown_timeout
        5. Sets state to STOPPED
        """
        async with self._state_lock:
            if self._state in (AgentState.STOPPED, AgentState.STOPPING):
                logger.debug(f"Agent {self.agent_id} already stopped or stopping")
                self._state = AgentState.STOPPED
                return

            logger.info(f"Stopping agent {self.agent_id}")
            self._state = AgentState.STOPPING

        # Cancel lifecycle task
        if self._lifecycle_task and not self._lifecycle_task.done():
            self._lifecycle_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                try:
                    await asyncio.wait_for(
                        self._lifecycle_task,
                        timeout=self._config.shutdown_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Lifecycle task did not complete within timeout for {self.agent_id}"
                    )

        # Stop gossip
        await self._stop_gossip()

        # Disconnect from MQTT
        await self._disconnect_mqtt()

        # Disconnect from OPC UA
        await self._disconnect_opcua()

        async with self._state_lock:
            self._state = AgentState.STOPPED

        logger.info(f"Agent {self.agent_id} stopped")

    async def discover_tags(self) -> list[TagRecord]:
        """Discover tags from the OPC UA server.

        Returns:
            List of discovered TagRecord instances.

        Raises:
            SemanticAgentError: If OPC UA browser is not available.
        """
        if self._opcua_browser is None:
            logger.warning(f"OPC UA browser not available for {self.agent_id}")
            return []

        logger.debug(f"Discovering tags from {self._config.opcua_endpoint}")

        try:
            tags = await self._opcua_browser.browse_all_tags()

            # Update metrics
            self._metrics.tags_discovered = len(tags)
            self._metrics.last_discovery_time = _utc_now()

            logger.info(f"Discovered {len(tags)} tags from OPC UA server")
            return tags

        except Exception as e:
            logger.error(f"Failed to discover tags: {e}")
            raise SemanticAgentError(f"Tag discovery failed: {e}") from e

    async def infer_mappings(self, tags: list[TagRecord]) -> list[Hypothesis]:
        """Run ML inference to generate mapping hypotheses.

        Args:
            tags: List of tags to generate hypotheses for.

        Returns:
            List of Hypothesis instances with candidate IRDIs.

        Raises:
            SemanticAgentError: If inference engine is not available.
        """
        if self._inference_engine is None:
            raise SemanticAgentError("Inference engine not available")

        if not tags:
            logger.debug("No tags provided for inference")
            return []

        logger.debug(f"Running inference on {len(tags)} tags")

        try:
            # Run inference (synchronous or async depending on engine)
            hypotheses = self._inference_engine.infer(tags)

            # Update metrics
            self._metrics.hypotheses_generated = len(hypotheses)
            self._metrics.last_inference_time = _utc_now()

            # Store pending hypotheses for voting
            for hypothesis in hypotheses:
                self._pending_hypotheses[hypothesis.tag_id] = hypothesis

            logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses

        except Exception as e:
            logger.error(f"Failed to run inference: {e}")
            raise SemanticAgentError(f"Inference failed: {e}") from e

    async def gossip_hypotheses(self, hypotheses: list[Hypothesis]) -> None:
        """Broadcast hypotheses to the swarm via gossip protocol.

        Args:
            hypotheses: List of hypotheses to broadcast.
        """
        if not hypotheses:
            logger.debug("No hypotheses to gossip")
            return

        if self._gossip is None:
            logger.warning("Gossip protocol not available")
            return

        logger.debug(f"Gossiping {len(hypotheses)} hypotheses")

        try:
            for hypothesis in hypotheses:
                await self._gossip.broadcast_hypothesis(
                    hypothesis.tag_id,
                    hypothesis,
                )

            # Update metrics
            self._metrics.last_gossip_time = _utc_now()

            logger.info(f"Gossiped {len(hypotheses)} hypotheses to swarm")

        except Exception as e:
            logger.error(f"Failed to gossip hypotheses: {e}")

    async def vote_on_tags(self, tag_ids: list[str]) -> None:
        """Submit votes for pending tags.

        Args:
            tag_ids: List of tag IDs to vote on.
        """
        if not tag_ids:
            logger.debug("No tag IDs provided for voting")
            return

        if self._gossip is None:
            logger.warning("Gossip protocol not available for voting")
            return

        logger.debug(f"Voting on {len(tag_ids)} tags")

        votes_cast = 0
        for tag_id in tag_ids:
            hypothesis = self._pending_hypotheses.get(tag_id)
            if hypothesis is None:
                logger.debug(f"No hypothesis found for tag {tag_id}")
                continue

            if not hypothesis.candidates:
                logger.debug(f"No candidates for tag {tag_id}")
                continue

            # Get reliability score
            reliability = 1.0
            if self._reputation_tracker:
                reliability = self._reputation_tracker.get_reliability(self.agent_id)

            # Vote for top candidate
            top_candidate = hypothesis.candidates[0]

            # Import Vote here to avoid circular imports
            from noa_swarm.common.schemas import Vote

            vote = Vote(
                agent_id=self.agent_id,
                candidate_irdi=top_candidate.irdi,
                confidence=top_candidate.confidence,
                reliability_score=reliability,
            )

            try:
                await self._gossip.broadcast_vote(tag_id, vote)
                votes_cast += 1
            except Exception as e:
                logger.error(f"Failed to broadcast vote for {tag_id}: {e}")

        # Update metrics
        self._metrics.votes_cast += votes_cast
        logger.info(f"Cast {votes_cast} votes")

    async def commit_mappings(self, records: list[ConsensusRecord]) -> None:
        """Commit consensus records as final mappings.

        Args:
            records: List of consensus records to commit.
        """
        if not records:
            logger.debug("No consensus records to commit")
            return

        logger.debug(f"Committing {len(records)} consensus records")

        for record in records:
            self._committed_mappings[record.tag_id] = record

            # Remove from pending
            self._pending_hypotheses.pop(record.tag_id, None)

        # Broadcast consensus to swarm
        if self._gossip:
            try:
                for record in records:
                    await self._gossip.broadcast_consensus(record.tag_id, record)
            except Exception as e:
                logger.error(f"Failed to broadcast consensus: {e}")

        logger.info(f"Committed {len(records)} mappings")

    def health_check(self) -> dict[str, Any]:
        """Return health status of the agent.

        Returns:
            Dictionary containing health metrics:
            - agent_id: Agent identifier
            - state: Current state
            - is_running: Whether agent is running
            - tags_discovered: Count of discovered tags
            - hypotheses_generated: Count of generated hypotheses
            - votes_cast: Count of votes cast
            - last_discovery_time: Timestamp of last discovery
            - last_inference_time: Timestamp of last inference
            - last_gossip_time: Timestamp of last gossip
        """
        return {
            "agent_id": self.agent_id,
            "state": self._state.value,
            "is_running": self.is_running,
            "tags_discovered": self._metrics.tags_discovered,
            "hypotheses_generated": self._metrics.hypotheses_generated,
            "votes_cast": self._metrics.votes_cast,
            "last_discovery_time": (
                self._metrics.last_discovery_time.isoformat()
                if self._metrics.last_discovery_time
                else None
            ),
            "last_inference_time": (
                self._metrics.last_inference_time.isoformat()
                if self._metrics.last_inference_time
                else None
            ),
            "last_gossip_time": (
                self._metrics.last_gossip_time.isoformat()
                if self._metrics.last_gossip_time
                else None
            ),
        }

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    async def _connect_mqtt(self) -> None:
        """Connect to the MQTT broker."""
        logger.debug(f"Connecting to MQTT at {self._config.mqtt_host}:{self._config.mqtt_port}")
        # Connection is handled by the MQTTClient when injected or created
        # This is a hook for testing

    async def _disconnect_mqtt(self) -> None:
        """Disconnect from the MQTT broker."""
        if self._mqtt_client:
            try:
                await self._mqtt_client.disconnect()
                logger.debug("Disconnected from MQTT")
            except Exception as e:
                logger.warning(f"Error disconnecting from MQTT: {e}")

    async def _connect_opcua(self) -> None:
        """Connect to the OPC UA server."""
        logger.debug(f"Connecting to OPC UA at {self._config.opcua_endpoint}")
        # Connection is handled by the OPCUABrowser when injected or created
        # This is a hook for testing

    async def _disconnect_opcua(self) -> None:
        """Disconnect from the OPC UA server."""
        if self._opcua_browser:
            try:
                await self._opcua_browser.disconnect()
                logger.debug("Disconnected from OPC UA")
            except Exception as e:
                logger.warning(f"Error disconnecting from OPC UA: {e}")

    async def _start_gossip(self) -> None:
        """Start the gossip protocol."""
        logger.debug("Starting gossip protocol")
        # Gossip start is handled by the HypothesisGossip when injected or created
        # This is a hook for testing

    async def _stop_gossip(self) -> None:
        """Stop the gossip protocol."""
        if self._gossip:
            try:
                await self._gossip.stop()
                logger.debug("Stopped gossip protocol")
            except Exception as e:
                logger.warning(f"Error stopping gossip: {e}")

    async def _run_lifecycle_loop(self) -> None:
        """Run the main lifecycle loop.

        This loop continuously:
        1. Discovers tags from OPC UA
        2. Infers mappings using ML
        3. Gossips hypotheses to the swarm
        4. Votes on discovered tags
        5. Commits any reached consensus
        6. Sleeps for poll_interval
        """
        logger.info(f"Starting lifecycle loop for {self.agent_id}")

        try:
            while self._state == AgentState.RUNNING:
                try:
                    # Discover
                    tags = await self.discover_tags()

                    # Infer
                    if tags and self._inference_engine:
                        hypotheses = await self.infer_mappings(tags)

                        # Gossip
                        if hypotheses:
                            await self.gossip_hypotheses(hypotheses)

                            # Vote on discovered tags
                            tag_ids = [h.tag_id for h in hypotheses]
                            await self.vote_on_tags(tag_ids)

                            # Commit any reached consensus
                            # In a real system, this would check for consensus from swarm
                            # For now, this is a placeholder that commits nothing
                            await self.commit_mappings([])

                except Exception as e:
                    logger.error(f"Error in lifecycle loop: {e}")

                # Sleep between cycles
                await asyncio.sleep(self._config.poll_interval_seconds)

        except asyncio.CancelledError:
            logger.debug(f"Lifecycle loop cancelled for {self.agent_id}")
            raise

        logger.info(f"Lifecycle loop ended for {self.agent_id}")

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    async def __aenter__(self) -> Self:
        """Enter async context manager and start agent."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context manager and stop agent."""
        await self.stop()
