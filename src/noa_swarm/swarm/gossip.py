"""Hypothesis Gossip Protocol via MQTT for NOA Semantic Swarm Mapper.

This module provides gossip-based communication for swarm agents to:
- Broadcast hypotheses for tag mappings
- Share votes during consensus
- Distribute consensus state via retained messages

Topic Structure:
- noa/agents/{agent_id}/status - Agent status/heartbeat
- noa/agents/{agent_id}/hypothesis - Agent hypothesis broadcasts
- noa/tags/{tag_id}/candidates - Tag candidates from all agents
- noa/tags/{tag_id}/consensus - Retained consensus state
- noa/system/config - System-wide configuration
- noa/system/health - System health checks
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Self

from loguru import logger

from noa_swarm.common.schemas import ConsensusRecord, Hypothesis, Vote
from noa_swarm.connectors.mqtt import MQTTClient, MQTTClientError

# ============================================================================
# TOPIC STRUCTURE CONSTANTS
# ============================================================================

TOPIC_AGENTS_PREFIX = "noa/agents/"  # {agent_id}/status, {agent_id}/hypothesis
TOPIC_TAGS_PREFIX = "noa/tags/"  # {tag_id}/candidates, {tag_id}/consensus
TOPIC_SYSTEM_PREFIX = "noa/system/"  # config, health

# Wildcard subscriptions
TOPIC_ALL_AGENTS = "noa/agents/+/+"
TOPIC_ALL_TAGS = "noa/tags/+/+"
TOPIC_ALL_SYSTEM = "noa/system/+"


def agent_status_topic(agent_id: str) -> str:
    """Get the status topic for an agent."""
    return f"{TOPIC_AGENTS_PREFIX}{agent_id}/status"


def agent_hypothesis_topic(agent_id: str) -> str:
    """Get the hypothesis topic for an agent."""
    return f"{TOPIC_AGENTS_PREFIX}{agent_id}/hypothesis"


def tag_candidates_topic(tag_id: str) -> str:
    """Get the candidates topic for a tag."""
    # Sanitize tag_id for MQTT topic (replace special chars)
    safe_tag_id = _sanitize_topic_segment(tag_id)
    return f"{TOPIC_TAGS_PREFIX}{safe_tag_id}/candidates"


def tag_consensus_topic(tag_id: str) -> str:
    """Get the consensus topic for a tag."""
    safe_tag_id = _sanitize_topic_segment(tag_id)
    return f"{TOPIC_TAGS_PREFIX}{safe_tag_id}/consensus"


def _sanitize_topic_segment(segment: str) -> str:
    """Sanitize a string for use in MQTT topic.

    MQTT topics cannot contain: +, #, null character
    Also replace common problematic characters.

    Args:
        segment: The string to sanitize.

    Returns:
        Sanitized string safe for MQTT topics.
    """
    # Replace problematic characters with underscores
    replacements = {
        "+": "_plus_",
        "#": "_hash_",
        "/": "_slash_",
        "|": "_pipe_",
        "\x00": "",
    }
    result = segment
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    return result


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


# ============================================================================
# GOSSIP MESSAGE
# ============================================================================

GossipMessageType = Literal["hypothesis", "vote", "consensus", "heartbeat"]


@dataclass
class GossipMessage:
    """A message in the gossip protocol.

    Attributes:
        message_type: Type of gossip message.
        agent_id: ID of the sending agent.
        payload: Message-specific payload data.
        timestamp: When the message was created.
        correlation_id: Optional correlation ID for tracking.
    """

    message_type: GossipMessageType
    agent_id: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=_utc_now)
    correlation_id: str | None = None

    def to_json(self) -> str:
        """Serialize the message to JSON string.

        Returns:
            JSON string representation.
        """
        data = {
            "message_type": self.message_type,
            "agent_id": self.agent_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
        }
        return json.dumps(data, default=str)

    def to_bytes(self) -> bytes:
        """Serialize the message to bytes.

        Returns:
            UTF-8 encoded JSON bytes.
        """
        return self.to_json().encode("utf-8")

    @classmethod
    def from_json(cls, json_str: str) -> GossipMessage:
        """Deserialize a message from JSON string.

        Args:
            json_str: JSON string to parse.

        Returns:
            GossipMessage instance.

        Raises:
            ValueError: If JSON is invalid or missing required fields.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        required_fields = ["message_type", "agent_id", "payload", "timestamp"]
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Missing required field: {field_name}")

        # Parse timestamp
        timestamp_str = data["timestamp"]
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str)
        else:
            timestamp = _utc_now()

        return cls(
            message_type=data["message_type"],
            agent_id=data["agent_id"],
            payload=data["payload"],
            timestamp=timestamp,
            correlation_id=data.get("correlation_id"),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> GossipMessage:
        """Deserialize a message from bytes.

        Args:
            data: UTF-8 encoded JSON bytes.

        Returns:
            GossipMessage instance.
        """
        return cls.from_json(data.decode("utf-8"))


# ============================================================================
# CALLBACK TYPES
# ============================================================================

HypothesisCallback = Callable[[str, Hypothesis], Awaitable[None] | None]
VoteCallback = Callable[[str, Vote], Awaitable[None] | None]
ConsensusCallback = Callable[[str, ConsensusRecord], Awaitable[None] | None]
HeartbeatCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]


# ============================================================================
# HYPOTHESIS GOSSIP
# ============================================================================


class HypothesisGossipError(Exception):
    """Base exception for hypothesis gossip errors."""

    pass


class HypothesisGossip:
    """Hypothesis gossip protocol implementation via MQTT.

    This class manages the pub/sub communication for sharing hypotheses,
    votes, and consensus state between swarm agents.

    Usage:
        gossip = HypothesisGossip(mqtt_client, "agent-001")
        gossip.on_hypothesis(handle_hypothesis)
        gossip.on_vote(handle_vote)

        await gossip.start()

        # Broadcast a hypothesis
        await gossip.broadcast_hypothesis(tag_id, hypothesis)

        # Broadcast consensus (retained)
        await gossip.broadcast_consensus(tag_id, consensus)

        await gossip.stop()
    """

    def __init__(
        self,
        mqtt_client: MQTTClient,
        agent_id: str,
        *,
        default_qos: int = 1,
    ) -> None:
        """Initialize the hypothesis gossip handler.

        Args:
            mqtt_client: The MQTT client to use for communication.
            agent_id: This agent's unique identifier.
            default_qos: Default QoS level for messages.
        """
        self._mqtt = mqtt_client
        self._agent_id = agent_id
        self._default_qos = default_qos
        self._running = False

        # Callbacks
        self._hypothesis_callbacks: list[HypothesisCallback] = []
        self._vote_callbacks: list[VoteCallback] = []
        self._consensus_callbacks: list[ConsensusCallback] = []
        self._heartbeat_callbacks: list[HeartbeatCallback] = []

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    @property
    def agent_id(self) -> str:
        """Return the agent ID."""
        return self._agent_id

    @property
    def is_running(self) -> bool:
        """Return True if gossip is running."""
        return self._running

    async def start(self) -> None:
        """Start the gossip protocol by subscribing to relevant topics.

        Raises:
            HypothesisGossipError: If starting fails.
        """
        if self._running:
            logger.warning(f"HypothesisGossip for {self._agent_id} already running")
            return

        try:
            # Register message handler
            self._mqtt.on_message(self._handle_message)

            # Subscribe to relevant topics
            await self._mqtt.subscribe(TOPIC_ALL_AGENTS, qos=self._default_qos)
            await self._mqtt.subscribe(TOPIC_ALL_TAGS, qos=self._default_qos)
            await self._mqtt.subscribe(TOPIC_ALL_SYSTEM, qos=self._default_qos)

            self._running = True
            logger.info(f"HypothesisGossip started for agent {self._agent_id}")

            # Send initial heartbeat
            await self.send_heartbeat()

        except MQTTClientError as e:
            raise HypothesisGossipError(f"Failed to start gossip: {e}") from e
        except Exception as e:
            raise HypothesisGossipError(f"Failed to start gossip: {e}") from e

    async def stop(self) -> None:
        """Stop the gossip protocol and unsubscribe from topics."""
        if not self._running:
            logger.warning(f"HypothesisGossip for {self._agent_id} not running")
            return

        try:
            # Unsubscribe from topics
            await self._mqtt.unsubscribe(TOPIC_ALL_AGENTS)
            await self._mqtt.unsubscribe(TOPIC_ALL_TAGS)
            await self._mqtt.unsubscribe(TOPIC_ALL_SYSTEM)

            self._running = False
            logger.info(f"HypothesisGossip stopped for agent {self._agent_id}")

        except Exception as e:
            logger.warning(f"Error stopping gossip: {e}")
            self._running = False

    async def broadcast_hypothesis(
        self,
        tag_id: str,
        hypothesis: Hypothesis,
        correlation_id: str | None = None,
    ) -> None:
        """Broadcast a hypothesis for a tag.

        Args:
            tag_id: The tag ID the hypothesis is for.
            hypothesis: The hypothesis to broadcast.
            correlation_id: Optional correlation ID for tracking.

        Raises:
            HypothesisGossipError: If broadcasting fails.
        """
        if not self._running:
            raise HypothesisGossipError("Gossip not running")

        try:
            # Create gossip message
            message = GossipMessage(
                message_type="hypothesis",
                agent_id=self._agent_id,
                payload={
                    "tag_id": tag_id,
                    "hypothesis": hypothesis.model_dump(mode="json"),
                },
                correlation_id=correlation_id or str(uuid.uuid4()),
            )

            # Publish to agent's hypothesis topic
            topic = agent_hypothesis_topic(self._agent_id)
            await self._mqtt.publish(
                topic,
                message.to_bytes(),
                qos=self._default_qos,
                retain=False,
            )

            # Also publish to tag candidates topic for aggregation
            tag_topic = tag_candidates_topic(tag_id)
            await self._mqtt.publish(
                tag_topic,
                message.to_bytes(),
                qos=self._default_qos,
                retain=False,
            )

            logger.debug(f"Broadcast hypothesis for tag {tag_id}")

        except MQTTClientError as e:
            raise HypothesisGossipError(f"Failed to broadcast hypothesis: {e}") from e

    async def broadcast_vote(
        self,
        tag_id: str,
        vote: Vote,
        correlation_id: str | None = None,
    ) -> None:
        """Broadcast a vote for a tag consensus.

        Args:
            tag_id: The tag ID the vote is for.
            vote: The vote to broadcast.
            correlation_id: Optional correlation ID for tracking.

        Raises:
            HypothesisGossipError: If broadcasting fails.
        """
        if not self._running:
            raise HypothesisGossipError("Gossip not running")

        try:
            message = GossipMessage(
                message_type="vote",
                agent_id=self._agent_id,
                payload={
                    "tag_id": tag_id,
                    "vote": vote.model_dump(mode="json"),
                },
                correlation_id=correlation_id or str(uuid.uuid4()),
            )

            # Publish to tag candidates topic
            topic = tag_candidates_topic(tag_id)
            await self._mqtt.publish(
                topic,
                message.to_bytes(),
                qos=self._default_qos,
                retain=False,
            )

            logger.debug(f"Broadcast vote for tag {tag_id}")

        except MQTTClientError as e:
            raise HypothesisGossipError(f"Failed to broadcast vote: {e}") from e

    async def broadcast_consensus(
        self,
        tag_id: str,
        consensus: ConsensusRecord,
        correlation_id: str | None = None,
    ) -> None:
        """Broadcast consensus state for a tag (with retain=True).

        The retained message ensures new agents receive the current
        consensus state when they subscribe.

        Args:
            tag_id: The tag ID the consensus is for.
            consensus: The consensus record to broadcast.
            correlation_id: Optional correlation ID for tracking.

        Raises:
            HypothesisGossipError: If broadcasting fails.
        """
        if not self._running:
            raise HypothesisGossipError("Gossip not running")

        try:
            message = GossipMessage(
                message_type="consensus",
                agent_id=self._agent_id,
                payload={
                    "tag_id": tag_id,
                    "consensus": consensus.model_dump(mode="json"),
                },
                correlation_id=correlation_id or str(uuid.uuid4()),
            )

            # Publish to consensus topic with retain=True
            topic = tag_consensus_topic(tag_id)
            await self._mqtt.publish(
                topic,
                message.to_bytes(),
                qos=self._default_qos,
                retain=True,  # Important: Retain for late-joining agents
            )

            logger.debug(f"Broadcast consensus for tag {tag_id}")

        except MQTTClientError as e:
            raise HypothesisGossipError(f"Failed to broadcast consensus: {e}") from e

    async def send_heartbeat(self, extra_info: dict[str, Any] | None = None) -> None:
        """Send a heartbeat message.

        Args:
            extra_info: Optional additional information to include.
        """
        if not self._running:
            return

        try:
            payload: dict[str, Any] = {
                "status": "alive",
                "timestamp": _utc_now().isoformat(),
            }
            if extra_info:
                payload.update(extra_info)

            message = GossipMessage(
                message_type="heartbeat",
                agent_id=self._agent_id,
                payload=payload,
            )

            topic = agent_status_topic(self._agent_id)
            await self._mqtt.publish(
                topic,
                message.to_bytes(),
                qos=0,  # Heartbeats use QoS 0 for efficiency
                retain=True,  # Retain for agent discovery
            )

            logger.debug(f"Sent heartbeat for agent {self._agent_id}")

        except Exception as e:
            logger.warning(f"Failed to send heartbeat: {e}")

    # =========================================================================
    # CALLBACK REGISTRATION
    # =========================================================================

    def on_hypothesis(self, callback: HypothesisCallback) -> Self:
        """Register a callback for received hypotheses.

        Args:
            callback: Function to call when a hypothesis is received.
                     Takes (tag_id, hypothesis) parameters.

        Returns:
            Self for method chaining.
        """
        self._hypothesis_callbacks.append(callback)
        return self

    def on_vote(self, callback: VoteCallback) -> Self:
        """Register a callback for received votes.

        Args:
            callback: Function to call when a vote is received.
                     Takes (tag_id, vote) parameters.

        Returns:
            Self for method chaining.
        """
        self._vote_callbacks.append(callback)
        return self

    def on_consensus(self, callback: ConsensusCallback) -> Self:
        """Register a callback for received consensus updates.

        Args:
            callback: Function to call when consensus is received.
                     Takes (tag_id, consensus) parameters.

        Returns:
            Self for method chaining.
        """
        self._consensus_callbacks.append(callback)
        return self

    def on_heartbeat(self, callback: HeartbeatCallback) -> Self:
        """Register a callback for received heartbeats.

        Args:
            callback: Function to call when a heartbeat is received.
                     Takes (agent_id, payload) parameters.

        Returns:
            Self for method chaining.
        """
        self._heartbeat_callbacks.append(callback)
        return self

    # =========================================================================
    # MESSAGE HANDLING
    # =========================================================================

    async def _handle_message(self, topic: str, payload: bytes) -> None:
        """Handle incoming MQTT messages.

        Args:
            topic: The topic the message was received on.
            payload: The message payload.
        """
        try:
            # Parse the gossip message
            message = GossipMessage.from_bytes(payload)

            # Ignore our own messages
            if message.agent_id == self._agent_id:
                return

            # Route based on message type
            if message.message_type == "hypothesis":
                await self._handle_hypothesis(message)
            elif message.message_type == "vote":
                await self._handle_vote(message)
            elif message.message_type == "consensus":
                await self._handle_consensus(message)
            elif message.message_type == "heartbeat":
                await self._handle_heartbeat(message)

        except Exception as e:
            logger.warning(f"Error handling message from topic {topic}: {e}")

    async def _handle_hypothesis(self, message: GossipMessage) -> None:
        """Handle a hypothesis message."""
        try:
            tag_id = message.payload.get("tag_id")
            hypothesis_data = message.payload.get("hypothesis")

            if not tag_id or not hypothesis_data:
                logger.warning("Invalid hypothesis message: missing tag_id or hypothesis")
                return

            hypothesis = Hypothesis.model_validate(hypothesis_data)

            for callback in self._hypothesis_callbacks:
                await self._fire_callback(callback, tag_id, hypothesis)

            logger.debug(f"Received hypothesis for tag {tag_id} from {message.agent_id}")

        except Exception as e:
            logger.warning(f"Error handling hypothesis: {e}")

    async def _handle_vote(self, message: GossipMessage) -> None:
        """Handle a vote message."""
        try:
            tag_id = message.payload.get("tag_id")
            vote_data = message.payload.get("vote")

            if not tag_id or not vote_data:
                logger.warning("Invalid vote message: missing tag_id or vote")
                return

            vote = Vote.model_validate(vote_data)

            for callback in self._vote_callbacks:
                await self._fire_callback(callback, tag_id, vote)

            logger.debug(f"Received vote for tag {tag_id} from {message.agent_id}")

        except Exception as e:
            logger.warning(f"Error handling vote: {e}")

    async def _handle_consensus(self, message: GossipMessage) -> None:
        """Handle a consensus message."""
        try:
            tag_id = message.payload.get("tag_id")
            consensus_data = message.payload.get("consensus")

            if not tag_id or not consensus_data:
                logger.warning("Invalid consensus message: missing tag_id or consensus")
                return

            consensus = ConsensusRecord.model_validate(consensus_data)

            for callback in self._consensus_callbacks:
                await self._fire_callback(callback, tag_id, consensus)

            logger.debug(f"Received consensus for tag {tag_id} from {message.agent_id}")

        except Exception as e:
            logger.warning(f"Error handling consensus: {e}")

    async def _handle_heartbeat(self, message: GossipMessage) -> None:
        """Handle a heartbeat message."""
        try:
            for callback in self._heartbeat_callbacks:
                await self._fire_heartbeat_callback(callback, message.agent_id, message.payload)

            logger.debug(f"Received heartbeat from {message.agent_id}")

        except Exception as e:
            logger.warning(f"Error handling heartbeat: {e}")

    async def _fire_callback(
        self,
        callback: Callable[..., Awaitable[None] | None],
        *args: Any,
    ) -> None:
        """Fire a callback, handling both sync and async."""
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning(f"Callback error: {e}")

    async def _fire_heartbeat_callback(
        self,
        callback: HeartbeatCallback,
        agent_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Fire a heartbeat callback."""
        try:
            result = callback(agent_id, payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning(f"Heartbeat callback error: {e}")
