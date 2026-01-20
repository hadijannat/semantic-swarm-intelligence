"""Unit tests for hypothesis gossip protocol via MQTT."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from noa_swarm.common.schemas import Candidate, ConsensusRecord, Hypothesis, Vote
from noa_swarm.connectors.mqtt import MQTTClient
from noa_swarm.swarm.gossip import (
    TOPIC_AGENTS_PREFIX,
    TOPIC_ALL_AGENTS,
    TOPIC_ALL_SYSTEM,
    TOPIC_ALL_TAGS,
    TOPIC_SYSTEM_PREFIX,
    TOPIC_TAGS_PREFIX,
    GossipMessage,
    HypothesisGossip,
    HypothesisGossipError,
    _sanitize_topic_segment,
    agent_hypothesis_topic,
    agent_status_topic,
    tag_candidates_topic,
    tag_consensus_topic,
)


class TestTopicConstants:
    """Tests for topic structure constants."""

    def test_topic_prefixes(self) -> None:
        """Test topic prefix constants."""
        assert TOPIC_AGENTS_PREFIX == "noa/agents/"
        assert TOPIC_TAGS_PREFIX == "noa/tags/"
        assert TOPIC_SYSTEM_PREFIX == "noa/system/"

    def test_wildcard_topics(self) -> None:
        """Test wildcard topic patterns."""
        assert TOPIC_ALL_AGENTS == "noa/agents/+/+"
        assert TOPIC_ALL_TAGS == "noa/tags/+/+"
        assert TOPIC_ALL_SYSTEM == "noa/system/+"


class TestTopicHelpers:
    """Tests for topic helper functions."""

    def test_agent_status_topic(self) -> None:
        """Test agent status topic generation."""
        topic = agent_status_topic("agent-001")
        assert topic == "noa/agents/agent-001/status"

    def test_agent_hypothesis_topic(self) -> None:
        """Test agent hypothesis topic generation."""
        topic = agent_hypothesis_topic("agent-002")
        assert topic == "noa/agents/agent-002/hypothesis"

    def test_tag_candidates_topic(self) -> None:
        """Test tag candidates topic generation."""
        topic = tag_candidates_topic("opc.tcp://localhost:4840|ns=2;s=Temperature")
        # Special chars should be sanitized
        assert "+" not in topic
        assert "#" not in topic
        assert topic.startswith(TOPIC_TAGS_PREFIX)
        assert topic.endswith("/candidates")

    def test_tag_consensus_topic(self) -> None:
        """Test tag consensus topic generation."""
        topic = tag_consensus_topic("tag-123")
        assert topic == "noa/tags/tag-123/consensus"

    def test_sanitize_topic_segment(self) -> None:
        """Test topic segment sanitization."""
        # Test problematic characters
        assert "+" not in _sanitize_topic_segment("tag+name")
        assert "#" not in _sanitize_topic_segment("tag#name")
        assert "/" not in _sanitize_topic_segment("tag/name")
        assert "|" not in _sanitize_topic_segment("tag|name")

        # Test replacement strings
        result = _sanitize_topic_segment("a+b#c/d|e")
        assert "_plus_" in result
        assert "_hash_" in result
        assert "_slash_" in result
        assert "_pipe_" in result


class TestGossipMessage:
    """Tests for GossipMessage dataclass."""

    def test_create_message(self) -> None:
        """Test creating a gossip message."""
        msg = GossipMessage(
            message_type="hypothesis",
            agent_id="agent-001",
            payload={"tag_id": "tag-123", "data": "test"},
        )

        assert msg.message_type == "hypothesis"
        assert msg.agent_id == "agent-001"
        assert msg.payload == {"tag_id": "tag-123", "data": "test"}
        assert msg.timestamp is not None
        assert msg.correlation_id is None

    def test_create_message_with_all_fields(self) -> None:
        """Test creating a gossip message with all fields."""
        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        msg = GossipMessage(
            message_type="vote",
            agent_id="agent-002",
            payload={"tag_id": "tag-456"},
            timestamp=ts,
            correlation_id="corr-123",
        )

        assert msg.message_type == "vote"
        assert msg.agent_id == "agent-002"
        assert msg.timestamp == ts
        assert msg.correlation_id == "corr-123"

    def test_to_json(self) -> None:
        """Test serializing message to JSON."""
        msg = GossipMessage(
            message_type="consensus",
            agent_id="agent-001",
            payload={"result": "success"},
            correlation_id="abc-123",
        )

        json_str = msg.to_json()
        data = json.loads(json_str)

        assert data["message_type"] == "consensus"
        assert data["agent_id"] == "agent-001"
        assert data["payload"] == {"result": "success"}
        assert data["correlation_id"] == "abc-123"
        assert "timestamp" in data

    def test_to_bytes(self) -> None:
        """Test serializing message to bytes."""
        msg = GossipMessage(
            message_type="heartbeat",
            agent_id="agent-001",
            payload={"status": "alive"},
        )

        data = msg.to_bytes()
        assert isinstance(data, bytes)

        # Should be valid JSON
        parsed = json.loads(data.decode("utf-8"))
        assert parsed["message_type"] == "heartbeat"

    def test_from_json(self) -> None:
        """Test deserializing message from JSON."""
        json_str = json.dumps(
            {
                "message_type": "hypothesis",
                "agent_id": "agent-002",
                "payload": {"data": "test"},
                "timestamp": "2024-01-15T10:00:00+00:00",
                "correlation_id": "xyz-789",
            }
        )

        msg = GossipMessage.from_json(json_str)

        assert msg.message_type == "hypothesis"
        assert msg.agent_id == "agent-002"
        assert msg.payload == {"data": "test"}
        assert msg.correlation_id == "xyz-789"

    def test_from_json_invalid_json(self) -> None:
        """Test from_json with invalid JSON raises error."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            GossipMessage.from_json("not valid json")

    def test_from_json_missing_field(self) -> None:
        """Test from_json with missing required field raises error."""
        json_str = json.dumps(
            {
                "message_type": "hypothesis",
                # Missing agent_id and other required fields
            }
        )

        with pytest.raises(ValueError, match="Missing required field"):
            GossipMessage.from_json(json_str)

    def test_from_bytes(self) -> None:
        """Test deserializing message from bytes."""
        json_str = json.dumps(
            {
                "message_type": "vote",
                "agent_id": "agent-003",
                "payload": {"vote": "yes"},
                "timestamp": "2024-01-15T10:00:00+00:00",
            }
        )
        data = json_str.encode("utf-8")

        msg = GossipMessage.from_bytes(data)

        assert msg.message_type == "vote"
        assert msg.agent_id == "agent-003"

    def test_roundtrip_serialization(self) -> None:
        """Test that serialization and deserialization are reversible."""
        original = GossipMessage(
            message_type="hypothesis",
            agent_id="agent-001",
            payload={"complex": {"nested": "data", "list": [1, 2, 3]}},
            correlation_id="test-correlation",
        )

        # Serialize and deserialize
        data = original.to_bytes()
        restored = GossipMessage.from_bytes(data)

        assert restored.message_type == original.message_type
        assert restored.agent_id == original.agent_id
        assert restored.payload == original.payload
        assert restored.correlation_id == original.correlation_id


class TestHypothesisGossipInitialization:
    """Tests for HypothesisGossip initialization."""

    def test_create_gossip(self) -> None:
        """Test creating HypothesisGossip instance."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        assert gossip.agent_id == "agent-001"
        assert gossip.is_running is False

    def test_create_gossip_with_qos(self) -> None:
        """Test creating HypothesisGossip with custom QoS."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001", default_qos=2)

        assert gossip._default_qos == 2


class TestHypothesisGossipStartStop:
    """Tests for HypothesisGossip start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_subscribes_to_topics(self) -> None:
        """Test that start() subscribes to relevant topics."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        mock_mqtt.subscribe = AsyncMock()
        mock_mqtt.publish = AsyncMock()
        mock_mqtt.on_message = MagicMock(return_value=mock_mqtt)

        gossip = HypothesisGossip(mock_mqtt, "agent-001")
        await gossip.start()

        assert gossip.is_running is True
        assert mock_mqtt.subscribe.call_count == 3
        mock_mqtt.on_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_already_running(self) -> None:
        """Test that starting when already running logs warning."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        mock_mqtt.subscribe = AsyncMock()
        mock_mqtt.publish = AsyncMock()
        mock_mqtt.on_message = MagicMock(return_value=mock_mqtt)

        gossip = HypothesisGossip(mock_mqtt, "agent-001")
        await gossip.start()
        await gossip.start()  # Second start should just log warning

        assert gossip.is_running is True

    @pytest.mark.asyncio
    async def test_stop_unsubscribes_from_topics(self) -> None:
        """Test that stop() unsubscribes from topics."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        mock_mqtt.subscribe = AsyncMock()
        mock_mqtt.unsubscribe = AsyncMock()
        mock_mqtt.publish = AsyncMock()
        mock_mqtt.on_message = MagicMock(return_value=mock_mqtt)

        gossip = HypothesisGossip(mock_mqtt, "agent-001")
        await gossip.start()
        await gossip.stop()

        assert gossip.is_running is False
        assert mock_mqtt.unsubscribe.call_count == 3

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self) -> None:
        """Test that stopping when not running logs warning."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        await gossip.stop()  # Should not raise
        assert gossip.is_running is False


class TestHypothesisGossipBroadcast:
    """Tests for HypothesisGossip broadcast operations."""

    @pytest.fixture
    def running_gossip(self) -> HypothesisGossip:
        """Create a gossip instance that appears to be running."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        mock_mqtt.publish = AsyncMock()
        gossip = HypothesisGossip(mock_mqtt, "agent-001")
        gossip._running = True
        return gossip

    @pytest.fixture
    def sample_hypothesis(self) -> Hypothesis:
        """Create a sample hypothesis for testing."""
        return Hypothesis(
            tag_id="opc.tcp://localhost|ns=2;s=Temperature",
            candidates=[
                Candidate(
                    irdi="0173-1#01-ABA234#001",
                    confidence=0.95,
                    source_model="semantic-v1",
                )
            ],
            agent_id="agent-001",
        )

    @pytest.fixture
    def sample_vote(self) -> Vote:
        """Create a sample vote for testing."""
        return Vote(
            agent_id="agent-001",
            candidate_irdi="0173-1#01-ABA234#001",
            confidence=0.9,
            reliability_score=0.85,
        )

    @pytest.fixture
    def sample_consensus(self) -> ConsensusRecord:
        """Create a sample consensus record for testing."""
        return ConsensusRecord(
            tag_id="tag-123",
            agreed_irdi="0173-1#01-ABA234#001",
            consensus_confidence=0.92,
            votes=[
                Vote(
                    agent_id="agent-001",
                    candidate_irdi="0173-1#01-ABA234#001",
                    confidence=0.9,
                    reliability_score=0.85,
                )
            ],
            quorum_type="hard",
        )

    @pytest.mark.asyncio
    async def test_broadcast_hypothesis(
        self,
        running_gossip: HypothesisGossip,
        sample_hypothesis: Hypothesis,
    ) -> None:
        """Test broadcasting a hypothesis."""
        await running_gossip.broadcast_hypothesis("tag-123", sample_hypothesis)

        # Should publish to both agent hypothesis topic and tag candidates topic
        assert running_gossip._mqtt.publish.call_count == 2

    @pytest.mark.asyncio
    async def test_broadcast_hypothesis_not_running(
        self,
        sample_hypothesis: Hypothesis,
    ) -> None:
        """Test that broadcasting when not running raises error."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        with pytest.raises(HypothesisGossipError, match="not running"):
            await gossip.broadcast_hypothesis("tag-123", sample_hypothesis)

    @pytest.mark.asyncio
    async def test_broadcast_vote(
        self,
        running_gossip: HypothesisGossip,
        sample_vote: Vote,
    ) -> None:
        """Test broadcasting a vote."""
        await running_gossip.broadcast_vote("tag-123", sample_vote)

        running_gossip._mqtt.publish.assert_called_once()
        call_args = running_gossip._mqtt.publish.call_args
        assert call_args.kwargs["retain"] is False

    @pytest.mark.asyncio
    async def test_broadcast_consensus_retained(
        self,
        running_gossip: HypothesisGossip,
        sample_consensus: ConsensusRecord,
    ) -> None:
        """Test that broadcasting consensus uses retain=True."""
        await running_gossip.broadcast_consensus("tag-123", sample_consensus)

        running_gossip._mqtt.publish.assert_called_once()
        call_args = running_gossip._mqtt.publish.call_args
        assert call_args.kwargs["retain"] is True

    @pytest.mark.asyncio
    async def test_broadcast_with_correlation_id(
        self,
        running_gossip: HypothesisGossip,
        sample_hypothesis: Hypothesis,
    ) -> None:
        """Test broadcasting with explicit correlation ID."""
        await running_gossip.broadcast_hypothesis(
            "tag-123", sample_hypothesis, correlation_id="custom-correlation"
        )

        # Verify the message contains the correlation ID
        call_args = running_gossip._mqtt.publish.call_args_list[0]
        payload = call_args.args[1]
        msg = GossipMessage.from_bytes(payload)
        assert msg.correlation_id == "custom-correlation"


class TestHypothesisGossipHeartbeat:
    """Tests for HypothesisGossip heartbeat."""

    @pytest.mark.asyncio
    async def test_send_heartbeat(self) -> None:
        """Test sending a heartbeat."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        mock_mqtt.publish = AsyncMock()
        gossip = HypothesisGossip(mock_mqtt, "agent-001")
        gossip._running = True

        await gossip.send_heartbeat()

        mock_mqtt.publish.assert_called_once()
        call_args = mock_mqtt.publish.call_args
        assert agent_status_topic("agent-001") in call_args.args[0]
        assert call_args.kwargs["qos"] == 0  # Heartbeats use QoS 0
        assert call_args.kwargs["retain"] is True

    @pytest.mark.asyncio
    async def test_send_heartbeat_with_extra_info(self) -> None:
        """Test sending a heartbeat with extra info."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        mock_mqtt.publish = AsyncMock()
        gossip = HypothesisGossip(mock_mqtt, "agent-001")
        gossip._running = True

        await gossip.send_heartbeat(extra_info={"model_version": "v2.0"})

        call_args = mock_mqtt.publish.call_args
        payload = call_args.args[1]
        msg = GossipMessage.from_bytes(payload)
        assert msg.payload.get("model_version") == "v2.0"


class TestHypothesisGossipCallbacks:
    """Tests for HypothesisGossip callback registration."""

    def test_on_hypothesis_callback_registration(self) -> None:
        """Test registering hypothesis callback."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        async def callback(tag_id: str, hypothesis: Hypothesis) -> None:
            pass

        result = gossip.on_hypothesis(callback)

        assert result is gossip  # Method chaining
        assert callback in gossip._hypothesis_callbacks

    def test_on_vote_callback_registration(self) -> None:
        """Test registering vote callback."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        async def callback(tag_id: str, vote: Vote) -> None:
            pass

        result = gossip.on_vote(callback)

        assert result is gossip
        assert callback in gossip._vote_callbacks

    def test_on_consensus_callback_registration(self) -> None:
        """Test registering consensus callback."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        async def callback(tag_id: str, consensus: ConsensusRecord) -> None:
            pass

        result = gossip.on_consensus(callback)

        assert result is gossip
        assert callback in gossip._consensus_callbacks

    def test_on_heartbeat_callback_registration(self) -> None:
        """Test registering heartbeat callback."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        async def callback(agent_id: str, payload: dict) -> None:
            pass

        result = gossip.on_heartbeat(callback)

        assert result is gossip
        assert callback in gossip._heartbeat_callbacks

    def test_callback_chaining(self) -> None:
        """Test method chaining with callbacks."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        async def hypo_cb(tag_id: str, hypothesis: Hypothesis) -> None:
            pass

        async def vote_cb(tag_id: str, vote: Vote) -> None:
            pass

        async def cons_cb(tag_id: str, consensus: ConsensusRecord) -> None:
            pass

        async def hb_cb(agent_id: str, payload: dict) -> None:
            pass

        gossip.on_hypothesis(hypo_cb).on_vote(vote_cb).on_consensus(cons_cb).on_heartbeat(hb_cb)

        assert hypo_cb in gossip._hypothesis_callbacks
        assert vote_cb in gossip._vote_callbacks
        assert cons_cb in gossip._consensus_callbacks
        assert hb_cb in gossip._heartbeat_callbacks


class TestHypothesisGossipMessageHandling:
    """Tests for HypothesisGossip message handling."""

    @pytest.mark.asyncio
    async def test_handle_message_ignores_own_messages(self) -> None:
        """Test that own messages are ignored."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        received: list[str] = []

        async def callback(tag_id: str, hypothesis: Hypothesis) -> None:
            received.append(tag_id)

        gossip.on_hypothesis(callback)

        # Create a message from the same agent
        msg = GossipMessage(
            message_type="hypothesis",
            agent_id="agent-001",  # Same as gossip agent
            payload={
                "tag_id": "tag-123",
                "hypothesis": {
                    "tag_id": "tag-123",
                    "candidates": [
                        {
                            "irdi": "0173-1#01-ABA234#001",
                            "confidence": 0.9,
                            "source_model": "model-v1",
                        }
                    ],
                    "agent_id": "agent-001",
                    "created_at": "2024-01-15T10:00:00Z",
                },
            },
        )

        await gossip._handle_message("test/topic", msg.to_bytes())

        assert len(received) == 0  # Should be empty since it was our own message

    @pytest.mark.asyncio
    async def test_handle_hypothesis_message(self) -> None:
        """Test handling a hypothesis message."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        received: list[tuple[str, Hypothesis]] = []

        async def callback(tag_id: str, hypothesis: Hypothesis) -> None:
            received.append((tag_id, hypothesis))

        gossip.on_hypothesis(callback)

        # Create a message from a different agent
        msg = GossipMessage(
            message_type="hypothesis",
            agent_id="agent-002",  # Different agent
            payload={
                "tag_id": "tag-123",
                "hypothesis": {
                    "tag_id": "tag-123",
                    "candidates": [
                        {
                            "irdi": "0173-1#01-ABA234#001",
                            "confidence": 0.9,
                            "source_model": "model-v1",
                        }
                    ],
                    "agent_id": "agent-002",
                    "created_at": "2024-01-15T10:00:00Z",
                },
            },
        )

        await gossip._handle_message("test/topic", msg.to_bytes())

        assert len(received) == 1
        assert received[0][0] == "tag-123"
        assert received[0][1].agent_id == "agent-002"

    @pytest.mark.asyncio
    async def test_handle_vote_message(self) -> None:
        """Test handling a vote message."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        received: list[tuple[str, Vote]] = []

        async def callback(tag_id: str, vote: Vote) -> None:
            received.append((tag_id, vote))

        gossip.on_vote(callback)

        msg = GossipMessage(
            message_type="vote",
            agent_id="agent-002",
            payload={
                "tag_id": "tag-456",
                "vote": {
                    "agent_id": "agent-002",
                    "candidate_irdi": "0173-1#01-ABA234#001",
                    "confidence": 0.85,
                    "reliability_score": 0.9,
                    "timestamp": "2024-01-15T10:00:00Z",
                },
            },
        )

        await gossip._handle_message("test/topic", msg.to_bytes())

        assert len(received) == 1
        assert received[0][0] == "tag-456"
        assert received[0][1].confidence == 0.85

    @pytest.mark.asyncio
    async def test_handle_consensus_message(self) -> None:
        """Test handling a consensus message."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        received: list[tuple[str, ConsensusRecord]] = []

        async def callback(tag_id: str, consensus: ConsensusRecord) -> None:
            received.append((tag_id, consensus))

        gossip.on_consensus(callback)

        msg = GossipMessage(
            message_type="consensus",
            agent_id="agent-002",
            payload={
                "tag_id": "tag-789",
                "consensus": {
                    "tag_id": "tag-789",
                    "agreed_irdi": "0173-1#01-ABA234#001",
                    "consensus_confidence": 0.92,
                    "votes": [
                        {
                            "agent_id": "agent-002",
                            "candidate_irdi": "0173-1#01-ABA234#001",
                            "confidence": 0.9,
                            "reliability_score": 0.85,
                            "timestamp": "2024-01-15T10:00:00Z",
                        }
                    ],
                    "quorum_type": "hard",
                    "created_at": "2024-01-15T10:00:00Z",
                    "updated_at": "2024-01-15T10:00:00Z",
                },
            },
        )

        await gossip._handle_message("test/topic", msg.to_bytes())

        assert len(received) == 1
        assert received[0][0] == "tag-789"
        assert received[0][1].quorum_type == "hard"

    @pytest.mark.asyncio
    async def test_handle_heartbeat_message(self) -> None:
        """Test handling a heartbeat message."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        received: list[tuple[str, dict]] = []

        async def callback(agent_id: str, payload: dict) -> None:
            received.append((agent_id, payload))

        gossip.on_heartbeat(callback)

        msg = GossipMessage(
            message_type="heartbeat",
            agent_id="agent-003",
            payload={"status": "alive", "model_version": "v1.5"},
        )

        await gossip._handle_message("test/topic", msg.to_bytes())

        assert len(received) == 1
        assert received[0][0] == "agent-003"
        assert received[0][1]["status"] == "alive"

    @pytest.mark.asyncio
    async def test_handle_invalid_message(self) -> None:
        """Test that invalid messages are handled gracefully."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        # Should not raise, just log warning
        await gossip._handle_message("test/topic", b"not valid json")

    @pytest.mark.asyncio
    async def test_callback_error_handling(self) -> None:
        """Test that callback errors are caught and logged."""
        mock_mqtt = MagicMock(spec=MQTTClient)
        gossip = HypothesisGossip(mock_mqtt, "agent-001")

        async def failing_callback(agent_id: str, payload: dict) -> None:
            raise RuntimeError("Test error")

        successful_calls: list[str] = []

        async def success_callback(agent_id: str, payload: dict) -> None:
            successful_calls.append(agent_id)

        gossip.on_heartbeat(failing_callback)
        gossip.on_heartbeat(success_callback)

        msg = GossipMessage(
            message_type="heartbeat",
            agent_id="agent-002",
            payload={"status": "alive"},
        )

        # Should not raise
        await gossip._handle_message("test/topic", msg.to_bytes())

        # Second callback should still be called
        assert len(successful_calls) == 1


class TestHypothesisGossipError:
    """Tests for HypothesisGossipError exception."""

    def test_hypothesis_gossip_error(self) -> None:
        """Test HypothesisGossipError exception."""
        error = HypothesisGossipError("Test error message")
        assert str(error) == "Test error message"
