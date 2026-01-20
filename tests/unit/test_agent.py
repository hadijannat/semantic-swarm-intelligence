"""Unit tests for the SemanticAgent orchestration class."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noa_swarm.common.schemas import (
    Candidate,
    ConsensusRecord,
    Hypothesis,
    TagRecord,
    Vote,
)
from noa_swarm.swarm.agent import (
    AgentState,
    SemanticAgent,
    SemanticAgentConfig,
    SemanticAgentError,
)


class TestSemanticAgentConfig:
    """Tests for SemanticAgentConfig dataclass."""

    def test_default_values(self) -> None:
        """Test SemanticAgentConfig has expected default values."""
        config = SemanticAgentConfig(
            agent_id="agent-001",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
        )

        assert config.agent_id == "agent-001"
        assert config.opcua_endpoint == "opc.tcp://localhost:4840"
        assert config.mqtt_host == "localhost"
        assert config.mqtt_port == 1883
        assert config.poll_interval_seconds == 30.0
        assert config.shutdown_timeout_seconds == 10.0

    def test_custom_values(self) -> None:
        """Test SemanticAgentConfig accepts custom values."""
        config = SemanticAgentConfig(
            agent_id="agent-002",
            opcua_endpoint="opc.tcp://192.168.1.100:4840",
            mqtt_host="mqtt.example.com",
            mqtt_port=8883,
            poll_interval_seconds=60.0,
            shutdown_timeout_seconds=30.0,
        )

        assert config.agent_id == "agent-002"
        assert config.opcua_endpoint == "opc.tcp://192.168.1.100:4840"
        assert config.mqtt_host == "mqtt.example.com"
        assert config.mqtt_port == 8883
        assert config.poll_interval_seconds == 60.0
        assert config.shutdown_timeout_seconds == 30.0

    def test_empty_agent_id_raises(self) -> None:
        """Test that empty agent_id raises ValueError."""
        with pytest.raises(ValueError, match="agent_id"):
            SemanticAgentConfig(
                agent_id="",
                opcua_endpoint="opc.tcp://localhost:4840",
                mqtt_host="localhost",
            )

    def test_empty_opcua_endpoint_raises(self) -> None:
        """Test that empty opcua_endpoint raises ValueError."""
        with pytest.raises(ValueError, match="opcua_endpoint"):
            SemanticAgentConfig(
                agent_id="agent-001",
                opcua_endpoint="",
                mqtt_host="localhost",
            )

    def test_empty_mqtt_host_raises(self) -> None:
        """Test that empty mqtt_host raises ValueError."""
        with pytest.raises(ValueError, match="mqtt_host"):
            SemanticAgentConfig(
                agent_id="agent-001",
                opcua_endpoint="opc.tcp://localhost:4840",
                mqtt_host="",
            )

    def test_invalid_mqtt_port_raises(self) -> None:
        """Test that invalid mqtt_port raises ValueError."""
        with pytest.raises(ValueError, match="mqtt_port"):
            SemanticAgentConfig(
                agent_id="agent-001",
                opcua_endpoint="opc.tcp://localhost:4840",
                mqtt_host="localhost",
                mqtt_port=-1,
            )

        with pytest.raises(ValueError, match="mqtt_port"):
            SemanticAgentConfig(
                agent_id="agent-001",
                opcua_endpoint="opc.tcp://localhost:4840",
                mqtt_host="localhost",
                mqtt_port=70000,
            )

    def test_invalid_poll_interval_raises(self) -> None:
        """Test that invalid poll_interval_seconds raises ValueError."""
        with pytest.raises(ValueError, match="poll_interval"):
            SemanticAgentConfig(
                agent_id="agent-001",
                opcua_endpoint="opc.tcp://localhost:4840",
                mqtt_host="localhost",
                poll_interval_seconds=0.0,
            )

        with pytest.raises(ValueError, match="poll_interval"):
            SemanticAgentConfig(
                agent_id="agent-001",
                opcua_endpoint="opc.tcp://localhost:4840",
                mqtt_host="localhost",
                poll_interval_seconds=-1.0,
            )

    def test_invalid_shutdown_timeout_raises(self) -> None:
        """Test that invalid shutdown_timeout_seconds raises ValueError."""
        with pytest.raises(ValueError, match="shutdown_timeout"):
            SemanticAgentConfig(
                agent_id="agent-001",
                opcua_endpoint="opc.tcp://localhost:4840",
                mqtt_host="localhost",
                shutdown_timeout_seconds=-1.0,
            )


class TestAgentState:
    """Tests for AgentState enum."""

    def test_all_states_defined(self) -> None:
        """Test that all expected states are defined."""
        assert AgentState.INITIALIZING.value == "initializing"
        assert AgentState.RUNNING.value == "running"
        assert AgentState.STOPPING.value == "stopping"
        assert AgentState.STOPPED.value == "stopped"

    def test_state_names(self) -> None:
        """Test state names are strings."""
        for state in AgentState:
            assert isinstance(state.value, str)


class TestSemanticAgentInit:
    """Tests for SemanticAgent initialization."""

    @pytest.fixture
    def config(self) -> SemanticAgentConfig:
        """Create a sample config for testing."""
        return SemanticAgentConfig(
            agent_id="test-agent",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
        )

    def test_init_with_config_only(self, config: SemanticAgentConfig) -> None:
        """Test initialization with config only (no injected dependencies)."""
        agent = SemanticAgent(config)

        assert agent.agent_id == "test-agent"
        assert agent.state == AgentState.INITIALIZING
        assert not agent.is_running

    def test_init_with_injected_dependencies(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test initialization with injected dependencies."""
        mock_consensus = MagicMock()
        mock_reputation = MagicMock()
        mock_inference = MagicMock()

        agent = SemanticAgent(
            config,
            consensus_engine=mock_consensus,
            reputation_tracker=mock_reputation,
            inference_engine=mock_inference,
        )

        assert agent.agent_id == "test-agent"
        assert agent.state == AgentState.INITIALIZING

    def test_agent_id_property(self, config: SemanticAgentConfig) -> None:
        """Test agent_id property returns correct value."""
        agent = SemanticAgent(config)
        assert agent.agent_id == config.agent_id

    def test_state_property(self, config: SemanticAgentConfig) -> None:
        """Test state property returns current state."""
        agent = SemanticAgent(config)
        assert agent.state == AgentState.INITIALIZING

    def test_is_running_property(self, config: SemanticAgentConfig) -> None:
        """Test is_running property."""
        agent = SemanticAgent(config)
        assert agent.is_running is False


class TestSemanticAgentLifecycle:
    """Tests for SemanticAgent lifecycle methods."""

    @pytest.fixture
    def config(self) -> SemanticAgentConfig:
        """Create a sample config for testing."""
        return SemanticAgentConfig(
            agent_id="test-agent",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
            poll_interval_seconds=0.1,  # Short interval for testing
            shutdown_timeout_seconds=1.0,
        )

    @pytest.fixture
    def mock_dependencies(self) -> dict[str, Any]:
        """Create mock dependencies for testing."""
        return {
            "consensus_engine": MagicMock(),
            "reputation_tracker": MagicMock(),
            "inference_engine": MagicMock(),
        }

    @pytest.mark.asyncio
    async def test_start_transitions_to_running(
        self, config: SemanticAgentConfig, mock_dependencies: dict[str, Any]
    ) -> None:
        """Test that start() transitions state to RUNNING."""
        agent = SemanticAgent(config, **mock_dependencies)

        # Mock the internal components to prevent actual connections
        with patch.object(agent, "_connect_mqtt", new_callable=AsyncMock), \
             patch.object(agent, "_connect_opcua", new_callable=AsyncMock), \
             patch.object(agent, "_start_gossip", new_callable=AsyncMock), \
             patch.object(agent, "_run_lifecycle_loop", new_callable=AsyncMock):
            await agent.start()

            assert agent.state == AgentState.RUNNING
            assert agent.is_running is True

    @pytest.mark.asyncio
    async def test_start_when_already_running_logs_warning(
        self, config: SemanticAgentConfig, mock_dependencies: dict[str, Any]
    ) -> None:
        """Test that start() when already running logs warning."""
        agent = SemanticAgent(config, **mock_dependencies)

        with patch.object(agent, "_connect_mqtt", new_callable=AsyncMock), \
             patch.object(agent, "_connect_opcua", new_callable=AsyncMock), \
             patch.object(agent, "_start_gossip", new_callable=AsyncMock), \
             patch.object(agent, "_run_lifecycle_loop", new_callable=AsyncMock):
            await agent.start()
            # Second start should not raise but should be a no-op
            await agent.start()  # Should not raise

            assert agent.state == AgentState.RUNNING

    @pytest.mark.asyncio
    async def test_stop_transitions_through_stopping_to_stopped(
        self, config: SemanticAgentConfig, mock_dependencies: dict[str, Any]
    ) -> None:
        """Test that stop() transitions through STOPPING to STOPPED."""
        agent = SemanticAgent(config, **mock_dependencies)

        with patch.object(agent, "_connect_mqtt", new_callable=AsyncMock), \
             patch.object(agent, "_connect_opcua", new_callable=AsyncMock), \
             patch.object(agent, "_start_gossip", new_callable=AsyncMock), \
             patch.object(agent, "_run_lifecycle_loop", new_callable=AsyncMock), \
             patch.object(agent, "_disconnect_mqtt", new_callable=AsyncMock), \
             patch.object(agent, "_disconnect_opcua", new_callable=AsyncMock), \
             patch.object(agent, "_stop_gossip", new_callable=AsyncMock):
            await agent.start()
            await agent.stop()

            assert agent.state == AgentState.STOPPED
            assert agent.is_running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_noop(
        self, config: SemanticAgentConfig, mock_dependencies: dict[str, Any]
    ) -> None:
        """Test that stop() when not running is a no-op."""
        agent = SemanticAgent(config, **mock_dependencies)

        await agent.stop()  # Should not raise

        assert agent.state == AgentState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_cancels_running_tasks(
        self, config: SemanticAgentConfig, mock_dependencies: dict[str, Any]
    ) -> None:
        """Test that stop() cancels running tasks."""
        agent = SemanticAgent(config, **mock_dependencies)

        with patch.object(agent, "_connect_mqtt", new_callable=AsyncMock), \
             patch.object(agent, "_connect_opcua", new_callable=AsyncMock), \
             patch.object(agent, "_start_gossip", new_callable=AsyncMock), \
             patch.object(agent, "_run_lifecycle_loop", new_callable=AsyncMock), \
             patch.object(agent, "_disconnect_mqtt", new_callable=AsyncMock), \
             patch.object(agent, "_disconnect_opcua", new_callable=AsyncMock), \
             patch.object(agent, "_stop_gossip", new_callable=AsyncMock):
            await agent.start()

            # Create a real asyncio task that will be cancelled
            async def long_running():
                await asyncio.sleep(100)

            real_task = asyncio.create_task(long_running())
            agent._lifecycle_task = real_task

            await agent.stop()

            assert real_task.cancelled()


class TestSemanticAgentDiscovery:
    """Tests for SemanticAgent tag discovery."""

    @pytest.fixture
    def config(self) -> SemanticAgentConfig:
        """Create a sample config for testing."""
        return SemanticAgentConfig(
            agent_id="test-agent",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
        )

    @pytest.fixture
    def sample_tags(self) -> list[TagRecord]:
        """Create sample tags for testing."""
        return [
            TagRecord(
                node_id="ns=2;s=Temperature.PV",
                browse_name="Temperature_PV",
                display_name="Temperature Process Value",
                data_type="Double",
                source_server="opc.tcp://localhost:4840",
            ),
            TagRecord(
                node_id="ns=2;s=Pressure.PV",
                browse_name="Pressure_PV",
                display_name="Pressure Process Value",
                data_type="Double",
                source_server="opc.tcp://localhost:4840",
            ),
        ]

    @pytest.mark.asyncio
    async def test_discover_tags_returns_tag_records(
        self, config: SemanticAgentConfig, sample_tags: list[TagRecord]
    ) -> None:
        """Test that discover_tags returns TagRecord list."""
        agent = SemanticAgent(config)

        # Mock the OPC UA browser
        mock_browser = AsyncMock()
        mock_browser.browse_all_tags = AsyncMock(return_value=sample_tags)
        agent._opcua_browser = mock_browser

        tags = await agent.discover_tags()

        assert len(tags) == 2
        assert all(isinstance(t, TagRecord) for t in tags)
        mock_browser.browse_all_tags.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_tags_updates_metrics(
        self, config: SemanticAgentConfig, sample_tags: list[TagRecord]
    ) -> None:
        """Test that discover_tags updates health metrics."""
        agent = SemanticAgent(config)

        mock_browser = AsyncMock()
        mock_browser.browse_all_tags = AsyncMock(return_value=sample_tags)
        agent._opcua_browser = mock_browser

        await agent.discover_tags()

        health = agent.health_check()
        assert health["tags_discovered"] == 2
        assert health["last_discovery_time"] is not None

    @pytest.mark.asyncio
    async def test_discover_tags_handles_empty_result(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that discover_tags handles empty results."""
        agent = SemanticAgent(config)

        mock_browser = AsyncMock()
        mock_browser.browse_all_tags = AsyncMock(return_value=[])
        agent._opcua_browser = mock_browser

        tags = await agent.discover_tags()

        assert tags == []


class TestSemanticAgentInference:
    """Tests for SemanticAgent ML inference."""

    @pytest.fixture
    def config(self) -> SemanticAgentConfig:
        """Create a sample config for testing."""
        return SemanticAgentConfig(
            agent_id="test-agent",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
        )

    @pytest.fixture
    def sample_tags(self) -> list[TagRecord]:
        """Create sample tags for testing."""
        return [
            TagRecord(
                node_id="ns=2;s=Temperature.PV",
                browse_name="Temperature_PV",
                display_name="Temperature Process Value",
                data_type="Double",
                source_server="opc.tcp://localhost:4840",
            ),
        ]

    @pytest.fixture
    def sample_hypotheses(self) -> list[Hypothesis]:
        """Create sample hypotheses for testing."""
        return [
            Hypothesis(
                tag_id="opc.tcp://localhost:4840|ns=2;s=Temperature.PV",
                candidates=[
                    Candidate(
                        irdi="0173-1#01-ABA234#001",
                        confidence=0.95,
                        source_model="charcnn-v1",
                    ),
                    Candidate(
                        irdi="0173-1#01-XYZ789#001",
                        confidence=0.75,
                        source_model="charcnn-v1",
                    ),
                ],
                agent_id="test-agent",
            ),
        ]

    @pytest.mark.asyncio
    async def test_infer_mappings_returns_hypotheses(
        self,
        config: SemanticAgentConfig,
        sample_tags: list[TagRecord],
        sample_hypotheses: list[Hypothesis],
    ) -> None:
        """Test that infer_mappings returns Hypothesis list."""
        mock_inference = MagicMock()
        mock_inference.infer = MagicMock(return_value=sample_hypotheses)

        agent = SemanticAgent(config, inference_engine=mock_inference)

        hypotheses = await agent.infer_mappings(sample_tags)

        assert len(hypotheses) == 1
        assert all(isinstance(h, Hypothesis) for h in hypotheses)

    @pytest.mark.asyncio
    async def test_infer_mappings_updates_metrics(
        self,
        config: SemanticAgentConfig,
        sample_tags: list[TagRecord],
        sample_hypotheses: list[Hypothesis],
    ) -> None:
        """Test that infer_mappings updates health metrics."""
        mock_inference = MagicMock()
        mock_inference.infer = MagicMock(return_value=sample_hypotheses)

        agent = SemanticAgent(config, inference_engine=mock_inference)

        await agent.infer_mappings(sample_tags)

        health = agent.health_check()
        assert health["hypotheses_generated"] == 1
        assert health["last_inference_time"] is not None

    @pytest.mark.asyncio
    async def test_infer_mappings_handles_empty_tags(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that infer_mappings handles empty tag list."""
        mock_inference = MagicMock()
        mock_inference.infer = MagicMock(return_value=[])

        agent = SemanticAgent(config, inference_engine=mock_inference)

        hypotheses = await agent.infer_mappings([])

        assert hypotheses == []

    @pytest.mark.asyncio
    async def test_infer_mappings_without_engine_raises(
        self, config: SemanticAgentConfig, sample_tags: list[TagRecord]
    ) -> None:
        """Test that infer_mappings without engine raises error."""
        agent = SemanticAgent(config)
        agent._inference_engine = None

        with pytest.raises(SemanticAgentError, match="(?i)inference engine"):
            await agent.infer_mappings(sample_tags)


class TestSemanticAgentGossip:
    """Tests for SemanticAgent gossip broadcasting."""

    @pytest.fixture
    def config(self) -> SemanticAgentConfig:
        """Create a sample config for testing."""
        return SemanticAgentConfig(
            agent_id="test-agent",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
        )

    @pytest.fixture
    def sample_hypotheses(self) -> list[Hypothesis]:
        """Create sample hypotheses for testing."""
        return [
            Hypothesis(
                tag_id="opc.tcp://localhost:4840|ns=2;s=Temperature.PV",
                candidates=[
                    Candidate(
                        irdi="0173-1#01-ABA234#001",
                        confidence=0.95,
                        source_model="charcnn-v1",
                    ),
                ],
                agent_id="test-agent",
            ),
        ]

    @pytest.mark.asyncio
    async def test_gossip_hypotheses_broadcasts_to_swarm(
        self, config: SemanticAgentConfig, sample_hypotheses: list[Hypothesis]
    ) -> None:
        """Test that gossip_hypotheses broadcasts via MQTT."""
        agent = SemanticAgent(config)

        mock_gossip = AsyncMock()
        mock_gossip.broadcast_hypothesis = AsyncMock()
        agent._gossip = mock_gossip

        await agent.gossip_hypotheses(sample_hypotheses)

        mock_gossip.broadcast_hypothesis.assert_called_once()

    @pytest.mark.asyncio
    async def test_gossip_hypotheses_updates_metrics(
        self, config: SemanticAgentConfig, sample_hypotheses: list[Hypothesis]
    ) -> None:
        """Test that gossip_hypotheses updates health metrics."""
        agent = SemanticAgent(config)

        mock_gossip = AsyncMock()
        mock_gossip.broadcast_hypothesis = AsyncMock()
        agent._gossip = mock_gossip

        await agent.gossip_hypotheses(sample_hypotheses)

        health = agent.health_check()
        assert health["last_gossip_time"] is not None

    @pytest.mark.asyncio
    async def test_gossip_hypotheses_handles_empty_list(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that gossip_hypotheses handles empty list."""
        agent = SemanticAgent(config)

        mock_gossip = AsyncMock()
        mock_gossip.broadcast_hypothesis = AsyncMock()
        agent._gossip = mock_gossip

        await agent.gossip_hypotheses([])

        # Should not call broadcast for empty list
        mock_gossip.broadcast_hypothesis.assert_not_called()


class TestSemanticAgentVoting:
    """Tests for SemanticAgent voting."""

    @pytest.fixture
    def config(self) -> SemanticAgentConfig:
        """Create a sample config for testing."""
        return SemanticAgentConfig(
            agent_id="test-agent",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
        )

    @pytest.mark.asyncio
    async def test_vote_on_tags_broadcasts_votes(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that vote_on_tags broadcasts votes via gossip."""
        agent = SemanticAgent(config)

        mock_gossip = AsyncMock()
        mock_gossip.broadcast_vote = AsyncMock()
        agent._gossip = mock_gossip

        # Setup mock reputation tracker
        mock_reputation = MagicMock()
        mock_reputation.get_reliability = MagicMock(return_value=0.9)
        agent._reputation_tracker = mock_reputation

        # Setup pending hypotheses
        agent._pending_hypotheses = {
            "tag-1": Hypothesis(
                tag_id="tag-1",
                candidates=[
                    Candidate(
                        irdi="0173-1#01-ABA234#001",
                        confidence=0.95,
                        source_model="charcnn-v1",
                    ),
                ],
                agent_id="test-agent",
            ),
        }

        await agent.vote_on_tags(["tag-1"])

        mock_gossip.broadcast_vote.assert_called_once()

    @pytest.mark.asyncio
    async def test_vote_on_tags_updates_metrics(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that vote_on_tags updates health metrics."""
        agent = SemanticAgent(config)

        mock_gossip = AsyncMock()
        mock_gossip.broadcast_vote = AsyncMock()
        agent._gossip = mock_gossip

        mock_reputation = MagicMock()
        mock_reputation.get_reliability = MagicMock(return_value=0.9)
        agent._reputation_tracker = mock_reputation

        agent._pending_hypotheses = {
            "tag-1": Hypothesis(
                tag_id="tag-1",
                candidates=[
                    Candidate(
                        irdi="0173-1#01-ABA234#001",
                        confidence=0.95,
                        source_model="charcnn-v1",
                    ),
                ],
                agent_id="test-agent",
            ),
        }

        await agent.vote_on_tags(["tag-1"])

        health = agent.health_check()
        assert health["votes_cast"] == 1


class TestSemanticAgentCommit:
    """Tests for SemanticAgent commit operations."""

    @pytest.fixture
    def config(self) -> SemanticAgentConfig:
        """Create a sample config for testing."""
        return SemanticAgentConfig(
            agent_id="test-agent",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
        )

    @pytest.fixture
    def sample_records(self) -> list[ConsensusRecord]:
        """Create sample consensus records for testing."""
        return [
            ConsensusRecord(
                tag_id="tag-1",
                agreed_irdi="0173-1#01-ABA234#001",
                consensus_confidence=0.95,
                votes=[
                    Vote(
                        agent_id="test-agent",
                        candidate_irdi="0173-1#01-ABA234#001",
                        confidence=0.95,
                        reliability_score=0.9,
                    ),
                    Vote(
                        agent_id="agent-002",
                        candidate_irdi="0173-1#01-ABA234#001",
                        confidence=0.88,
                        reliability_score=0.85,
                    ),
                ],
                quorum_type="hard",
            ),
        ]

    @pytest.mark.asyncio
    async def test_commit_mappings_stores_records(
        self, config: SemanticAgentConfig, sample_records: list[ConsensusRecord]
    ) -> None:
        """Test that commit_mappings stores consensus records."""
        agent = SemanticAgent(config)

        await agent.commit_mappings(sample_records)

        # Verify records are stored
        assert len(agent._committed_mappings) == 1
        assert "tag-1" in agent._committed_mappings

    @pytest.mark.asyncio
    async def test_commit_mappings_broadcasts_consensus(
        self, config: SemanticAgentConfig, sample_records: list[ConsensusRecord]
    ) -> None:
        """Test that commit_mappings broadcasts via gossip."""
        agent = SemanticAgent(config)

        mock_gossip = AsyncMock()
        mock_gossip.broadcast_consensus = AsyncMock()
        agent._gossip = mock_gossip

        await agent.commit_mappings(sample_records)

        mock_gossip.broadcast_consensus.assert_called_once()

    @pytest.mark.asyncio
    async def test_commit_mappings_handles_empty_list(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that commit_mappings handles empty list."""
        agent = SemanticAgent(config)

        await agent.commit_mappings([])

        assert len(agent._committed_mappings) == 0


class TestSemanticAgentHealthCheck:
    """Tests for SemanticAgent health monitoring."""

    @pytest.fixture
    def config(self) -> SemanticAgentConfig:
        """Create a sample config for testing."""
        return SemanticAgentConfig(
            agent_id="test-agent",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
        )

    def test_health_check_returns_dict(self, config: SemanticAgentConfig) -> None:
        """Test that health_check returns a dictionary."""
        agent = SemanticAgent(config)

        health = agent.health_check()

        assert isinstance(health, dict)

    def test_health_check_includes_agent_id(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that health_check includes agent_id."""
        agent = SemanticAgent(config)

        health = agent.health_check()

        assert health["agent_id"] == "test-agent"

    def test_health_check_includes_state(self, config: SemanticAgentConfig) -> None:
        """Test that health_check includes current state."""
        agent = SemanticAgent(config)

        health = agent.health_check()

        assert health["state"] == AgentState.INITIALIZING.value

    def test_health_check_includes_metrics(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that health_check includes all metrics."""
        agent = SemanticAgent(config)

        health = agent.health_check()

        # Verify all expected keys are present
        assert "agent_id" in health
        assert "state" in health
        assert "is_running" in health
        assert "tags_discovered" in health
        assert "hypotheses_generated" in health
        assert "votes_cast" in health
        assert "last_discovery_time" in health
        assert "last_inference_time" in health
        assert "last_gossip_time" in health

    def test_health_check_initial_counts_are_zero(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that initial metric counts are zero."""
        agent = SemanticAgent(config)

        health = agent.health_check()

        assert health["tags_discovered"] == 0
        assert health["hypotheses_generated"] == 0
        assert health["votes_cast"] == 0

    def test_health_check_initial_times_are_none(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that initial timestamps are None."""
        agent = SemanticAgent(config)

        health = agent.health_check()

        assert health["last_discovery_time"] is None
        assert health["last_inference_time"] is None
        assert health["last_gossip_time"] is None


class TestSemanticAgentContextManager:
    """Tests for SemanticAgent async context manager support."""

    @pytest.fixture
    def config(self) -> SemanticAgentConfig:
        """Create a sample config for testing."""
        return SemanticAgentConfig(
            agent_id="test-agent",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
        )

    @pytest.mark.asyncio
    async def test_async_context_manager_starts_and_stops(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that async context manager starts and stops agent."""
        agent = SemanticAgent(config)

        with patch.object(agent, "start", new_callable=AsyncMock) as mock_start, \
             patch.object(agent, "stop", new_callable=AsyncMock) as mock_stop:
            async with agent:
                mock_start.assert_called_once()

            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_stops_on_exception(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that async context manager stops even on exception."""
        agent = SemanticAgent(config)

        with patch.object(agent, "start", new_callable=AsyncMock), \
             patch.object(agent, "stop", new_callable=AsyncMock) as mock_stop:
            try:
                async with agent:
                    raise ValueError("Test exception")
            except ValueError:
                pass

            mock_stop.assert_called_once()


class TestSemanticAgentGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.fixture
    def config(self) -> SemanticAgentConfig:
        """Create a sample config with short timeout."""
        return SemanticAgentConfig(
            agent_id="test-agent",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
            shutdown_timeout_seconds=0.5,
        )

    @pytest.mark.asyncio
    async def test_graceful_shutdown_disconnects_mqtt(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that graceful shutdown disconnects from MQTT."""
        agent = SemanticAgent(config)

        mock_mqtt = AsyncMock()
        mock_mqtt.disconnect = AsyncMock()
        agent._mqtt_client = mock_mqtt
        agent._state = AgentState.RUNNING

        with patch.object(agent, "_disconnect_opcua", new_callable=AsyncMock), \
             patch.object(agent, "_stop_gossip", new_callable=AsyncMock):
            await agent.stop()

        mock_mqtt.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_respects_timeout(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test that shutdown respects timeout."""
        agent = SemanticAgent(config)
        agent._state = AgentState.RUNNING

        # Create a slow task
        slow_task = asyncio.create_task(asyncio.sleep(10))
        agent._lifecycle_task = slow_task

        with patch.object(agent, "_disconnect_mqtt", new_callable=AsyncMock), \
             patch.object(agent, "_disconnect_opcua", new_callable=AsyncMock), \
             patch.object(agent, "_stop_gossip", new_callable=AsyncMock):
            # Should complete within timeout (0.5 seconds)
            await asyncio.wait_for(agent.stop(), timeout=2.0)

        assert agent.state == AgentState.STOPPED
        assert slow_task.cancelled()


class TestSemanticAgentError:
    """Tests for SemanticAgentError exception."""

    def test_exception_message(self) -> None:
        """Test SemanticAgentError stores message."""
        error = SemanticAgentError("Test error message")

        assert str(error) == "Test error message"

    def test_exception_inheritance(self) -> None:
        """Test SemanticAgentError inherits from Exception."""
        error = SemanticAgentError("Test")

        assert isinstance(error, Exception)


class TestSemanticAgentIntegration:
    """Integration tests for complete agent workflows."""

    @pytest.fixture
    def config(self) -> SemanticAgentConfig:
        """Create a sample config for testing."""
        return SemanticAgentConfig(
            agent_id="integration-agent",
            opcua_endpoint="opc.tcp://localhost:4840",
            mqtt_host="localhost",
            poll_interval_seconds=0.1,
        )

    @pytest.mark.asyncio
    async def test_full_lifecycle_discover_infer_gossip_vote_commit(
        self, config: SemanticAgentConfig
    ) -> None:
        """Test complete lifecycle: discover -> infer -> gossip -> vote -> commit."""
        sample_tags = [
            TagRecord(
                node_id="ns=2;s=Temperature.PV",
                browse_name="Temperature_PV",
                display_name="Temperature Process Value",
                data_type="Double",
                source_server="opc.tcp://localhost:4840",
            ),
        ]

        sample_hypotheses = [
            Hypothesis(
                tag_id="opc.tcp://localhost:4840|ns=2;s=Temperature.PV",
                candidates=[
                    Candidate(
                        irdi="0173-1#01-ABA234#001",
                        confidence=0.95,
                        source_model="charcnn-v1",
                    ),
                ],
                agent_id="integration-agent",
            ),
        ]

        # Setup mocks
        mock_inference = MagicMock()
        mock_inference.infer = MagicMock(return_value=sample_hypotheses)

        mock_reputation = MagicMock()
        mock_reputation.get_reliability = MagicMock(return_value=0.9)

        agent = SemanticAgent(
            config,
            inference_engine=mock_inference,
            reputation_tracker=mock_reputation,
        )

        # Mock OPC UA browser
        mock_browser = AsyncMock()
        mock_browser.browse_all_tags = AsyncMock(return_value=sample_tags)
        agent._opcua_browser = mock_browser

        # Mock gossip
        mock_gossip = AsyncMock()
        mock_gossip.broadcast_hypothesis = AsyncMock()
        mock_gossip.broadcast_vote = AsyncMock()
        mock_gossip.broadcast_consensus = AsyncMock()
        agent._gossip = mock_gossip

        # Execute lifecycle steps: discover -> infer -> gossip -> vote -> commit
        tags = await agent.discover_tags()
        hypotheses = await agent.infer_mappings(tags)
        await agent.gossip_hypotheses(hypotheses)

        # Vote on discovered tags
        tag_ids = [h.tag_id for h in hypotheses]
        await agent.vote_on_tags(tag_ids)

        # Commit mappings (empty for now as consensus checking is not implemented)
        await agent.commit_mappings([])

        # Verify discovery, inference, and gossip
        assert len(tags) == 1
        assert len(hypotheses) == 1

        health = agent.health_check()
        assert health["tags_discovered"] == 1
        assert health["hypotheses_generated"] == 1
        assert health["last_discovery_time"] is not None
        assert health["last_inference_time"] is not None
        assert health["last_gossip_time"] is not None

        # Verify voting was called
        assert health["votes_cast"] == 1
        mock_gossip.broadcast_vote.assert_called_once()

        # Verify gossip methods were called correctly
        mock_gossip.broadcast_hypothesis.assert_called_once()

    def test_swarm_module_exports_agent(self) -> None:
        """Test that SemanticAgent can be imported from swarm module."""
        from noa_swarm.swarm import (
            AgentState,
            SemanticAgent,
            SemanticAgentConfig,
            SemanticAgentError,
        )

        assert SemanticAgent is not None
        assert SemanticAgentConfig is not None
        assert AgentState is not None
        assert SemanticAgentError is not None
