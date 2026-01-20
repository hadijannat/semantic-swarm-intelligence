"""Unit tests for SWIM membership protocol integration."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noa_swarm.swarm.membership import (
    SwarmAlreadyRunningError,
    SwarmMember,
    SwarmMembership,
    SwarmMembershipError,
    SwarmNotStartedError,
)


class TestSwarmMember:
    """Tests for SwarmMember dataclass."""

    def test_create_minimal_member(self) -> None:
        """Test creating SwarmMember with required fields only."""
        member = SwarmMember(
            agent_id="agent-001",
            host="192.168.1.100",
            port=7946,
        )

        assert member.agent_id == "agent-001"
        assert member.host == "192.168.1.100"
        assert member.port == 7946
        assert member.model_version is None
        assert member.capabilities == []
        assert member.status == "alive"
        assert member.joined_at is not None
        assert member.last_seen is not None

    def test_create_full_member(self) -> None:
        """Test creating SwarmMember with all fields."""
        joined = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        last_seen = datetime(2024, 1, 15, 10, 5, 0, tzinfo=UTC)

        member = SwarmMember(
            agent_id="agent-002",
            host="10.0.0.50",
            port=8000,
            model_version="semantic-v2.1",
            capabilities=["classification", "embedding", "reasoning"],
            joined_at=joined,
            last_seen=last_seen,
            status="suspect",
        )

        assert member.agent_id == "agent-002"
        assert member.host == "10.0.0.50"
        assert member.port == 8000
        assert member.model_version == "semantic-v2.1"
        assert member.capabilities == ["classification", "embedding", "reasoning"]
        assert member.joined_at == joined
        assert member.last_seen == last_seen
        assert member.status == "suspect"

    def test_member_address_property(self) -> None:
        """Test address property returns host:port format."""
        member = SwarmMember(
            agent_id="agent-001",
            host="192.168.1.100",
            port=7946,
        )

        assert member.address == "192.168.1.100:7946"

    def test_member_swim_name_property(self) -> None:
        """Test swim_name property returns SWIM format."""
        member = SwarmMember(
            agent_id="agent-001",
            host="192.168.1.100",
            port=7946,
        )

        assert member.swim_name == "agent-001@192.168.1.100:7946"

    def test_member_to_metadata(self) -> None:
        """Test converting member to SWIM metadata format."""
        member = SwarmMember(
            agent_id="agent-001",
            host="192.168.1.100",
            port=7946,
            model_version="v1.0",
            capabilities=["classify", "embed"],
        )

        metadata = member.to_metadata()

        assert "model_version" in metadata
        assert metadata["model_version"] == b"v1.0"
        assert "capabilities" in metadata
        assert metadata["capabilities"] == b'["classify", "embed"]'

    def test_member_to_metadata_empty(self) -> None:
        """Test converting member with no optional metadata."""
        member = SwarmMember(
            agent_id="agent-001",
            host="192.168.1.100",
            port=7946,
        )

        metadata = member.to_metadata()

        assert metadata == {}

    def test_member_update_last_seen(self) -> None:
        """Test update_last_seen creates new member with updated timestamp."""
        original = SwarmMember(
            agent_id="agent-001",
            host="192.168.1.100",
            port=7946,
        )
        original_last_seen = original.last_seen

        # Small delay to ensure timestamp difference
        import time

        time.sleep(0.01)

        updated = original.update_last_seen()

        assert updated is not original  # New instance
        assert updated.agent_id == original.agent_id
        assert updated.last_seen > original_last_seen

    def test_member_with_status(self) -> None:
        """Test with_status creates new member with changed status."""
        original = SwarmMember(
            agent_id="agent-001",
            host="192.168.1.100",
            port=7946,
            status="alive",
        )

        suspect = original.with_status("suspect")
        dead = original.with_status("dead")

        assert suspect is not original
        assert suspect.status == "suspect"
        assert suspect.agent_id == original.agent_id

        assert dead.status == "dead"
        assert original.status == "alive"  # Original unchanged

    def test_member_validation_empty_agent_id(self) -> None:
        """Test validation rejects empty agent_id."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            SwarmMember(
                agent_id="",
                host="192.168.1.100",
                port=7946,
            )

    def test_member_validation_empty_host(self) -> None:
        """Test validation rejects empty host."""
        with pytest.raises(ValueError, match="host cannot be empty"):
            SwarmMember(
                agent_id="agent-001",
                host="",
                port=7946,
            )

    def test_member_validation_allows_port_zero(self) -> None:
        """Test validation allows port 0 as sentinel for unknown."""
        member = SwarmMember(
            agent_id="agent-001",
            host="unknown",
            port=0,
        )
        assert member.port == 0

    def test_member_validation_invalid_port_negative(self) -> None:
        """Test validation rejects negative port."""
        with pytest.raises(ValueError, match="port must be between"):
            SwarmMember(
                agent_id="agent-001",
                host="192.168.1.100",
                port=-1,
            )

    def test_member_validation_invalid_port_too_high(self) -> None:
        """Test validation rejects port > 65535."""
        with pytest.raises(ValueError, match="port must be between"):
            SwarmMember(
                agent_id="agent-001",
                host="192.168.1.100",
                port=65536,
            )


class TestSwarmMemberFromSwim:
    """Tests for SwarmMember.from_swim_member classmethod."""

    def test_from_swim_member_full_name(self) -> None:
        """Test creating from SWIM member with full name format."""
        swim_member = MagicMock()
        swim_member.name = "agent-001@192.168.1.100:7946"
        swim_member.metadata = {
            "model_version": b"v1.0",
            "capabilities": b'["classify", "embed"]',
        }

        member = SwarmMember.from_swim_member(swim_member, "alive")

        assert member.agent_id == "agent-001"
        assert member.host == "192.168.1.100"
        assert member.port == 7946
        assert member.model_version == "v1.0"
        assert member.capabilities == ["classify", "embed"]
        assert member.status == "alive"

    def test_from_swim_member_simple_name(self) -> None:
        """Test creating from SWIM member with simple name."""
        swim_member = MagicMock()
        swim_member.name = "agent-002"
        swim_member.metadata = {}

        member = SwarmMember.from_swim_member(swim_member, "suspect")

        assert member.agent_id == "agent-002"
        assert member.host == "unknown"
        assert member.port == 0
        assert member.status == "suspect"

    def test_from_swim_member_no_metadata(self) -> None:
        """Test creating from SWIM member without metadata."""
        swim_member = MagicMock()
        swim_member.name = "agent-003@10.0.0.1:8080"
        swim_member.metadata = None

        member = SwarmMember.from_swim_member(swim_member, "dead")

        assert member.agent_id == "agent-003"
        assert member.model_version is None
        assert member.capabilities == []
        assert member.status == "dead"

    def test_from_swim_member_bytes_metadata(self) -> None:
        """Test creating from SWIM member with bytes metadata keys."""
        swim_member = MagicMock()
        swim_member.name = "agent-004@localhost:9000"
        swim_member.metadata = {
            b"model_version": b"semantic-v2",
            b"capabilities": b'["inference"]',
        }

        member = SwarmMember.from_swim_member(swim_member, "alive")

        assert member.model_version == "semantic-v2"
        assert member.capabilities == ["inference"]


class TestSwarmMembershipInitialization:
    """Tests for SwarmMembership initialization."""

    def test_create_membership(self) -> None:
        """Test creating SwarmMembership instance."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="192.168.1.100",
            port=7946,
        )

        assert membership.agent_id == "agent-001"
        assert membership.host == "192.168.1.100"
        assert membership.port == 7946
        assert membership.is_running is False
        assert membership.local_member is None

    def test_create_membership_with_all_options(self) -> None:
        """Test creating SwarmMembership with all options."""
        membership = SwarmMembership(
            agent_id="agent-002",
            host="10.0.0.1",
            port=8000,
            secret="my-secret",
            ping_interval=2.0,
            ping_timeout=0.5,
            suspect_timeout=10.0,
        )

        assert membership.agent_id == "agent-002"
        assert membership.is_running is False


class TestSwarmMembershipLifecycle:
    """Tests for SwarmMembership start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_initializes_local_member(self) -> None:
        """Test that start() initializes the local member."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        # Mock SWIM components to avoid actual network operations
        with patch(
            "noa_swarm.swarm.membership.SwarmMembership._run_worker", new_callable=AsyncMock
        ):
            with patch(
                "noa_swarm.swarm.membership.SwarmMembership._sync_members", new_callable=AsyncMock
            ):
                await membership.start()

                assert membership.is_running is True
                assert membership.local_member is not None
                assert membership.local_member.agent_id == "agent-001"
                assert membership.local_member.status == "alive"

                await membership.stop()

    @pytest.mark.asyncio
    async def test_start_twice_raises_error(self) -> None:
        """Test that starting twice raises SwarmAlreadyRunningError."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        with patch(
            "noa_swarm.swarm.membership.SwarmMembership._run_worker", new_callable=AsyncMock
        ):
            with patch(
                "noa_swarm.swarm.membership.SwarmMembership._sync_members", new_callable=AsyncMock
            ):
                await membership.start()

                with pytest.raises(SwarmAlreadyRunningError):
                    await membership.start()

                await membership.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self) -> None:
        """Test that stop() when not running logs warning but doesn't error."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        # Should not raise
        await membership.stop()
        assert membership.is_running is False

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self) -> None:
        """Test that stop() properly cleans up resources."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        with patch(
            "noa_swarm.swarm.membership.SwarmMembership._run_worker", new_callable=AsyncMock
        ):
            with patch(
                "noa_swarm.swarm.membership.SwarmMembership._sync_members", new_callable=AsyncMock
            ):
                await membership.start()
                assert membership.is_running is True

                await membership.stop()
                assert membership.is_running is False

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test using SwarmMembership as async context manager."""
        with patch(
            "noa_swarm.swarm.membership.SwarmMembership._run_worker", new_callable=AsyncMock
        ):
            with patch(
                "noa_swarm.swarm.membership.SwarmMembership._sync_members", new_callable=AsyncMock
            ):
                async with SwarmMembership(
                    agent_id="agent-001",
                    host="127.0.0.1",
                    port=7946,
                ) as membership:
                    assert membership.is_running is True
                    assert membership.local_member is not None

                assert membership.is_running is False


class TestSwarmMembershipOperations:
    """Tests for SwarmMembership member operations."""

    @pytest.fixture
    def membership(self) -> SwarmMembership:
        """Create a membership instance with mock members."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )
        # Add some mock members
        membership._members = {
            "agent-001": SwarmMember(
                agent_id="agent-001",
                host="127.0.0.1",
                port=7946,
                status="alive",
            ),
            "agent-002": SwarmMember(
                agent_id="agent-002",
                host="127.0.0.2",
                port=7946,
                status="alive",
            ),
            "agent-003": SwarmMember(
                agent_id="agent-003",
                host="127.0.0.3",
                port=7946,
                status="suspect",
            ),
            "agent-004": SwarmMember(
                agent_id="agent-004",
                host="127.0.0.4",
                port=7946,
                status="dead",
            ),
        }
        return membership

    def test_get_members(self, membership: SwarmMembership) -> None:
        """Test get_members returns all members."""
        members = membership.get_members()

        assert len(members) == 4
        agent_ids = {m.agent_id for m in members}
        assert agent_ids == {"agent-001", "agent-002", "agent-003", "agent-004"}

    def test_get_alive_members(self, membership: SwarmMembership) -> None:
        """Test get_alive_members returns only alive members."""
        alive = membership.get_alive_members()

        assert len(alive) == 2
        agent_ids = {m.agent_id for m in alive}
        assert agent_ids == {"agent-001", "agent-002"}

    def test_get_member_exists(self, membership: SwarmMembership) -> None:
        """Test get_member returns member when it exists."""
        member = membership.get_member("agent-002")

        assert member is not None
        assert member.agent_id == "agent-002"

    def test_get_member_not_exists(self, membership: SwarmMembership) -> None:
        """Test get_member returns None when member doesn't exist."""
        member = membership.get_member("agent-999")

        assert member is None


class TestSwarmMembershipMetadata:
    """Tests for SwarmMembership metadata operations."""

    def test_set_metadata_before_start(self) -> None:
        """Test setting metadata before starting."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        membership.set_metadata(
            model_version="v2.0",
            capabilities=["classify", "embed"],
        )

        # Metadata stored but not propagated until start
        assert membership._model_version == "v2.0"
        assert membership._capabilities == ["classify", "embed"]

    @pytest.mark.asyncio
    async def test_set_metadata_after_start(self) -> None:
        """Test setting metadata after starting updates local member."""
        with patch(
            "noa_swarm.swarm.membership.SwarmMembership._run_worker", new_callable=AsyncMock
        ):
            with patch(
                "noa_swarm.swarm.membership.SwarmMembership._sync_members", new_callable=AsyncMock
            ):
                async with SwarmMembership(
                    agent_id="agent-001",
                    host="127.0.0.1",
                    port=7946,
                ) as membership:
                    membership.set_metadata(
                        model_version="v3.0",
                        capabilities=["inference", "training"],
                    )

                    assert membership.local_member is not None
                    assert membership.local_member.model_version == "v3.0"
                    assert membership.local_member.capabilities == ["inference", "training"]

    def test_set_metadata_partial_update(self) -> None:
        """Test partial metadata update."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        membership.set_metadata(model_version="v1.0")
        assert membership._model_version == "v1.0"
        assert membership._capabilities == []

        membership.set_metadata(capabilities=["new-cap"])
        assert membership._model_version == "v1.0"  # Unchanged
        assert membership._capabilities == ["new-cap"]


class TestSwarmMembershipCallbacks:
    """Tests for SwarmMembership callback registration."""

    def test_on_member_join_registration(self) -> None:
        """Test registering join callback."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        async def callback(member: SwarmMember) -> None:
            pass

        result = membership.on_member_join(callback)

        assert result is membership  # Method chaining
        assert callback in membership._on_join_callbacks

    def test_on_member_leave_registration(self) -> None:
        """Test registering leave callback."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        async def callback(member: SwarmMember) -> None:
            pass

        result = membership.on_member_leave(callback)

        assert result is membership
        assert callback in membership._on_leave_callbacks

    def test_on_member_update_registration(self) -> None:
        """Test registering update callback."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        async def callback(member: SwarmMember) -> None:
            pass

        result = membership.on_member_update(callback)

        assert result is membership
        assert callback in membership._on_update_callbacks

    def test_callback_chaining(self) -> None:
        """Test method chaining with callbacks."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        async def join_cb(member: SwarmMember) -> None:
            pass

        async def leave_cb(member: SwarmMember) -> None:
            pass

        async def update_cb(member: SwarmMember) -> None:
            pass

        membership.on_member_join(join_cb).on_member_leave(leave_cb).on_member_update(update_cb)

        assert join_cb in membership._on_join_callbacks
        assert leave_cb in membership._on_leave_callbacks
        assert update_cb in membership._on_update_callbacks


class TestSwarmMembershipCallbackExecution:
    """Tests for callback execution."""

    @pytest.mark.asyncio
    async def test_fire_async_callback(self) -> None:
        """Test firing async callbacks."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        called_with: list[SwarmMember] = []

        async def async_callback(member: SwarmMember) -> None:
            called_with.append(member)

        membership._on_join_callbacks.append(async_callback)

        test_member = SwarmMember(
            agent_id="test",
            host="127.0.0.1",
            port=7946,
        )

        await membership._fire_callbacks(membership._on_join_callbacks, test_member)

        assert len(called_with) == 1
        assert called_with[0].agent_id == "test"

    @pytest.mark.asyncio
    async def test_fire_sync_callback(self) -> None:
        """Test firing sync callbacks."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        called_with: list[SwarmMember] = []

        def sync_callback(member: SwarmMember) -> None:
            called_with.append(member)

        membership._on_join_callbacks.append(sync_callback)

        test_member = SwarmMember(
            agent_id="test",
            host="127.0.0.1",
            port=7946,
        )

        await membership._fire_callbacks(membership._on_join_callbacks, test_member)

        assert len(called_with) == 1

    @pytest.mark.asyncio
    async def test_callback_error_handling(self) -> None:
        """Test that callback errors are caught and logged."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        async def failing_callback(member: SwarmMember) -> None:
            raise RuntimeError("Test error")

        successful_calls: list[SwarmMember] = []

        async def success_callback(member: SwarmMember) -> None:
            successful_calls.append(member)

        membership._on_join_callbacks.append(failing_callback)
        membership._on_join_callbacks.append(success_callback)

        test_member = SwarmMember(
            agent_id="test",
            host="127.0.0.1",
            port=7946,
        )

        # Should not raise
        await membership._fire_callbacks(membership._on_join_callbacks, test_member)

        # Second callback should still be called
        assert len(successful_calls) == 1


class TestSwarmMembershipJoin:
    """Tests for joining swarm operations."""

    @pytest.mark.asyncio
    async def test_join_not_started_raises_error(self) -> None:
        """Test that join() raises error when not started."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        with pytest.raises(SwarmNotStartedError):
            await membership.join([("192.168.1.100", 7946)])

    @pytest.mark.asyncio
    async def test_join_empty_seeds(self) -> None:
        """Test that join() with empty seeds logs warning."""
        with patch(
            "noa_swarm.swarm.membership.SwarmMembership._run_worker", new_callable=AsyncMock
        ):
            with patch(
                "noa_swarm.swarm.membership.SwarmMembership._sync_members", new_callable=AsyncMock
            ):
                async with SwarmMembership(
                    agent_id="agent-001",
                    host="127.0.0.1",
                    port=7946,
                ) as membership:
                    # Should not raise
                    await membership.join([])


class TestSwarmMembershipStatusConversion:
    """Tests for SWIM status conversion."""

    def test_swim_status_to_member_status_online(self) -> None:
        """Test converting ONLINE status."""
        from swimprotocol.status import Status as SwimStatus

        result = SwarmMembership._swim_status_to_member_status(SwimStatus.ONLINE)
        assert result == "alive"

    def test_swim_status_to_member_status_suspect(self) -> None:
        """Test converting SUSPECT status."""
        from swimprotocol.status import Status as SwimStatus

        result = SwarmMembership._swim_status_to_member_status(SwimStatus.SUSPECT)
        assert result == "suspect"

    def test_swim_status_to_member_status_offline(self) -> None:
        """Test converting OFFLINE status."""
        from swimprotocol.status import Status as SwimStatus

        result = SwarmMembership._swim_status_to_member_status(SwimStatus.OFFLINE)
        assert result == "dead"


class TestSwarmMembershipAgentIdExtraction:
    """Tests for agent ID extraction from SWIM names."""

    def test_extract_agent_id_full_name(self) -> None:
        """Test extracting agent ID from full SWIM name."""
        result = SwarmMembership._extract_agent_id("agent-001@192.168.1.100:7946")
        assert result == "agent-001"

    def test_extract_agent_id_simple_name(self) -> None:
        """Test extracting agent ID from simple name."""
        result = SwarmMembership._extract_agent_id("agent-002")
        assert result == "agent-002"

    def test_extract_agent_id_with_at_in_name(self) -> None:
        """Test extracting agent ID when @ appears multiple times."""
        result = SwarmMembership._extract_agent_id("agent@domain@host:port")
        assert result == "agent"


class TestSwarmMembershipExceptions:
    """Tests for custom exceptions."""

    def test_swarm_membership_error(self) -> None:
        """Test SwarmMembershipError base exception."""
        error = SwarmMembershipError("Test error message")
        assert str(error) == "Test error message"

    def test_swarm_not_started_error_default_message(self) -> None:
        """Test SwarmNotStartedError with default message."""
        error = SwarmNotStartedError()
        assert "not been started" in str(error)

    def test_swarm_not_started_error_custom_message(self) -> None:
        """Test SwarmNotStartedError with custom message."""
        error = SwarmNotStartedError("Custom message")
        assert str(error) == "Custom message"

    def test_swarm_already_running_error_default_message(self) -> None:
        """Test SwarmAlreadyRunningError with default message."""
        error = SwarmAlreadyRunningError()
        assert "already running" in str(error)

    def test_swarm_already_running_error_custom_message(self) -> None:
        """Test SwarmAlreadyRunningError with custom message."""
        error = SwarmAlreadyRunningError("Custom message")
        assert str(error) == "Custom message"


class TestSwarmMembershipBuildMetadata:
    """Tests for metadata building."""

    def test_build_metadata_empty(self) -> None:
        """Test building metadata with no values set."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )

        metadata = membership._build_metadata()
        assert metadata == {}

    def test_build_metadata_with_model_version(self) -> None:
        """Test building metadata with model version."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )
        membership._model_version = "v1.5"

        metadata = membership._build_metadata()
        assert metadata == {"model_version": b"v1.5"}

    def test_build_metadata_with_capabilities(self) -> None:
        """Test building metadata with capabilities."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )
        membership._capabilities = ["classify", "embed"]

        metadata = membership._build_metadata()
        assert "capabilities" in metadata
        assert metadata["capabilities"] == b'["classify", "embed"]'

    def test_build_metadata_full(self) -> None:
        """Test building metadata with all values."""
        membership = SwarmMembership(
            agent_id="agent-001",
            host="127.0.0.1",
            port=7946,
        )
        membership._model_version = "v2.0"
        membership._capabilities = ["inference"]

        metadata = membership._build_metadata()
        assert metadata["model_version"] == b"v2.0"
        assert metadata["capabilities"] == b'["inference"]'
