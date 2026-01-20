"""Unit tests for MQTT client wrapper."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from paho.mqtt.enums import MQTTErrorCode

from noa_swarm.connectors.mqtt import (
    MQTTClient,
    MQTTClientConfig,
    MQTTClientError,
    MQTTConnectionError,
    MQTTPublishError,
    MQTTSubscribeError,
    PendingPublish,
)


class TestMQTTClientConfig:
    """Tests for MQTTClientConfig dataclass."""

    def test_create_minimal_config(self) -> None:
        """Test creating config with required fields only."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )

        assert config.broker_host == "localhost"
        assert config.client_id == "test-client"
        assert config.broker_port == 1883
        assert config.username is None
        assert config.password is None
        assert config.use_tls is False
        assert config.keepalive == 60
        assert config.clean_session is True

    def test_create_full_config(self) -> None:
        """Test creating config with all fields."""
        config = MQTTClientConfig(
            broker_host="mqtt.example.com",
            broker_port=8883,
            client_id="agent-001",
            username="user",
            password="secret",
            use_tls=True,
            ca_cert_path=Path("/certs/ca.pem"),
            client_cert_path=Path("/certs/client.pem"),
            client_key_path=Path("/certs/client.key"),
            keepalive=120,
            clean_session=False,
        )

        assert config.broker_host == "mqtt.example.com"
        assert config.broker_port == 8883
        assert config.client_id == "agent-001"
        assert config.username == "user"
        assert config.password == "secret"
        assert config.use_tls is True
        assert config.ca_cert_path == Path("/certs/ca.pem")
        assert config.client_cert_path == Path("/certs/client.pem")
        assert config.client_key_path == Path("/certs/client.key")
        assert config.keepalive == 120
        assert config.clean_session is False


class TestMQTTClientInitialization:
    """Tests for MQTTClient initialization."""

    def test_create_client(self) -> None:
        """Test creating MQTTClient instance."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        assert client.client_id == "test-client"
        assert client.is_connected is False

    def test_client_properties(self) -> None:
        """Test client properties before connection."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="my-agent",
        )
        client = MQTTClient(config)

        assert client.client_id == "my-agent"
        assert client.is_connected is False


class TestMQTTClientConnect:
    """Tests for MQTTClient connect operations."""

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        """Test successful connection to broker."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        mock_paho = MagicMock()
        mock_paho.connect.return_value = MQTTErrorCode.MQTT_ERR_SUCCESS

        with patch("noa_swarm.connectors.mqtt.mqtt_client.Client", return_value=mock_paho):
            # Simulate connection callback
            async def complete_connection() -> None:
                await asyncio.sleep(0.01)
                # Simulate the on_connect callback being called
                client._connected = True
                if client._connect_future and not client._connect_future.done():
                    client._connect_future.set_result(None)

            connect_task = asyncio.create_task(client.connect())
            complete_task = asyncio.create_task(complete_connection())

            await asyncio.gather(connect_task, complete_task)

            assert client.is_connected is True
            mock_paho.loop_start.assert_called_once()

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_connect_already_connected(self) -> None:
        """Test that connecting when already connected logs warning."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)
        client._connected = True

        # Should not raise, just log warning
        await client.connect()

    @pytest.mark.asyncio
    async def test_connect_timeout(self) -> None:
        """Test connection timeout handling."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        mock_paho = MagicMock()
        mock_paho.connect.return_value = MQTTErrorCode.MQTT_ERR_SUCCESS

        with patch("noa_swarm.connectors.mqtt.mqtt_client.Client", return_value=mock_paho):
            with patch("asyncio.wait_for", side_effect=TimeoutError()):
                with pytest.raises(MQTTConnectionError, match="timeout"):
                    await client.connect()

    @pytest.mark.asyncio
    async def test_connect_failure(self) -> None:
        """Test connection failure handling."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        mock_paho = MagicMock()
        mock_paho.connect.return_value = MQTTErrorCode.MQTT_ERR_CONN_REFUSED

        with patch("noa_swarm.connectors.mqtt.mqtt_client.Client", return_value=mock_paho):
            with pytest.raises(MQTTConnectionError, match="Failed to initiate"):
                await client.connect()


class TestMQTTClientDisconnect:
    """Tests for MQTTClient disconnect operations."""

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self) -> None:
        """Test disconnect when not connected logs warning."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        # Should not raise
        await client.disconnect()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_success(self) -> None:
        """Test successful disconnection."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        mock_paho = MagicMock()
        client._client = mock_paho
        client._connected = True

        await client.disconnect()

        assert client.is_connected is False
        mock_paho.disconnect.assert_called_once()
        mock_paho.loop_stop.assert_called_once()


class TestMQTTClientPublish:
    """Tests for MQTTClient publish operations."""

    @pytest.mark.asyncio
    async def test_publish_not_connected_raises_error(self) -> None:
        """Test that publishing when not connected raises error."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        with pytest.raises(MQTTConnectionError, match="Not connected"):
            await client.publish("test/topic", b"test")

    @pytest.mark.asyncio
    async def test_publish_qos0_success(self) -> None:
        """Test publishing with QoS 0."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)
        client._connected = True
        client._loop = asyncio.get_event_loop()

        mock_paho = MagicMock()
        mock_result = MagicMock()
        mock_result.rc = MQTTErrorCode.MQTT_ERR_SUCCESS
        mock_result.mid = 1
        mock_paho.publish.return_value = mock_result
        client._client = mock_paho

        await client.publish("test/topic", b"test payload", qos=0)

        mock_paho.publish.assert_called_once_with("test/topic", b"test payload", qos=0, retain=False)

    @pytest.mark.asyncio
    async def test_publish_string_payload(self) -> None:
        """Test publishing with string payload converts to bytes."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)
        client._connected = True
        client._loop = asyncio.get_event_loop()

        mock_paho = MagicMock()
        mock_result = MagicMock()
        mock_result.rc = MQTTErrorCode.MQTT_ERR_SUCCESS
        mock_result.mid = 1
        mock_paho.publish.return_value = mock_result
        client._client = mock_paho

        await client.publish("test/topic", "string payload", qos=0)

        mock_paho.publish.assert_called_once_with(
            "test/topic", b"string payload", qos=0, retain=False
        )

    @pytest.mark.asyncio
    async def test_publish_with_retain(self) -> None:
        """Test publishing with retain flag."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)
        client._connected = True
        client._loop = asyncio.get_event_loop()

        mock_paho = MagicMock()
        mock_result = MagicMock()
        mock_result.rc = MQTTErrorCode.MQTT_ERR_SUCCESS
        mock_result.mid = 1
        mock_paho.publish.return_value = mock_result
        client._client = mock_paho

        await client.publish("test/topic", b"data", qos=0, retain=True)

        mock_paho.publish.assert_called_once_with("test/topic", b"data", qos=0, retain=True)

    @pytest.mark.asyncio
    async def test_publish_failure(self) -> None:
        """Test publish failure handling."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)
        client._connected = True
        client._loop = asyncio.get_event_loop()

        mock_paho = MagicMock()
        mock_result = MagicMock()
        mock_result.rc = MQTTErrorCode.MQTT_ERR_NO_CONN
        mock_paho.publish.return_value = mock_result
        client._client = mock_paho

        with pytest.raises(MQTTPublishError, match="Failed to publish"):
            await client.publish("test/topic", b"data")


class TestMQTTClientSubscribe:
    """Tests for MQTTClient subscribe operations."""

    @pytest.mark.asyncio
    async def test_subscribe_not_connected_raises_error(self) -> None:
        """Test that subscribing when not connected raises error."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        with pytest.raises(MQTTConnectionError, match="Not connected"):
            await client.subscribe("test/topic")

    @pytest.mark.asyncio
    async def test_subscribe_failure(self) -> None:
        """Test subscribe failure handling."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)
        client._connected = True
        client._loop = asyncio.get_event_loop()

        mock_paho = MagicMock()
        mock_paho.subscribe.return_value = (MQTTErrorCode.MQTT_ERR_NO_CONN, 1)
        client._client = mock_paho

        with pytest.raises(MQTTSubscribeError, match="Failed to subscribe"):
            await client.subscribe("test/topic")


class TestMQTTClientUnsubscribe:
    """Tests for MQTTClient unsubscribe operations."""

    @pytest.mark.asyncio
    async def test_unsubscribe_not_connected_raises_error(self) -> None:
        """Test that unsubscribing when not connected raises error."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        with pytest.raises(MQTTConnectionError, match="Not connected"):
            await client.unsubscribe("test/topic")


class TestMQTTClientCallbacks:
    """Tests for MQTTClient callback registration."""

    def test_on_message_callback_registration(self) -> None:
        """Test registering message callback."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        async def callback(topic: str, payload: bytes) -> None:
            pass

        result = client.on_message(callback)

        assert result is client  # Method chaining
        assert callback in client._message_callbacks

    def test_on_connect_callback_registration(self) -> None:
        """Test registering connect callback."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        async def callback() -> None:
            pass

        result = client.on_connect(callback)

        assert result is client
        assert callback in client._connect_callbacks

    def test_on_disconnect_callback_registration(self) -> None:
        """Test registering disconnect callback."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        async def callback(reason: int | None) -> None:
            pass

        result = client.on_disconnect(callback)

        assert result is client
        assert callback in client._disconnect_callbacks

    def test_callback_chaining(self) -> None:
        """Test method chaining with callbacks."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        async def msg_cb(topic: str, payload: bytes) -> None:
            pass

        async def connect_cb() -> None:
            pass

        async def disconnect_cb(reason: int | None) -> None:
            pass

        client.on_message(msg_cb).on_connect(connect_cb).on_disconnect(disconnect_cb)

        assert msg_cb in client._message_callbacks
        assert connect_cb in client._connect_callbacks
        assert disconnect_cb in client._disconnect_callbacks


class TestMQTTClientContextManager:
    """Tests for MQTTClient async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_connect_disconnect(self) -> None:
        """Test context manager calls connect and disconnect."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)

        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(client, "disconnect", new_callable=AsyncMock) as mock_disconnect:
                async with client as ctx:
                    assert ctx is client

                mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()


class TestMQTTClientPahoCallbacks:
    """Tests for paho-mqtt callback handling."""

    @pytest.mark.asyncio
    async def test_on_connect_success_sets_connected(self) -> None:
        """Test on_connect callback sets connected state."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)
        client._loop = asyncio.get_event_loop()
        client._connect_future = asyncio.get_event_loop().create_future()

        # Create mock reason code that indicates success
        mock_reason_code = MagicMock()
        mock_reason_code.__eq__ = lambda self, other: other == 0
        mock_reason_code.is_failure = False

        mock_flags = MagicMock()

        # Call the on_connect callback
        client._on_connect(MagicMock(), None, mock_flags, mock_reason_code, None)

        assert client._connected is True

    @pytest.mark.asyncio
    async def test_on_disconnect_sets_not_connected(self) -> None:
        """Test on_disconnect callback sets not connected state."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)
        client._connected = True
        client._loop = asyncio.get_event_loop()

        mock_flags = MagicMock()
        mock_reason_code = MagicMock()

        client._on_disconnect(MagicMock(), None, mock_flags, mock_reason_code, None)

        assert client._connected is False

    @pytest.mark.asyncio
    async def test_on_publish_resolves_future(self) -> None:
        """Test on_publish callback resolves pending future."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
        )
        client = MQTTClient(config)
        client._loop = asyncio.get_event_loop()

        # Create a pending publish
        future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        client._pending_publishes[123] = PendingPublish(topic="test", future=future)

        mock_reason_code = MagicMock()

        # Call the on_publish callback
        client._on_publish(MagicMock(), None, 123, mock_reason_code, None)

        # Give the event loop a chance to process
        await asyncio.sleep(0.01)

        # Pending publish should be removed
        assert 123 not in client._pending_publishes


class TestMQTTClientExceptions:
    """Tests for MQTT client exceptions."""

    def test_mqtt_client_error(self) -> None:
        """Test MQTTClientError base exception."""
        error = MQTTClientError("Test error")
        assert str(error) == "Test error"

    def test_mqtt_connection_error(self) -> None:
        """Test MQTTConnectionError exception."""
        error = MQTTConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, MQTTClientError)

    def test_mqtt_publish_error(self) -> None:
        """Test MQTTPublishError exception."""
        error = MQTTPublishError("Publish failed")
        assert str(error) == "Publish failed"
        assert isinstance(error, MQTTClientError)

    def test_mqtt_subscribe_error(self) -> None:
        """Test MQTTSubscribeError exception."""
        error = MQTTSubscribeError("Subscribe failed")
        assert str(error) == "Subscribe failed"
        assert isinstance(error, MQTTClientError)


class TestPendingPublish:
    """Tests for PendingPublish dataclass."""

    @pytest.mark.asyncio
    async def test_pending_publish_creation(self) -> None:
        """Test creating PendingPublish instance."""
        pending = PendingPublish(topic="test/topic")

        assert pending.topic == "test/topic"
        assert pending.future is not None
        assert not pending.future.done()


class TestMQTTClientTLS:
    """Tests for MQTT client TLS configuration."""

    def test_configure_tls_with_ca_only(self) -> None:
        """Test TLS configuration with CA certificate only."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
            use_tls=True,
            ca_cert_path=Path("/certs/ca.pem"),
        )
        client = MQTTClient(config)

        mock_paho = MagicMock()
        client._client = mock_paho

        client._configure_tls()

        mock_paho.tls_set.assert_called_once()
        call_args = mock_paho.tls_set.call_args
        assert call_args.kwargs["ca_certs"] == "/certs/ca.pem"
        assert call_args.kwargs["certfile"] is None
        assert call_args.kwargs["keyfile"] is None

    def test_configure_tls_with_client_certs(self) -> None:
        """Test TLS configuration with client certificates."""
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="test-client",
            use_tls=True,
            ca_cert_path=Path("/certs/ca.pem"),
            client_cert_path=Path("/certs/client.pem"),
            client_key_path=Path("/certs/client.key"),
        )
        client = MQTTClient(config)

        mock_paho = MagicMock()
        client._client = mock_paho

        client._configure_tls()

        mock_paho.tls_set.assert_called_once()
        call_args = mock_paho.tls_set.call_args
        assert call_args.kwargs["ca_certs"] == "/certs/ca.pem"
        assert call_args.kwargs["certfile"] == "/certs/client.pem"
        assert call_args.kwargs["keyfile"] == "/certs/client.key"
