"""MQTT Client Wrapper for NOA Semantic Swarm Mapper.

This module provides an async MQTT client wrapper using paho-mqtt library
for swarm agent communication via pub/sub messaging.

Features:
- Async connect/disconnect lifecycle
- Publish/subscribe operations with QoS support
- Message callbacks for incoming messages
- Async context manager support
- TLS/SSL support
"""

from __future__ import annotations

import asyncio
import ssl
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from loguru import logger
from paho.mqtt import client as mqtt_client
from paho.mqtt.enums import CallbackAPIVersion, MQTTErrorCode
from paho.mqtt.reasoncodes import ReasonCode

if TYPE_CHECKING:
    from pathlib import Path

    from paho.mqtt.properties import Properties


class MQTTClientError(Exception):
    """Base exception for MQTT client errors."""

    pass


class MQTTConnectionError(MQTTClientError):
    """Exception raised when connection to MQTT broker fails."""

    pass


class MQTTPublishError(MQTTClientError):
    """Exception raised when publishing a message fails."""

    pass


class MQTTSubscribeError(MQTTClientError):
    """Exception raised when subscribing to a topic fails."""

    pass


# Type alias for message callback
MessageCallback = Callable[[str, bytes], Awaitable[None] | None]
ConnectCallback = Callable[[], Awaitable[None] | None]
DisconnectCallback = Callable[[int | ReasonCode | None], Awaitable[None] | None]


@dataclass
class MQTTClientConfig:
    """Configuration for MQTT client connection.

    Attributes:
        broker_host: MQTT broker hostname or IP address.
        broker_port: MQTT broker port number.
        client_id: Unique client identifier.
        username: Optional username for authentication.
        password: Optional password for authentication.
        use_tls: Enable TLS/SSL connection.
        ca_cert_path: Path to CA certificate for TLS.
        client_cert_path: Path to client certificate for mTLS.
        client_key_path: Path to client private key for mTLS.
        keepalive: Keepalive interval in seconds.
        clean_session: Start with a clean session.
    """

    broker_host: str
    client_id: str
    broker_port: int = 1883
    username: str | None = None
    password: str | None = None
    use_tls: bool = False
    ca_cert_path: Path | None = None
    client_cert_path: Path | None = None
    client_key_path: Path | None = None
    keepalive: int = 60
    clean_session: bool = True


@dataclass
class PendingPublish:
    """Tracks a pending publish operation."""

    topic: str
    future: asyncio.Future[None] = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class MQTTClient:
    """Async MQTT client wrapper using paho-mqtt.

    This client provides an async interface to the paho-mqtt library,
    supporting publish/subscribe operations with callback registration.

    Usage:
        config = MQTTClientConfig(
            broker_host="localhost",
            client_id="agent-001",
        )
        async with MQTTClient(config) as client:
            await client.subscribe("noa/agents/#")
            await client.publish("noa/agents/status", b"online")
    """

    def __init__(self, config: MQTTClientConfig) -> None:
        """Initialize the MQTT client.

        Args:
            config: MQTT client configuration.
        """
        self._config = config
        self._client: mqtt_client.Client | None = None
        self._connected = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # Callbacks
        self._message_callbacks: list[MessageCallback] = []
        self._connect_callbacks: list[ConnectCallback] = []
        self._disconnect_callbacks: list[DisconnectCallback] = []

        # Pending operations tracking
        self._pending_publishes: dict[int, PendingPublish] = {}
        self._pending_subscribes: dict[int, asyncio.Future[None]] = {}
        self._pending_unsubscribes: dict[int, asyncio.Future[None]] = {}

        # Connection future for async connect
        self._connect_future: asyncio.Future[None] | None = None

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Return True if connected to the broker."""
        return self._connected

    @property
    def client_id(self) -> str:
        """Return the client ID."""
        return self._config.client_id

    async def connect(self) -> None:
        """Connect to the MQTT broker.

        Raises:
            MQTTConnectionError: If connection fails.
        """
        if self._connected:
            logger.warning(f"MQTT client {self._config.client_id} already connected")
            return

        try:
            self._loop = asyncio.get_running_loop()

            # Create paho MQTT client with callback API version
            self._client = mqtt_client.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=self._config.client_id,
                clean_session=self._config.clean_session,
            )

            # Set up callbacks
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message
            self._client.on_publish = self._on_publish
            self._client.on_subscribe = self._on_subscribe
            self._client.on_unsubscribe = self._on_unsubscribe

            # Configure authentication
            if self._config.username:
                self._client.username_pw_set(
                    self._config.username,
                    self._config.password,
                )

            # Configure TLS
            if self._config.use_tls:
                self._configure_tls()

            # Create connection future
            self._connect_future = self._loop.create_future()

            # Start the network loop in a background thread
            self._client.loop_start()

            # Initiate connection
            result = self._client.connect(
                host=self._config.broker_host,
                port=self._config.broker_port,
                keepalive=self._config.keepalive,
            )

            if result != MQTTErrorCode.MQTT_ERR_SUCCESS:
                raise MQTTConnectionError(
                    f"Failed to initiate connection to {self._config.broker_host}:{self._config.broker_port}"
                )

            # Wait for connection to complete
            try:
                await asyncio.wait_for(self._connect_future, timeout=30.0)
            except TimeoutError as e:
                self._client.loop_stop()
                raise MQTTConnectionError(
                    f"Connection timeout to {self._config.broker_host}:{self._config.broker_port}"
                ) from e

            logger.info(
                f"MQTT client {self._config.client_id} connected to "
                f"{self._config.broker_host}:{self._config.broker_port}"
            )

        except MQTTConnectionError:
            raise
        except Exception as e:
            if self._client:
                self._client.loop_stop()
            raise MQTTConnectionError(
                f"Failed to connect to {self._config.broker_host}:{self._config.broker_port}: {e}"
            ) from e

    def _configure_tls(self) -> None:
        """Configure TLS/SSL for the connection."""
        if not self._client:
            return

        ca_certs = str(self._config.ca_cert_path) if self._config.ca_cert_path else None
        certfile = str(self._config.client_cert_path) if self._config.client_cert_path else None
        keyfile = str(self._config.client_key_path) if self._config.client_key_path else None

        self._client.tls_set(
            ca_certs=ca_certs,
            certfile=certfile,
            keyfile=keyfile,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )

    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker gracefully."""
        if not self._connected or not self._client:
            logger.warning(f"MQTT client {self._config.client_id} not connected")
            return

        try:
            self._client.disconnect()
            self._client.loop_stop()
            self._connected = False
            logger.info(f"MQTT client {self._config.client_id} disconnected")
        except Exception as e:
            logger.warning(f"Error during MQTT disconnect: {e}")
        finally:
            self._client = None

    async def publish(
        self,
        topic: str,
        payload: bytes | str,
        qos: int = 1,
        retain: bool = False,
    ) -> None:
        """Publish a message to a topic.

        Args:
            topic: The topic to publish to.
            payload: The message payload (bytes or string).
            qos: Quality of Service level (0, 1, or 2).
            retain: Whether to retain the message on the broker.

        Raises:
            MQTTConnectionError: If not connected.
            MQTTPublishError: If publishing fails.
        """
        if not self._connected or not self._client:
            raise MQTTConnectionError("Not connected to MQTT broker")

        # Convert string payload to bytes
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        try:
            # Publish and track the message
            result = self._client.publish(topic, payload, qos=qos, retain=retain)

            if result.rc != MQTTErrorCode.MQTT_ERR_SUCCESS:
                raise MQTTPublishError(f"Failed to publish to {topic}: {result.rc}")

            # For QoS > 0, wait for acknowledgment
            if qos > 0 and self._loop:
                future: asyncio.Future[None] = self._loop.create_future()
                self._pending_publishes[result.mid] = PendingPublish(topic=topic, future=future)

                try:
                    await asyncio.wait_for(future, timeout=30.0)
                except TimeoutError as e:
                    self._pending_publishes.pop(result.mid, None)
                    raise MQTTPublishError(f"Publish timeout for topic {topic}") from e

            logger.debug(f"Published to {topic} (QoS={qos}, retain={retain})")

        except MQTTPublishError:
            raise
        except Exception as e:
            raise MQTTPublishError(f"Failed to publish to {topic}: {e}") from e

    async def subscribe(self, topic: str, qos: int = 1) -> None:
        """Subscribe to a topic.

        Args:
            topic: The topic pattern to subscribe to (supports wildcards).
            qos: Quality of Service level (0, 1, or 2).

        Raises:
            MQTTConnectionError: If not connected.
            MQTTSubscribeError: If subscription fails.
        """
        if not self._connected or not self._client:
            raise MQTTConnectionError("Not connected to MQTT broker")

        try:
            result, mid = self._client.subscribe(topic, qos)

            if result != MQTTErrorCode.MQTT_ERR_SUCCESS:
                raise MQTTSubscribeError(f"Failed to subscribe to {topic}: {result}")

            # Wait for subscription acknowledgment
            if self._loop:
                future: asyncio.Future[None] = self._loop.create_future()
                self._pending_subscribes[mid] = future

                try:
                    await asyncio.wait_for(future, timeout=30.0)
                except TimeoutError as e:
                    self._pending_subscribes.pop(mid, None)
                    raise MQTTSubscribeError(f"Subscribe timeout for topic {topic}") from e

            logger.debug(f"Subscribed to {topic} (QoS={qos})")

        except MQTTSubscribeError:
            raise
        except Exception as e:
            raise MQTTSubscribeError(f"Failed to subscribe to {topic}: {e}") from e

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic.

        Args:
            topic: The topic to unsubscribe from.

        Raises:
            MQTTConnectionError: If not connected.
        """
        if not self._connected or not self._client:
            raise MQTTConnectionError("Not connected to MQTT broker")

        try:
            result, mid = self._client.unsubscribe(topic)

            if result != MQTTErrorCode.MQTT_ERR_SUCCESS:
                logger.warning(f"Failed to unsubscribe from {topic}: {result}")
                return

            # Wait for unsubscribe acknowledgment
            if self._loop:
                future: asyncio.Future[None] = self._loop.create_future()
                self._pending_unsubscribes[mid] = future

                try:
                    await asyncio.wait_for(future, timeout=30.0)
                except TimeoutError:
                    self._pending_unsubscribes.pop(mid, None)
                    logger.warning(f"Unsubscribe timeout for topic {topic}")
                    return

            logger.debug(f"Unsubscribed from {topic}")

        except Exception as e:
            logger.warning(f"Error unsubscribing from {topic}: {e}")

    def on_message(self, callback: MessageCallback) -> Self:
        """Register a message callback.

        The callback will be invoked for each message received on subscribed topics.

        Args:
            callback: Async or sync function that takes (topic, payload) parameters.

        Returns:
            Self for method chaining.
        """
        self._message_callbacks.append(callback)
        return self

    def on_connect(self, callback: ConnectCallback) -> Self:
        """Register a connect callback.

        The callback will be invoked when the client connects to the broker.

        Args:
            callback: Async or sync function with no parameters.

        Returns:
            Self for method chaining.
        """
        self._connect_callbacks.append(callback)
        return self

    def on_disconnect(self, callback: DisconnectCallback) -> Self:
        """Register a disconnect callback.

        The callback will be invoked when the client disconnects from the broker.

        Args:
            callback: Async or sync function that takes reason code parameter.

        Returns:
            Self for method chaining.
        """
        self._disconnect_callbacks.append(callback)
        return self

    # =========================================================================
    # PAHO CALLBACKS (called from network thread)
    # =========================================================================

    def _on_connect(
        self,
        _client: mqtt_client.Client,
        _userdata: Any,
        _flags: mqtt_client.ConnectFlags,
        reason_code: ReasonCode,
        _properties: Properties | None,
    ) -> None:
        """Handle connection callback from paho-mqtt."""
        if reason_code == 0 or reason_code.is_failure is False:
            self._connected = True
            if self._connect_future and not self._connect_future.done() and self._loop:
                self._loop.call_soon_threadsafe(self._connect_future.set_result, None)

            # Fire connect callbacks (schedule coroutine via run_coroutine_threadsafe)
            if self._loop:
                for callback in self._connect_callbacks:
                    asyncio.run_coroutine_threadsafe(
                        self._fire_callback(callback),
                        self._loop,
                    )
        else:
            self._connected = False
            if self._connect_future and not self._connect_future.done() and self._loop:
                exc = MQTTConnectionError(f"Connection refused: {reason_code}")
                self._loop.call_soon_threadsafe(self._connect_future.set_exception, exc)

    def _on_disconnect(
        self,
        _client: mqtt_client.Client,
        _userdata: Any,
        _disconnect_flags: mqtt_client.DisconnectFlags,
        reason_code: ReasonCode,
        _properties: Properties | None,
    ) -> None:
        """Handle disconnection callback from paho-mqtt."""
        self._connected = False
        logger.info(f"MQTT client disconnected: {reason_code}")

        # Fire disconnect callbacks (schedule coroutine via run_coroutine_threadsafe)
        if self._loop:
            for callback in self._disconnect_callbacks:
                asyncio.run_coroutine_threadsafe(
                    self._fire_disconnect_callback(callback, reason_code),
                    self._loop,
                )

    def _on_message(
        self,
        _client: mqtt_client.Client,
        _userdata: Any,
        message: mqtt_client.MQTTMessage,
    ) -> None:
        """Handle incoming message callback from paho-mqtt."""
        if self._loop:
            for callback in self._message_callbacks:
                asyncio.run_coroutine_threadsafe(
                    self._fire_message_callback(callback, message.topic, message.payload),
                    self._loop,
                )

    def _on_publish(
        self,
        _client: mqtt_client.Client,
        _userdata: Any,
        mid: int,
        _reason_code: ReasonCode,
        _properties: Properties | None,
    ) -> None:
        """Handle publish acknowledgment callback from paho-mqtt."""
        pending = self._pending_publishes.pop(mid, None)
        if pending and not pending.future.done() and self._loop:
            self._loop.call_soon_threadsafe(pending.future.set_result, None)

    def _on_subscribe(
        self,
        _client: mqtt_client.Client,
        _userdata: Any,
        mid: int,
        reason_codes: list[ReasonCode],
        _properties: Properties | None,
    ) -> None:
        """Handle subscribe acknowledgment callback from paho-mqtt."""
        future = self._pending_subscribes.pop(mid, None)
        if future and not future.done() and self._loop:
            # Check if subscription was successful
            if all(not rc.is_failure for rc in reason_codes):
                self._loop.call_soon_threadsafe(future.set_result, None)
            else:
                exc = MQTTSubscribeError(f"Subscribe failed: {reason_codes}")
                self._loop.call_soon_threadsafe(future.set_exception, exc)

    def _on_unsubscribe(
        self,
        _client: mqtt_client.Client,
        _userdata: Any,
        mid: int,
        _reason_codes: list[ReasonCode],
        _properties: Properties | None,
    ) -> None:
        """Handle unsubscribe acknowledgment callback from paho-mqtt."""
        future = self._pending_unsubscribes.pop(mid, None)
        if future and not future.done() and self._loop:
            self._loop.call_soon_threadsafe(future.set_result, None)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    async def _fire_callback(self, callback: ConnectCallback) -> None:
        """Fire a connect callback, handling both sync and async."""
        try:
            result = callback()
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning(f"Connect callback error: {e}")

    async def _fire_disconnect_callback(
        self,
        callback: DisconnectCallback,
        reason_code: int | ReasonCode | None,
    ) -> None:
        """Fire a disconnect callback, handling both sync and async."""
        try:
            result = callback(reason_code)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning(f"Disconnect callback error: {e}")

    async def _fire_message_callback(
        self,
        callback: MessageCallback,
        topic: str,
        payload: bytes,
    ) -> None:
        """Fire a message callback, handling both sync and async."""
        try:
            result = callback(topic, payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning(f"Message callback error for topic {topic}: {e}")

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    async def __aenter__(self) -> Self:
        """Enter async context manager and connect."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context manager and disconnect."""
        await self.disconnect()
