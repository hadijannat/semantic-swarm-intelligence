"""OPC UA Browser using asyncua library.

This module provides a read-only OPC UA browser for discovering tags
from OPC UA servers. Write operations are explicitly prohibited.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Self

from asyncua import Client, Node, ua
from asyncua.common.node import Node as NodeType
from loguru import logger

from noa_swarm.common.config import OPCUASettings
from noa_swarm.common.schemas import TagRecord

if TYPE_CHECKING:
    from collections.abc import Sequence


class OPCUABrowserError(Exception):
    """Base exception for OPC UA browser errors."""

    pass


class OPCUAWriteAttemptError(OPCUABrowserError):
    """Exception raised when a write operation is attempted.

    This browser is READ-ONLY. Write operations are explicitly prohibited.
    """

    def __init__(self, message: str = "Write operations are not allowed") -> None:
        super().__init__(message)


class OPCUAConnectionError(OPCUABrowserError):
    """Exception raised when connection to OPC UA server fails."""

    pass


class OPCUABrowseError(OPCUABrowserError):
    """Exception raised when browsing operations fail."""

    pass


@dataclass
class BrowseResult:
    """Result of browsing a single node."""

    node: NodeType
    node_id: str
    browse_name: str
    display_name: str
    node_class: ua.NodeClass
    data_type: str | None = None
    description: str | None = None
    engineering_unit: str | None = None
    access_level: int | None = None
    parent_path: list[str] = field(default_factory=list)
    irdi: str | None = None  # From HasDictionaryEntry reference


# Reference type ID for HasDictionaryEntry (commonly used for semantic linkage)
HAS_DICTIONARY_ENTRY_ID = ua.NodeId(ua.ObjectIds.HasDictionaryEntry)


class OPCUABrowser:
    """Async OPC UA browser for tag discovery.

    This browser is READ-ONLY. All write methods are intentionally omitted,
    and any attempt to write will raise an OPCUAWriteAttemptError.

    Usage:
        async with OPCUABrowser(endpoint_url) as browser:
            tags = await browser.browse_all_tags()
            for tag in tags:
                print(tag.node_id, tag.browse_name)
    """

    def __init__(
        self,
        endpoint_url: str,
        settings: OPCUASettings | None = None,
    ) -> None:
        """Initialize the OPC UA browser.

        Args:
            endpoint_url: OPC UA server endpoint URL.
            settings: Optional OPC UA settings. If not provided, defaults are used.
        """
        self._endpoint_url = endpoint_url
        self._settings = settings or OPCUASettings()
        self._client: Client | None = None
        self._connected = False
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent_requests)
        self._request_timeout = self._settings.request_timeout

    @property
    def endpoint_url(self) -> str:
        """Return the OPC UA server endpoint URL."""
        return self._endpoint_url

    @property
    def is_connected(self) -> bool:
        """Return True if connected to the server."""
        return self._connected

    async def __aenter__(self) -> Self:
        """Enter async context manager and connect to server."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        """Exit async context manager and disconnect from server."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to the OPC UA server.

        Raises:
            OPCUAConnectionError: If connection fails.
        """
        if self._connected:
            logger.warning("Already connected to OPC UA server")
            return

        try:
            self._client = Client(url=self._endpoint_url, timeout=self._request_timeout)

            # Configure security if specified
            if self._settings.security_policy != "None":
                await self._configure_security()

            # Configure authentication if credentials provided
            if self._settings.username:
                password = (
                    self._settings.password.get_secret_value()
                    if self._settings.password
                    else ""
                )
                self._client.set_user(self._settings.username)
                self._client.set_password(password)

            await self._client.connect()
            self._connected = True
            logger.info(f"Connected to OPC UA server: {self._endpoint_url}")

        except Exception as e:
            self._client = None
            self._connected = False
            raise OPCUAConnectionError(f"Failed to connect to {self._endpoint_url}: {e}") from e

    async def _configure_security(self) -> None:
        """Configure security settings for the client."""
        if self._client is None:
            return

        policy_map = {
            "Basic256Sha256": ua.SecurityPolicyType.Basic256Sha256,
            "Aes128_Sha256_RsaOaep": ua.SecurityPolicyType.Aes128_Sha256_RsaOaep,
        }
        mode_map = {
            "Sign": ua.MessageSecurityMode.Sign,
            "SignAndEncrypt": ua.MessageSecurityMode.SignAndEncrypt,
        }

        policy = policy_map.get(self._settings.security_policy)
        mode = mode_map.get(self._settings.security_mode)

        if policy and mode:
            cert_path = str(self._settings.certificate_path) if self._settings.certificate_path else None
            key_path = str(self._settings.private_key_path) if self._settings.private_key_path else None
            await self._client.set_security(policy, cert_path, key_path, mode=mode)

    async def disconnect(self) -> None:
        """Disconnect from the OPC UA server."""
        if self._client and self._connected:
            try:
                await self._client.disconnect()
                logger.info(f"Disconnected from OPC UA server: {self._endpoint_url}")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._connected = False
                self._client = None

    def _ensure_connected(self) -> Client:
        """Ensure client is connected and return it.

        Raises:
            OPCUAConnectionError: If not connected.
        """
        if not self._connected or self._client is None:
            raise OPCUAConnectionError("Not connected to OPC UA server")
        return self._client

    # =========================================================================
    # READ-ONLY ENFORCEMENT: These methods are intentionally NOT implemented
    # =========================================================================

    async def write_value(self, *args: object, **kwargs: object) -> None:
        """PROHIBITED: Write operations are not allowed.

        Raises:
            OPCUAWriteAttemptError: Always raised.
        """
        raise OPCUAWriteAttemptError("write_value is not allowed - this browser is read-only")

    async def write_attribute(self, *args: object, **kwargs: object) -> None:
        """PROHIBITED: Write operations are not allowed.

        Raises:
            OPCUAWriteAttemptError: Always raised.
        """
        raise OPCUAWriteAttemptError("write_attribute is not allowed - this browser is read-only")

    async def call_method(self, *args: object, **kwargs: object) -> None:
        """PROHIBITED: Method calls that could modify state are not allowed.

        Raises:
            OPCUAWriteAttemptError: Always raised.
        """
        raise OPCUAWriteAttemptError("call_method is not allowed - this browser is read-only")

    # =========================================================================
    # BROWSING METHODS
    # =========================================================================

    async def browse_all_tags(
        self,
        start_node: str | None = None,
        max_depth: int = 10,
    ) -> list[TagRecord]:
        """Browse and discover all tags from the server.

        Args:
            start_node: Starting node ID (defaults to Objects folder).
            max_depth: Maximum depth to browse.

        Returns:
            List of discovered TagRecord instances.
        """
        client = self._ensure_connected()

        if start_node:
            root = client.get_node(start_node)
        else:
            root = client.nodes.objects

        results: list[BrowseResult] = []
        await self._browse_recursive(root, [], results, max_depth, 0)

        # Convert BrowseResult to TagRecord
        tags = []
        for result in results:
            if result.node_class == ua.NodeClass.Variable:
                tag = TagRecord(
                    node_id=result.node_id,
                    browse_name=result.browse_name,
                    display_name=result.display_name,
                    data_type=result.data_type,
                    description=result.description,
                    parent_path=result.parent_path,
                    source_server=self._endpoint_url,
                    engineering_unit=result.engineering_unit,
                    access_level=result.access_level,
                )
                tags.append(tag)

        logger.info(f"Discovered {len(tags)} tags from {self._endpoint_url}")
        return tags

    async def _browse_recursive(
        self,
        node: NodeType,
        parent_path: list[str],
        results: list[BrowseResult],
        max_depth: int,
        current_depth: int,
    ) -> None:
        """Recursively browse the address space.

        Args:
            node: Current node to browse.
            parent_path: Path from root to current node.
            results: List to append results to.
            max_depth: Maximum depth to browse.
            current_depth: Current depth level.
        """
        if current_depth > max_depth:
            return

        async with self._semaphore:  # Concurrency control
            try:
                # Get child nodes
                children = await asyncio.wait_for(
                    node.get_children(),
                    timeout=self._request_timeout,
                )
            except TimeoutError:
                logger.warning(f"Timeout browsing node at depth {current_depth}")
                return
            except Exception as e:
                logger.debug(f"Error browsing node: {e}")
                return

        # Process children concurrently with backpressure
        tasks = []
        for child in children:
            tasks.append(self._process_node(child, parent_path, results, max_depth, current_depth))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_node(
        self,
        node: NodeType,
        parent_path: list[str],
        results: list[BrowseResult],
        max_depth: int,
        current_depth: int,
    ) -> None:
        """Process a single node and continue recursion if needed.

        Args:
            node: Node to process.
            parent_path: Path from root to parent node.
            results: List to append results to.
            max_depth: Maximum depth to browse.
            current_depth: Current depth level.
        """
        async with self._semaphore:  # Concurrency control
            try:
                browse_result = await self._get_node_info(node, parent_path)
                if browse_result:
                    results.append(browse_result)

                    # Continue browsing if this is an Object node
                    if browse_result.node_class == ua.NodeClass.Object:
                        new_path = [*parent_path, browse_result.browse_name]
                        await self._browse_recursive(
                            node, new_path, results, max_depth, current_depth + 1
                        )
            except Exception as e:
                logger.debug(f"Error processing node: {e}")

    async def _get_node_info(
        self,
        node: NodeType,
        parent_path: list[str],
    ) -> BrowseResult | None:
        """Get information about a node.

        Args:
            node: Node to get info for.
            parent_path: Path from root to parent node.

        Returns:
            BrowseResult with node information, or None if node should be skipped.
        """
        try:
            # Read basic attributes
            node_id = node.nodeid.to_string()
            browse_name = (await node.read_browse_name()).Name
            display_name = (await node.read_display_name()).Text
            node_class = await node.read_node_class()

            # Skip server/types nodes
            if browse_name in ("Server", "Types", "Views"):
                return None

            result = BrowseResult(
                node=node,
                node_id=node_id,
                browse_name=browse_name or "",
                display_name=display_name or browse_name or "",
                node_class=node_class,
                parent_path=parent_path,
            )

            # Get additional attributes for Variable nodes
            if node_class == ua.NodeClass.Variable:
                await self._enrich_variable_info(node, result)

            # Try to extract IRDI from HasDictionaryEntry reference
            irdi = await self._extract_irdi_from_references(node)
            if irdi:
                result.irdi = irdi

            return result

        except Exception as e:
            logger.debug(f"Error getting node info: {e}")
            return None

    async def _enrich_variable_info(self, node: NodeType, result: BrowseResult) -> None:
        """Enrich a BrowseResult with Variable-specific attributes.

        Args:
            node: Variable node.
            result: BrowseResult to enrich.
        """
        try:
            # Get data type
            data_type_node = await node.read_data_type()
            if data_type_node:
                try:
                    client = self._ensure_connected()
                    dt_node = client.get_node(data_type_node)
                    result.data_type = (await dt_node.read_browse_name()).Name
                except Exception:
                    result.data_type = str(data_type_node)
        except Exception:
            pass

        try:
            # Get description
            desc = await node.read_description()
            if desc and desc.Text:
                result.description = desc.Text
        except Exception:
            pass

        try:
            # Get access level
            result.access_level = await node.read_access_level()
        except Exception:
            pass

        try:
            # Try to get engineering unit from EURange or EUInformation
            await self._extract_engineering_unit(node, result)
        except Exception:
            pass

    async def _extract_engineering_unit(self, node: NodeType, result: BrowseResult) -> None:
        """Try to extract engineering unit from a variable node.

        Args:
            node: Variable node.
            result: BrowseResult to update.
        """
        try:
            # Look for EngineeringUnits property
            children = await node.get_children()
            for child in children:
                browse_name = await child.read_browse_name()
                if browse_name.Name == "EngineeringUnits":
                    eu_value = await child.read_value()
                    if hasattr(eu_value, "DisplayName") and eu_value.DisplayName:
                        result.engineering_unit = eu_value.DisplayName.Text
                    elif hasattr(eu_value, "UnitId"):
                        result.engineering_unit = str(eu_value.UnitId)
                    break
        except Exception:
            pass

    async def _extract_irdi_from_references(self, node: NodeType) -> str | None:
        """Extract IRDI from HasDictionaryEntry references.

        HasDictionaryEntry references are used to link OPC UA nodes to
        semantic dictionary entries (like ECLASS or IEC CDD).

        Args:
            node: Node to check for dictionary references.

        Returns:
            IRDI string if found, None otherwise.
        """
        try:
            # Get all references from this node
            refs = await node.get_references(
                refs=HAS_DICTIONARY_ENTRY_ID,
                direction=ua.BrowseDirection.Forward,
            )

            for ref in refs:
                # The target node should contain the IRDI
                target_node_id = ref.NodeId
                client = self._ensure_connected()
                target_node = client.get_node(target_node_id)

                # Try to read the value which should contain the IRDI
                try:
                    value = await target_node.read_value()
                    if isinstance(value, str) and self._looks_like_irdi(value):
                        return value
                except Exception:
                    pass

                # Also check the browse name or display name
                try:
                    browse_name = (await target_node.read_browse_name()).Name
                    if browse_name and self._looks_like_irdi(browse_name):
                        return browse_name
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Error extracting IRDI from references: {e}")

        return None

    @staticmethod
    def _looks_like_irdi(value: str) -> bool:
        """Check if a string looks like an IRDI.

        Args:
            value: String to check.

        Returns:
            True if the string appears to be an IRDI.
        """
        # Basic IRDI format check: contains # or starts with 0173 (ECLASS)
        return "#" in value or value.startswith("0173") or "-" in value

    async def read_value(self, node_id: str) -> object:
        """Read the current value of a node.

        Args:
            node_id: Node ID string.

        Returns:
            Current value of the node.

        Raises:
            OPCUABrowseError: If reading fails.
        """
        client = self._ensure_connected()

        try:
            node = client.get_node(node_id)
            value = await asyncio.wait_for(
                node.read_value(),
                timeout=self._request_timeout,
            )
            return value
        except TimeoutError as e:
            raise OPCUABrowseError(f"Timeout reading value from {node_id}") from e
        except Exception as e:
            raise OPCUABrowseError(f"Failed to read value from {node_id}: {e}") from e

    async def read_values(self, node_ids: Sequence[str]) -> list[object]:
        """Read values from multiple nodes concurrently.

        Args:
            node_ids: List of node ID strings.

        Returns:
            List of values in the same order as node_ids.

        Raises:
            OPCUABrowseError: If reading fails.
        """
        tasks = [self.read_value(node_id) for node_id in node_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        values = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to read {node_ids[i]}: {result}")
                values.append(None)
            else:
                values.append(result)

        return values

    async def get_node_metadata(self, node_id: str) -> BrowseResult | None:
        """Get metadata for a specific node.

        Args:
            node_id: Node ID string.

        Returns:
            BrowseResult with node metadata, or None if not found.
        """
        client = self._ensure_connected()

        try:
            node = client.get_node(node_id)
            return await self._get_node_info(node, [])
        except Exception as e:
            logger.warning(f"Failed to get metadata for {node_id}: {e}")
            return None


@asynccontextmanager
async def create_opcua_browser(
    endpoint_url: str,
    settings: OPCUASettings | None = None,
) -> AsyncIterator[OPCUABrowser]:
    """Create an OPC UA browser as an async context manager.

    Args:
        endpoint_url: OPC UA server endpoint URL.
        settings: Optional OPC UA settings.

    Yields:
        Connected OPCUABrowser instance.

    Example:
        async with create_opcua_browser("opc.tcp://localhost:4840") as browser:
            tags = await browser.browse_all_tags()
    """
    browser = OPCUABrowser(endpoint_url, settings)
    try:
        await browser.connect()
        yield browser
    finally:
        await browser.disconnect()
