"""IRDI (International Registration Data Identifier) parsing and normalization.

IRDIs follow the ISO 29002-5 standard and are formatted like: `0173-1#01-ABA234#001`

Components:
- Organization code: `0173-1` (e.g., IEC CDD)
- Item type: `01` (class/property identifier)
- Item code: `ABA234` (unique item identifier)
- Version: `001` (version number)

This module provides an immutable IRDI class for parsing, normalizing,
and comparing IRDIs used in industrial semantic interoperability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


class IRDIError(ValueError):
    """Exception raised for IRDI parsing or validation errors."""

    pass


@dataclass(frozen=True, slots=True)
class IRDI:
    """Immutable International Registration Data Identifier.

    IRDIs are used in IEC 61360 / ISO 13584 standards for uniquely identifying
    concepts in industrial data dictionaries like ECLASS and IEC CDD.

    The format is: `{org_code}#{item_type}-{item_code}#{version}`
    Example: `0173-1#01-ABA234#001`

    Attributes:
        org_code: Organization code (e.g., '0173-1' for IEC CDD)
        item_type: Item type identifier (e.g., '01' for class, '02' for property)
        item_code: Unique item code within the organization
        version: Version identifier

    Examples:
        >>> irdi = IRDI.parse("0173-1#01-ABA234#001")
        >>> irdi.org_code
        '0173-1'
        >>> irdi.item_code
        'ABA234'
        >>> str(irdi)
        '0173-1#01-ABA234#001'
    """

    org_code: str
    item_type: str
    item_code: str
    version: str

    # Common organization codes
    IEC_CDD: ClassVar[str] = "0173-1"
    ECLASS: ClassVar[str] = "0173-1"  # ECLASS uses IEC CDD registration

    # Item type codes
    TYPE_CLASS: ClassVar[str] = "01"
    TYPE_PROPERTY: ClassVar[str] = "02"
    TYPE_VALUE: ClassVar[str] = "03"
    TYPE_UNIT: ClassVar[str] = "04"

    # Regex pattern for parsing IRDI strings
    # Format: ORG_CODE#ITEM_TYPE-ITEM_CODE#VERSION
    _PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<org_code>[\w-]+)#(?P<item_type>\w+)-(?P<item_code>\w+)#(?P<version>\w+)$"
    )

    # Alternative patterns that may be encountered
    _ALT_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        # Pattern without version separator (some systems use different delimiters)
        re.compile(r"^(?P<org_code>[\w-]+)#(?P<item_type>\w+)-(?P<item_code>\w+)$"),
        # Pattern with dots instead of hashes
        re.compile(
            r"^(?P<org_code>[\w-]+)\.(?P<item_type>\w+)-(?P<item_code>\w+)\.(?P<version>\w+)$"
        ),
    ]

    def __post_init__(self) -> None:
        """Validate IRDI components after initialization."""
        if not self.org_code:
            raise IRDIError("Organization code cannot be empty")
        if not self.item_type:
            raise IRDIError("Item type cannot be empty")
        if not self.item_code:
            raise IRDIError("Item code cannot be empty")
        if not self.version:
            raise IRDIError("Version cannot be empty")

    @classmethod
    def parse(cls, irdi_string: str) -> IRDI:
        """Parse an IRDI string into its components.

        Args:
            irdi_string: The IRDI string to parse.

        Returns:
            IRDI: Parsed IRDI instance.

        Raises:
            IRDIError: If the string cannot be parsed as a valid IRDI.

        Examples:
            >>> IRDI.parse("0173-1#01-ABA234#001")
            IRDI(org_code='0173-1', item_type='01', item_code='ABA234', version='001')
        """
        if not irdi_string:
            raise IRDIError("IRDI string cannot be empty")

        # Normalize the input
        normalized = irdi_string.strip().upper()

        # Try main pattern first
        match = cls._PATTERN.match(normalized)
        if match:
            return cls(
                org_code=match.group("org_code"),
                item_type=match.group("item_type"),
                item_code=match.group("item_code"),
                version=match.group("version"),
            )

        # Try alternative patterns
        for pattern in cls._ALT_PATTERNS:
            match = pattern.match(normalized)
            if match:
                groups = match.groupdict()
                return cls(
                    org_code=groups["org_code"],
                    item_type=groups["item_type"],
                    item_code=groups["item_code"],
                    version=groups.get("version", "001"),  # Default version if not present
                )

        raise IRDIError(
            f"Invalid IRDI format: '{irdi_string}'. "
            f"Expected format: ORG_CODE#ITEM_TYPE-ITEM_CODE#VERSION "
            f"(e.g., '0173-1#01-ABA234#001')"
        )

    @classmethod
    def create(
        cls,
        org_code: str,
        item_type: str,
        item_code: str,
        version: str = "001",
    ) -> IRDI:
        """Create an IRDI with normalized components.

        Args:
            org_code: Organization code.
            item_type: Item type identifier.
            item_code: Unique item code.
            version: Version identifier (defaults to '001').

        Returns:
            IRDI: New IRDI instance with normalized components.

        Examples:
            >>> IRDI.create("0173-1", "01", "aba234")
            IRDI(org_code='0173-1', item_type='01', item_code='ABA234', version='001')
        """
        return cls(
            org_code=org_code.strip().upper(),
            item_type=item_type.strip().upper(),
            item_code=item_code.strip().upper(),
            version=version.strip().upper(),
        )

    @classmethod
    def is_valid(cls, irdi_string: str) -> bool:
        """Check if a string is a valid IRDI.

        Args:
            irdi_string: The string to validate.

        Returns:
            bool: True if the string is a valid IRDI, False otherwise.

        Examples:
            >>> IRDI.is_valid("0173-1#01-ABA234#001")
            True
            >>> IRDI.is_valid("invalid")
            False
        """
        try:
            cls.parse(irdi_string)
            return True
        except IRDIError:
            return False

    def to_canonical(self) -> str:
        """Return the canonical string representation of the IRDI.

        The canonical form uses uppercase and the standard delimiter format.

        Returns:
            str: Canonical IRDI string.

        Examples:
            >>> IRDI.parse("0173-1#01-aba234#001").to_canonical()
            '0173-1#01-ABA234#001'
        """
        return f"{self.org_code}#{self.item_type}-{self.item_code}#{self.version}"

    def to_urn(self) -> str:
        """Return the IRDI as a URN (Uniform Resource Name).

        Returns:
            str: URN representation of the IRDI.

        Examples:
            >>> IRDI.parse("0173-1#01-ABA234#001").to_urn()
            'urn:irdi:0173-1:01-ABA234:001'
        """
        return f"urn:irdi:{self.org_code}:{self.item_type}-{self.item_code}:{self.version}"

    def to_aas_identifier(self) -> str:
        """Return the IRDI as an AAS (Asset Administration Shell) identifier.

        Returns:
            str: AAS-compatible identifier URL.

        Examples:
            >>> IRDI.parse("0173-1#01-ABA234#001").to_aas_identifier()
            'https://admin-shell.io/irdi/0173-1/01-ABA234/001'
        """
        return f"https://admin-shell.io/irdi/{self.org_code}/{self.item_type}-{self.item_code}/{self.version}"

    def with_version(self, version: str) -> IRDI:
        """Create a new IRDI with a different version.

        Args:
            version: The new version identifier.

        Returns:
            IRDI: New IRDI instance with the specified version.

        Examples:
            >>> irdi = IRDI.parse("0173-1#01-ABA234#001")
            >>> irdi.with_version("002")
            IRDI(org_code='0173-1', item_type='01', item_code='ABA234', version='002')
        """
        return IRDI(
            org_code=self.org_code,
            item_type=self.item_type,
            item_code=self.item_code,
            version=version.strip().upper(),
        )

    def is_same_concept(self, other: IRDI) -> bool:
        """Check if two IRDIs refer to the same concept (ignoring version).

        Args:
            other: Another IRDI to compare.

        Returns:
            bool: True if both IRDIs refer to the same concept.

        Examples:
            >>> irdi1 = IRDI.parse("0173-1#01-ABA234#001")
            >>> irdi2 = IRDI.parse("0173-1#01-ABA234#002")
            >>> irdi1.is_same_concept(irdi2)
            True
        """
        return (
            self.org_code == other.org_code
            and self.item_type == other.item_type
            and self.item_code == other.item_code
        )

    @property
    def is_class(self) -> bool:
        """Check if this IRDI represents a class definition.

        Returns:
            bool: True if item_type indicates a class.
        """
        return self.item_type == self.TYPE_CLASS

    @property
    def is_property(self) -> bool:
        """Check if this IRDI represents a property definition.

        Returns:
            bool: True if item_type indicates a property.
        """
        return self.item_type == self.TYPE_PROPERTY

    @property
    def is_value(self) -> bool:
        """Check if this IRDI represents a value definition.

        Returns:
            bool: True if item_type indicates a value.
        """
        return self.item_type == self.TYPE_VALUE

    def __str__(self) -> str:
        """Return the canonical string representation."""
        return self.to_canonical()

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (
            f"IRDI(org_code={self.org_code!r}, item_type={self.item_type!r}, "
            f"item_code={self.item_code!r}, version={self.version!r})"
        )

    def __hash__(self) -> int:
        """Return hash based on canonical representation."""
        return hash(self.to_canonical())

    def __eq__(self, other: object) -> bool:
        """Check equality based on all components."""
        if isinstance(other, IRDI):
            return (
                self.org_code == other.org_code
                and self.item_type == other.item_type
                and self.item_code == other.item_code
                and self.version == other.version
            )
        if isinstance(other, str):
            try:
                return self == IRDI.parse(other)
            except IRDIError:
                return False
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        """Support sorting by canonical representation."""
        if isinstance(other, IRDI):
            return self.to_canonical() < other.to_canonical()
        return NotImplemented
