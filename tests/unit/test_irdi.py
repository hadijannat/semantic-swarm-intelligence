"""Unit tests for IRDI parsing and normalization."""

from __future__ import annotations

import pytest

from noa_swarm.common.ids import IRDI, IRDIError


class TestIRDIParsing:
    """Tests for IRDI string parsing."""

    def test_parse_valid_irdi(self) -> None:
        """Test parsing a valid IRDI string."""
        irdi = IRDI.parse("0173-1#01-ABA234#001")

        assert irdi.org_code == "0173-1"
        assert irdi.item_type == "01"
        assert irdi.item_code == "ABA234"
        assert irdi.version == "001"

    def test_parse_lowercase_normalizes_to_uppercase(self) -> None:
        """Test that lowercase input is normalized to uppercase."""
        irdi = IRDI.parse("0173-1#01-aba234#001")

        assert irdi.item_code == "ABA234"
        assert str(irdi) == "0173-1#01-ABA234#001"

    def test_parse_with_whitespace(self) -> None:
        """Test parsing with leading/trailing whitespace."""
        irdi = IRDI.parse("  0173-1#01-ABA234#001  ")

        assert irdi.org_code == "0173-1"
        assert irdi.item_code == "ABA234"

    def test_parse_different_org_codes(self) -> None:
        """Test parsing IRDIs with different organization codes."""
        # IEC CDD style
        irdi1 = IRDI.parse("0173-1#01-ABC123#002")
        assert irdi1.org_code == "0173-1"

        # Different org code
        irdi2 = IRDI.parse("0112-1#01-XYZ789#001")
        assert irdi2.org_code == "0112-1"

    def test_parse_different_item_types(self) -> None:
        """Test parsing IRDIs with different item types."""
        # Class
        irdi_class = IRDI.parse("0173-1#01-ABA234#001")
        assert irdi_class.item_type == "01"
        assert irdi_class.is_class

        # Property
        irdi_prop = IRDI.parse("0173-1#02-XYZ789#001")
        assert irdi_prop.item_type == "02"
        assert irdi_prop.is_property

        # Value
        irdi_val = IRDI.parse("0173-1#03-DEF456#001")
        assert irdi_val.item_type == "03"
        assert irdi_val.is_value

    def test_parse_empty_string_raises_error(self) -> None:
        """Test that parsing an empty string raises IRDIError."""
        with pytest.raises(IRDIError, match="cannot be empty"):
            IRDI.parse("")

    def test_parse_invalid_format_raises_error(self) -> None:
        """Test that parsing invalid format raises IRDIError."""
        with pytest.raises(IRDIError, match="Invalid IRDI format"):
            IRDI.parse("invalid-irdi-string")

    def test_parse_missing_delimiter_raises_error(self) -> None:
        """Test that missing delimiter raises IRDIError."""
        with pytest.raises(IRDIError, match="Invalid IRDI format"):
            IRDI.parse("0173-1-01-ABA234-001")

    def test_parse_alternative_format_without_version(self) -> None:
        """Test parsing alternative format without explicit version."""
        irdi = IRDI.parse("0173-1#01-ABA234")

        assert irdi.org_code == "0173-1"
        assert irdi.item_code == "ABA234"
        assert irdi.version == "001"  # Default version

    def test_parse_alternative_format_with_dots(self) -> None:
        """Test parsing alternative format using dots as delimiters."""
        irdi = IRDI.parse("0173-1.01-ABA234.002")

        assert irdi.org_code == "0173-1"
        assert irdi.item_type == "01"
        assert irdi.item_code == "ABA234"
        assert irdi.version == "002"


class TestIRDICreate:
    """Tests for IRDI creation."""

    def test_create_with_defaults(self) -> None:
        """Test creating IRDI with default version."""
        irdi = IRDI.create("0173-1", "01", "ABC123")

        assert irdi.org_code == "0173-1"
        assert irdi.item_type == "01"
        assert irdi.item_code == "ABC123"
        assert irdi.version == "001"

    def test_create_with_explicit_version(self) -> None:
        """Test creating IRDI with explicit version."""
        irdi = IRDI.create("0173-1", "02", "XYZ789", "003")

        assert irdi.version == "003"

    def test_create_normalizes_to_uppercase(self) -> None:
        """Test that create normalizes to uppercase."""
        irdi = IRDI.create("0173-1", "01", "abc123", "001")

        assert irdi.item_code == "ABC123"

    def test_create_strips_whitespace(self) -> None:
        """Test that create strips whitespace."""
        irdi = IRDI.create("  0173-1  ", "  01  ", "  ABC123  ", "  001  ")

        assert irdi.org_code == "0173-1"
        assert irdi.item_type == "01"


class TestIRDIValidation:
    """Tests for IRDI validation."""

    def test_is_valid_with_valid_irdi(self) -> None:
        """Test is_valid returns True for valid IRDI."""
        assert IRDI.is_valid("0173-1#01-ABA234#001") is True

    def test_is_valid_with_invalid_irdi(self) -> None:
        """Test is_valid returns False for invalid IRDI."""
        assert IRDI.is_valid("invalid") is False
        assert IRDI.is_valid("") is False
        assert IRDI.is_valid("0173-1") is False

    def test_post_init_validation_empty_org_code(self) -> None:
        """Test that empty org_code raises error."""
        with pytest.raises(IRDIError, match="Organization code cannot be empty"):
            IRDI(org_code="", item_type="01", item_code="ABC", version="001")

    def test_post_init_validation_empty_item_type(self) -> None:
        """Test that empty item_type raises error."""
        with pytest.raises(IRDIError, match="Item type cannot be empty"):
            IRDI(org_code="0173-1", item_type="", item_code="ABC", version="001")

    def test_post_init_validation_empty_item_code(self) -> None:
        """Test that empty item_code raises error."""
        with pytest.raises(IRDIError, match="Item code cannot be empty"):
            IRDI(org_code="0173-1", item_type="01", item_code="", version="001")

    def test_post_init_validation_empty_version(self) -> None:
        """Test that empty version raises error."""
        with pytest.raises(IRDIError, match="Version cannot be empty"):
            IRDI(org_code="0173-1", item_type="01", item_code="ABC", version="")


class TestIRDIRepresentations:
    """Tests for IRDI string representations."""

    def test_to_canonical(self) -> None:
        """Test canonical string representation."""
        irdi = IRDI.parse("0173-1#01-aba234#001")

        assert irdi.to_canonical() == "0173-1#01-ABA234#001"

    def test_str_returns_canonical(self) -> None:
        """Test __str__ returns canonical form."""
        irdi = IRDI.parse("0173-1#01-ABA234#001")

        assert str(irdi) == "0173-1#01-ABA234#001"

    def test_repr(self) -> None:
        """Test __repr__ for debugging."""
        irdi = IRDI.parse("0173-1#01-ABA234#001")

        repr_str = repr(irdi)
        assert "IRDI" in repr_str
        assert "org_code='0173-1'" in repr_str
        assert "item_code='ABA234'" in repr_str

    def test_to_urn(self) -> None:
        """Test URN representation."""
        irdi = IRDI.parse("0173-1#01-ABA234#001")

        assert irdi.to_urn() == "urn:irdi:0173-1:01-ABA234:001"

    def test_to_aas_identifier(self) -> None:
        """Test AAS identifier representation."""
        irdi = IRDI.parse("0173-1#01-ABA234#001")

        expected = "https://admin-shell.io/irdi/0173-1/01-ABA234/001"
        assert irdi.to_aas_identifier() == expected


class TestIRDIComparison:
    """Tests for IRDI comparison and hashing."""

    def test_equality_same_irdi(self) -> None:
        """Test equality for same IRDI."""
        irdi1 = IRDI.parse("0173-1#01-ABA234#001")
        irdi2 = IRDI.parse("0173-1#01-ABA234#001")

        assert irdi1 == irdi2

    def test_equality_case_insensitive(self) -> None:
        """Test equality is case insensitive."""
        irdi1 = IRDI.parse("0173-1#01-ABA234#001")
        irdi2 = IRDI.parse("0173-1#01-aba234#001")

        assert irdi1 == irdi2

    def test_inequality_different_version(self) -> None:
        """Test inequality for different versions."""
        irdi1 = IRDI.parse("0173-1#01-ABA234#001")
        irdi2 = IRDI.parse("0173-1#01-ABA234#002")

        assert irdi1 != irdi2

    def test_inequality_different_item_code(self) -> None:
        """Test inequality for different item codes."""
        irdi1 = IRDI.parse("0173-1#01-ABA234#001")
        irdi2 = IRDI.parse("0173-1#01-XYZ789#001")

        assert irdi1 != irdi2

    def test_equality_with_string(self) -> None:
        """Test equality comparison with string."""
        irdi = IRDI.parse("0173-1#01-ABA234#001")

        assert irdi == "0173-1#01-ABA234#001"
        assert irdi == "0173-1#01-aba234#001"  # Case insensitive

    def test_inequality_with_invalid_string(self) -> None:
        """Test inequality with invalid string."""
        irdi = IRDI.parse("0173-1#01-ABA234#001")

        assert irdi != "invalid"

    def test_hash_consistency(self) -> None:
        """Test that equal IRDIs have same hash."""
        irdi1 = IRDI.parse("0173-1#01-ABA234#001")
        irdi2 = IRDI.parse("0173-1#01-aba234#001")

        assert hash(irdi1) == hash(irdi2)

    def test_usable_in_set(self) -> None:
        """Test that IRDIs can be used in sets."""
        irdi1 = IRDI.parse("0173-1#01-ABA234#001")
        irdi2 = IRDI.parse("0173-1#01-ABA234#001")
        irdi3 = IRDI.parse("0173-1#01-XYZ789#001")

        irdi_set = {irdi1, irdi2, irdi3}
        assert len(irdi_set) == 2

    def test_usable_as_dict_key(self) -> None:
        """Test that IRDIs can be used as dictionary keys."""
        irdi = IRDI.parse("0173-1#01-ABA234#001")

        data = {irdi: "test_value"}
        assert data[irdi] == "test_value"

    def test_less_than_comparison(self) -> None:
        """Test sorting with less than comparison."""
        irdi1 = IRDI.parse("0173-1#01-ABA234#001")
        irdi2 = IRDI.parse("0173-1#01-XYZ789#001")

        assert irdi1 < irdi2

    def test_sorting(self) -> None:
        """Test that IRDIs can be sorted."""
        irdi1 = IRDI.parse("0173-1#02-ZZZ999#001")
        irdi2 = IRDI.parse("0173-1#01-ABA234#001")
        irdi3 = IRDI.parse("0173-1#01-ABC000#001")

        sorted_irdis = sorted([irdi1, irdi2, irdi3])
        assert sorted_irdis[0] == irdi2
        assert sorted_irdis[1] == irdi3
        assert sorted_irdis[2] == irdi1


class TestIRDIOperations:
    """Tests for IRDI operations and methods."""

    def test_with_version(self) -> None:
        """Test creating new IRDI with different version."""
        irdi1 = IRDI.parse("0173-1#01-ABA234#001")
        irdi2 = irdi1.with_version("002")

        assert irdi2.version == "002"
        assert irdi2.item_code == irdi1.item_code
        assert irdi1.version == "001"  # Original unchanged

    def test_is_same_concept(self) -> None:
        """Test checking if IRDIs refer to same concept."""
        irdi1 = IRDI.parse("0173-1#01-ABA234#001")
        irdi2 = IRDI.parse("0173-1#01-ABA234#002")
        irdi3 = IRDI.parse("0173-1#01-XYZ789#001")

        assert irdi1.is_same_concept(irdi2) is True
        assert irdi1.is_same_concept(irdi3) is False

    def test_immutability(self) -> None:
        """Test that IRDI is immutable (frozen)."""
        irdi = IRDI.parse("0173-1#01-ABA234#001")

        with pytest.raises(AttributeError):
            irdi.version = "002"  # type: ignore[misc]


class TestIRDIClassConstants:
    """Tests for IRDI class constants."""

    def test_iec_cdd_constant(self) -> None:
        """Test IEC CDD organization code constant."""
        assert IRDI.IEC_CDD == "0173-1"

    def test_type_constants(self) -> None:
        """Test item type constants."""
        assert IRDI.TYPE_CLASS == "01"
        assert IRDI.TYPE_PROPERTY == "02"
        assert IRDI.TYPE_VALUE == "03"
        assert IRDI.TYPE_UNIT == "04"
