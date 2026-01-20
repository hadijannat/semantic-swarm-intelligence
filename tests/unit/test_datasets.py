"""Unit tests for ML dataset loaders and synthetic tag generator.

Tests cover:
- SyntheticTagGenerator: tag generation, IRDI mapping, splits
- TEPDataset: variable mapping, ISA tag conversion
- CMAPSSDataset: sensor mapping, ISA tag conversion
"""

from __future__ import annotations

import pytest

from noa_swarm.common.ids import IRDI
from noa_swarm.ml.datasets import (
    CMAPSS_OPERATIONAL_SETTINGS,
    CMAPSS_SENSORS,
    SEED_IRDIS,
    TAG_PATTERN_TO_IRDI,
    TEP_XMEAS_VARIABLES,
    TEP_XMV_VARIABLES,
    CMAPSSDataset,
    DatasetSplit,
    SyntheticTagGenerator,
    TagSample,
    TEPDataset,
)


class TestTagSample:
    """Tests for TagSample dataclass."""

    def test_create_tag_sample(self) -> None:
        """Test creating a TagSample instance."""
        sample = TagSample(
            tag_name="FIC-101",
            irdi="0173-1#01-AAA001#001",
            description="Flow controller",
            engineering_unit="m3/h",
            features={"prefix": "FIC", "number": "101"},
        )

        assert sample.tag_name == "FIC-101"
        assert sample.irdi == "0173-1#01-AAA001#001"
        assert sample.description == "Flow controller"
        assert sample.engineering_unit == "m3/h"
        assert sample.features["prefix"] == "FIC"

    def test_tag_sample_to_dict(self) -> None:
        """Test TagSample to_dict conversion."""
        sample = TagSample(
            tag_name="TI-201",
            irdi="0173-1#01-AAA002#001",
            description="Temperature indicator",
            engineering_unit="degC",
        )

        d = sample.to_dict()
        assert d["tag_name"] == "TI-201"
        assert d["irdi"] == "0173-1#01-AAA002#001"
        assert d["description"] == "Temperature indicator"
        assert d["engineering_unit"] == "degC"

    def test_tag_sample_immutable(self) -> None:
        """Test that TagSample is immutable (frozen dataclass)."""
        sample = TagSample(
            tag_name="PI-101",
            irdi="0173-1#01-AAA003#001",
        )

        with pytest.raises(AttributeError):
            sample.tag_name = "PI-102"  # type: ignore[misc]


class TestDatasetSplit:
    """Tests for DatasetSplit dataclass."""

    def test_create_split(self) -> None:
        """Test creating a DatasetSplit."""
        train = [TagSample("FI-101", "0173-1#01-AAA001#001")]
        val = [TagSample("TI-101", "0173-1#01-AAA002#001")]
        test = [TagSample("PI-101", "0173-1#01-AAA003#001")]

        split = DatasetSplit(train=train, val=val, test=test)

        assert len(split.train) == 1
        assert len(split.val) == 1
        assert len(split.test) == 1
        assert split.total_size == 3

    def test_split_irdi_distribution(self) -> None:
        """Test IRDI distribution calculation."""
        train = [
            TagSample("FI-101", "0173-1#01-AAA001#001"),
            TagSample("FI-102", "0173-1#01-AAA001#001"),
        ]
        val = [TagSample("TI-101", "0173-1#01-AAA002#001")]
        test = [TagSample("PI-101", "0173-1#01-AAA003#001")]

        split = DatasetSplit(train=train, val=val, test=test)
        dist = split.get_irdi_distribution()

        assert dist["0173-1#01-AAA001#001"] == 2
        assert dist["0173-1#01-AAA002#001"] == 1
        assert dist["0173-1#01-AAA003#001"] == 1


class TestSeedIRDIs:
    """Tests for seed IRDI definitions."""

    def test_seed_irdis_valid(self) -> None:
        """Test that all seed IRDIs are valid IRDI format."""
        for category, irdi_str in SEED_IRDIS.items():
            assert IRDI.is_valid(irdi_str), f"Invalid IRDI for {category}: {irdi_str}"

    def test_seed_irdis_unique(self) -> None:
        """Test that all seed IRDIs are unique."""
        irdis = list(SEED_IRDIS.values())
        unique_irdis = set(irdis)
        assert len(irdis) == len(unique_irdis), "Duplicate IRDIs found in SEED_IRDIS"

    def test_tag_pattern_irdis_valid(self) -> None:
        """Test that all TAG_PATTERN_TO_IRDI values are valid."""
        for pattern, irdi_str in TAG_PATTERN_TO_IRDI.items():
            assert IRDI.is_valid(irdi_str), f"Invalid IRDI for pattern {pattern}"

    def test_core_categories_present(self) -> None:
        """Test that core process categories are defined."""
        required = ["flow_rate", "temperature", "pressure", "level", "valve_position"]
        for category in required:
            assert category in SEED_IRDIS, f"Missing required category: {category}"


class TestSyntheticTagGenerator:
    """Tests for SyntheticTagGenerator."""

    def test_create_generator(self) -> None:
        """Test creating a generator with seed."""
        generator = SyntheticTagGenerator(seed=42)
        assert generator.seed == 42

    def test_generate_samples(self) -> None:
        """Test generating samples."""
        generator = SyntheticTagGenerator(seed=42)
        samples = generator.generate(num_samples=10)

        assert len(samples) == 10
        for sample in samples:
            assert isinstance(sample, TagSample)
            assert sample.tag_name
            assert IRDI.is_valid(sample.irdi)

    def test_generate_reproducible(self) -> None:
        """Test that generation is reproducible with same seed."""
        gen1 = SyntheticTagGenerator(seed=123)
        gen2 = SyntheticTagGenerator(seed=123)

        samples1 = gen1.generate(num_samples=5)
        samples2 = gen2.generate(num_samples=5)

        for s1, s2 in zip(samples1, samples2, strict=True):
            assert s1.tag_name == s2.tag_name
            assert s1.irdi == s2.irdi

    def test_generate_different_seeds(self) -> None:
        """Test that different seeds produce different results."""
        gen1 = SyntheticTagGenerator(seed=42)
        gen2 = SyntheticTagGenerator(seed=99)

        samples1 = gen1.generate(num_samples=10)
        samples2 = gen2.generate(num_samples=10)

        # At least some samples should differ
        tags1 = {s.tag_name for s in samples1}
        tags2 = {s.tag_name for s in samples2}
        assert tags1 != tags2

    def test_generate_split(self) -> None:
        """Test train/val/test split generation."""
        generator = SyntheticTagGenerator(seed=42)
        split = generator.generate_split(total=100, train_ratio=0.8, val_ratio=0.1)

        assert len(split.train) == 80
        assert len(split.val) == 10
        assert len(split.test) == 10
        assert split.total_size == 100

    def test_generate_split_custom_ratios(self) -> None:
        """Test split with custom ratios."""
        generator = SyntheticTagGenerator(seed=42)
        split = generator.generate_split(total=100, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

        assert len(split.train) == 60
        assert len(split.val) == 20
        assert len(split.test) == 20

    def test_generate_split_invalid_ratios(self) -> None:
        """Test that invalid ratios raise error."""
        generator = SyntheticTagGenerator(seed=42)

        # When all three ratios are explicitly provided and don't sum to 1.0
        with pytest.raises(ValueError, match="sum to 1.0"):
            generator.generate_split(
                total=100, train_ratio=0.8, val_ratio=0.3, test_ratio=0.1
            )

    def test_generate_balanced(self) -> None:
        """Test balanced IRDI generation."""
        generator = SyntheticTagGenerator(seed=42)
        samples = generator.generate_balanced(samples_per_irdi=5)

        # Should have multiple samples per IRDI
        irdi_counts: dict[str, int] = {}
        for sample in samples:
            irdi_counts[sample.irdi] = irdi_counts.get(sample.irdi, 0) + 1

        # Each IRDI should have exactly samples_per_irdi samples
        for count in irdi_counts.values():
            assert count == 5

    def test_generate_iter(self) -> None:
        """Test iterator generation."""
        generator = SyntheticTagGenerator(seed=42)
        samples = list(generator.generate_iter(num_samples=5))

        assert len(samples) == 5
        for sample in samples:
            assert isinstance(sample, TagSample)

    def test_tag_patterns(self) -> None:
        """Test that generated tags follow ISA patterns."""
        generator = SyntheticTagGenerator(seed=42)
        samples = generator.generate(num_samples=100)

        # Check that tags have expected prefixes
        valid_prefixes = {
            "FI", "FIC", "FT", "FCV", "FQ",  # Flow
            "TI", "TIC", "TT", "TSH", "TSL", "TSHH", "TSLL",  # Temperature
            "PI", "PIC", "PT", "PDI", "PDIC", "PSH", "PSL",  # Pressure
            "LI", "LIC", "LT", "LCV", "LSH", "LSL",  # Level
            "AI", "AIC", "AT",  # Analyzer
            "SI", "SIC", "ST",  # Speed
            "WI", "WIC", "WT",  # Weight
            "XV", "HV", "CV",  # Generic valves
        }

        for sample in samples:
            # Extract prefix (letters before delimiter or numbers)
            prefix = ""
            for char in sample.tag_name:
                if char.isalpha():
                    prefix += char
                else:
                    break

            assert prefix in valid_prefixes, f"Unknown prefix: {prefix} in {sample.tag_name}"

    def test_get_irdi_for_tag(self) -> None:
        """Test static IRDI lookup for tag names."""
        # Flow tags
        assert SyntheticTagGenerator.get_irdi_for_tag("FIC-101") == SEED_IRDIS["flow_rate"]
        assert SyntheticTagGenerator.get_irdi_for_tag("FI_201") == SEED_IRDIS["flow_rate"]

        # Temperature tags
        assert SyntheticTagGenerator.get_irdi_for_tag("TIC-301") == SEED_IRDIS["temperature"]

        # Pressure tags
        assert SyntheticTagGenerator.get_irdi_for_tag("PI401") == SEED_IRDIS["pressure"]

        # Unknown patterns
        assert SyntheticTagGenerator.get_irdi_for_tag("XYZ-123") is None
        assert SyntheticTagGenerator.get_irdi_for_tag("123-ABC") is None

    def test_get_all_irdis(self) -> None:
        """Test getting all seed IRDIs."""
        irdis = SyntheticTagGenerator.get_all_irdis()
        assert len(irdis) == len(SEED_IRDIS)
        assert "flow_rate" in irdis
        assert "temperature" in irdis

    def test_get_irdi_categories(self) -> None:
        """Test getting IRDI category list."""
        categories = SyntheticTagGenerator.get_irdi_categories()
        assert "flow_rate" in categories
        assert "temperature" in categories
        assert "pressure" in categories


class TestTEPDataset:
    """Tests for TEPDataset."""

    def test_create_dataset(self) -> None:
        """Test creating TEP dataset."""
        dataset = TEPDataset()
        assert dataset.num_variables == 52
        assert dataset.num_xmeas == 41
        assert dataset.num_xmv == 11

    def test_variable_definitions(self) -> None:
        """Test that TEP variable definitions are complete."""
        assert len(TEP_XMEAS_VARIABLES) == 41
        assert len(TEP_XMV_VARIABLES) == 11

        # Check all indices are present
        for i in range(1, 42):
            assert i in TEP_XMEAS_VARIABLES, f"Missing XMEAS_{i}"
        for i in range(1, 12):
            assert i in TEP_XMV_VARIABLES, f"Missing XMV_{i}"

    def test_get_variables(self) -> None:
        """Test getting all variables."""
        dataset = TEPDataset()
        variables = dataset.get_variables()

        assert len(variables) == 52

        # Check that all have valid IRDIs
        for var in variables:
            assert IRDI.is_valid(var.irdi), f"Invalid IRDI for {var.isa_tag}"

    def test_get_variable_by_tag(self) -> None:
        """Test finding variable by ISA tag."""
        dataset = TEPDataset()

        # Find flow indicator
        var = dataset.get_variable_by_tag("FI-101")
        assert var is not None
        assert var.name == "A Feed (stream 1)"
        assert var.category == "flow_rate"

        # Find temperature indicator
        var = dataset.get_variable_by_tag("TI-101")
        assert var is not None
        assert var.category == "temperature"

        # Non-existent tag
        var = dataset.get_variable_by_tag("XYZ-999")
        assert var is None

    def test_get_variable_by_index(self) -> None:
        """Test finding variable by type and index."""
        dataset = TEPDataset()

        # XMEAS_1
        var = dataset.get_variable_by_index("XMEAS", 1)
        assert var is not None
        assert var.isa_tag == "FI-101"

        # XMV_1
        var = dataset.get_variable_by_index("XMV", 1)
        assert var is not None
        assert var.category == "valve_position"

        # Non-existent
        var = dataset.get_variable_by_index("XMEAS", 100)
        assert var is None

    def test_get_tag_samples(self) -> None:
        """Test getting variables as TagSample instances."""
        dataset = TEPDataset()
        samples = dataset.get_tag_samples()

        assert len(samples) == 52
        for sample in samples:
            assert isinstance(sample, TagSample)
            assert IRDI.is_valid(sample.irdi)

    def test_get_xmeas_variables(self) -> None:
        """Test getting only XMEAS variables."""
        dataset = TEPDataset()
        xmeas = dataset.get_xmeas_variables()

        assert len(xmeas) == 41
        for var in xmeas:
            assert var.var_type == "XMEAS"

    def test_get_xmv_variables(self) -> None:
        """Test getting only XMV variables."""
        dataset = TEPDataset()
        xmv = dataset.get_xmv_variables()

        assert len(xmv) == 11
        for var in xmv:
            assert var.var_type == "XMV"

    def test_get_variables_by_category(self) -> None:
        """Test filtering variables by category."""
        dataset = TEPDataset()

        flow_vars = dataset.get_variables_by_category("flow_rate")
        assert len(flow_vars) > 0
        for var in flow_vars:
            assert var.category == "flow_rate"

        temp_vars = dataset.get_variables_by_category("temperature")
        assert len(temp_vars) > 0
        for var in temp_vars:
            assert var.category == "temperature"

    def test_get_irdi_mapping(self) -> None:
        """Test getting ISA tag to IRDI mapping."""
        dataset = TEPDataset()
        mapping = dataset.get_irdi_mapping()

        assert len(mapping) == 52
        assert "FI-101" in mapping
        assert IRDI.is_valid(mapping["FI-101"])

    def test_get_split(self) -> None:
        """Test train/val/test split."""
        dataset = TEPDataset()
        split = dataset.get_split(train_ratio=0.8, val_ratio=0.1, seed=42)

        assert split.total_size == 52
        assert len(split.train) == 41  # 80% of 52
        assert len(split.val) == 5  # 10% of 52
        assert len(split.test) == 6  # Remainder

    def test_variable_original_name(self) -> None:
        """Test TEPVariable original_name property."""
        dataset = TEPDataset()
        var = dataset.get_variable_by_index("XMEAS", 1)

        assert var is not None
        assert var.original_name == "XMEAS_1"

    def test_repr(self) -> None:
        """Test dataset string representation."""
        dataset = TEPDataset()
        repr_str = repr(dataset)

        assert "TEPDataset" in repr_str
        assert "52" in repr_str


class TestCMAPSSDataset:
    """Tests for CMAPSSDataset."""

    def test_create_dataset(self) -> None:
        """Test creating C-MAPSS dataset."""
        dataset = CMAPSSDataset()
        assert dataset.num_sensors == 24  # 3 settings + 21 sensors
        assert dataset.num_operational_settings == 3
        assert dataset.num_measurement_sensors == 21

    def test_create_with_subset(self) -> None:
        """Test creating dataset with specific subset."""
        dataset = CMAPSSDataset(subset="FD002")
        assert dataset.subset == "FD002"

    def test_sensor_definitions(self) -> None:
        """Test that sensor definitions are complete."""
        assert len(CMAPSS_OPERATIONAL_SETTINGS) == 3
        assert len(CMAPSS_SENSORS) == 21

        # Check all indices are present
        for i in range(1, 4):
            assert i in CMAPSS_OPERATIONAL_SETTINGS, f"Missing setting_{i}"
        for i in range(1, 22):
            assert i in CMAPSS_SENSORS, f"Missing sensor_{i}"

    def test_get_sensors(self) -> None:
        """Test getting all sensors."""
        dataset = CMAPSSDataset()
        sensors = dataset.get_sensors()

        assert len(sensors) == 24

        # Check that all have valid IRDIs
        for sensor in sensors:
            assert IRDI.is_valid(sensor.irdi), f"Invalid IRDI for {sensor.isa_tag}"

    def test_get_sensor_by_tag(self) -> None:
        """Test finding sensor by ISA tag."""
        dataset = CMAPSSDataset()

        # Find temperature sensor
        sensor = dataset.get_sensor_by_tag("TI-201")
        assert sensor is not None
        assert "Temperature" in sensor.name
        assert sensor.category == "temperature"

        # Find pressure sensor
        sensor = dataset.get_sensor_by_tag("PI-201")
        assert sensor is not None
        assert sensor.category == "pressure"

        # Non-existent tag
        sensor = dataset.get_sensor_by_tag("XYZ-999")
        assert sensor is None

    def test_get_sensor_by_index(self) -> None:
        """Test finding sensor by type and index."""
        dataset = CMAPSSDataset()

        # Setting 1 (Altitude)
        sensor = dataset.get_sensor_by_index("operational_setting", 1)
        assert sensor is not None
        assert sensor.name == "Altitude"

        # Sensor 1 (T2)
        sensor = dataset.get_sensor_by_index("sensor", 1)
        assert sensor is not None
        assert "T2" in sensor.name or "Fan Inlet" in sensor.name

        # Non-existent
        sensor = dataset.get_sensor_by_index("sensor", 100)
        assert sensor is None

    def test_get_tag_samples(self) -> None:
        """Test getting sensors as TagSample instances."""
        dataset = CMAPSSDataset()
        samples = dataset.get_tag_samples()

        assert len(samples) == 24
        for sample in samples:
            assert isinstance(sample, TagSample)
            assert IRDI.is_valid(sample.irdi)

    def test_get_operational_settings(self) -> None:
        """Test getting only operational settings."""
        dataset = CMAPSSDataset()
        settings = dataset.get_operational_settings()

        assert len(settings) == 3
        for sensor in settings:
            assert sensor.sensor_type == "operational_setting"

    def test_get_measurement_sensors(self) -> None:
        """Test getting only measurement sensors."""
        dataset = CMAPSSDataset()
        sensors = dataset.get_measurement_sensors()

        assert len(sensors) == 21
        for sensor in sensors:
            assert sensor.sensor_type == "sensor"

    def test_get_sensors_by_category(self) -> None:
        """Test filtering sensors by category."""
        dataset = CMAPSSDataset()

        temp_sensors = dataset.get_sensors_by_category("temperature")
        assert len(temp_sensors) > 0
        for sensor in temp_sensors:
            assert sensor.category == "temperature"

        speed_sensors = dataset.get_sensors_by_category("speed")
        assert len(speed_sensors) > 0
        for sensor in speed_sensors:
            assert sensor.category == "speed"

    def test_get_irdi_mapping(self) -> None:
        """Test getting ISA tag to IRDI mapping."""
        dataset = CMAPSSDataset()
        mapping = dataset.get_irdi_mapping()

        assert len(mapping) == 24
        assert "TI-201" in mapping
        assert IRDI.is_valid(mapping["TI-201"])

    def test_get_split(self) -> None:
        """Test train/val/test split."""
        dataset = CMAPSSDataset()
        split = dataset.get_split(train_ratio=0.8, val_ratio=0.1, seed=42)

        assert split.total_size == 24

    def test_get_subset_info(self) -> None:
        """Test getting subset information."""
        dataset = CMAPSSDataset(subset="FD001")
        info = dataset.get_subset_info()

        assert "description" in info
        assert "operating_conditions" in info
        assert "fault_modes" in info

    def test_get_available_subsets(self) -> None:
        """Test getting available subsets."""
        subsets = CMAPSSDataset.get_available_subsets()

        assert "FD001" in subsets
        assert "FD002" in subsets
        assert "FD003" in subsets
        assert "FD004" in subsets
        assert len(subsets) == 4

    def test_sensor_original_name(self) -> None:
        """Test CMAPSSSensor original_name property."""
        dataset = CMAPSSDataset()

        setting = dataset.get_sensor_by_index("operational_setting", 1)
        assert setting is not None
        assert setting.original_name == "setting_1"

        sensor = dataset.get_sensor_by_index("sensor", 1)
        assert sensor is not None
        assert sensor.original_name == "sensor_1"

    def test_repr(self) -> None:
        """Test dataset string representation."""
        dataset = CMAPSSDataset(subset="FD002")
        repr_str = repr(dataset)

        assert "CMAPSSDataset" in repr_str
        assert "FD002" in repr_str


class TestDatasetIntegration:
    """Integration tests combining multiple dataset components."""

    def test_combine_datasets(self) -> None:
        """Test combining samples from multiple datasets."""
        generator = SyntheticTagGenerator(seed=42)
        tep = TEPDataset()
        cmapss = CMAPSSDataset()

        # Combine samples
        synthetic_samples = generator.generate(num_samples=50)
        tep_samples = tep.get_tag_samples()
        cmapss_samples = cmapss.get_tag_samples()

        all_samples = synthetic_samples + tep_samples + cmapss_samples
        assert len(all_samples) == 50 + 52 + 24

        # All should have valid IRDIs
        for sample in all_samples:
            assert IRDI.is_valid(sample.irdi)

    def test_irdi_coverage(self) -> None:
        """Test that datasets cover multiple IRDI categories."""
        tep = TEPDataset()
        cmapss = CMAPSSDataset()

        tep_categories = {v.category for v in tep.get_variables()}
        cmapss_categories = {s.category for s in cmapss.get_sensors()}

        # Both should have temperature and pressure
        assert "temperature" in tep_categories
        assert "temperature" in cmapss_categories
        assert "pressure" in tep_categories
        assert "pressure" in cmapss_categories

    def test_split_reproducibility(self) -> None:
        """Test that splits are reproducible across datasets."""
        tep1 = TEPDataset()
        tep2 = TEPDataset()

        split1 = tep1.get_split(seed=42)
        split2 = tep2.get_split(seed=42)

        # Same seed should produce same split
        for s1, s2 in zip(split1.train, split2.train, strict=True):
            assert s1.tag_name == s2.tag_name
