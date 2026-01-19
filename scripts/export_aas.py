#!/usr/bin/env python3
"""CLI tool for exporting AAS packages.

This script provides a command-line interface for creating and exporting
Asset Administration Shell packages from tag mapping data.

Usage:
    # Export to AASX package
    python scripts/export_aas.py --output plant001.aasx --format aasx

    # Export to JSON
    python scripts/export_aas.py --output plant001.json --format json

    # Export with custom IDs
    python scripts/export_aas.py \\
        --output plant001.aasx \\
        --aas-id "urn:company:aas:plant001" \\
        --asset-id "urn:company:asset:plant001"

    # Load tags from file
    python scripts/export_aas.py \\
        --input tags.json \\
        --output plant001.aasx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from noa_swarm.aas import (
    TagMappingSubmodel,
    DiscoveredTag,
    MappingStatus,
    ConsensusInfo,
    AASExporter,
    ExportFormat,
    ExportConfig,
    create_tag_mapping_aas,
)
from noa_swarm.common.logging import get_logger

logger = get_logger(__name__)


def load_tags_from_json(input_path: Path) -> list[DiscoveredTag]:
    """Load discovered tags from a JSON file.

    Expected format:
    {
        "tags": [
            {
                "tag_name": "TIC-101.PV",
                "browse_path": "/Objects/TIC-101/PV",
                "irdi": "0173-1#02-AAB663#001",
                "status": "mapped",
                "confidence": 0.95
            },
            ...
        ]
    }

    Args:
        input_path: Path to the JSON file.

    Returns:
        List of DiscoveredTag objects.
    """
    with open(input_path) as f:
        data = json.load(f)

    tags = []
    for item in data.get("tags", []):
        status_str = item.get("status", "pending")
        try:
            status = MappingStatus(status_str)
        except ValueError:
            status = MappingStatus.PENDING

        consensus = None
        if "confidence" in item:
            consensus = ConsensusInfo(
                confidence=item["confidence"],
                participating_agents=item.get("participating_agents", 0),
            )

        tag = DiscoveredTag(
            tag_name=item["tag_name"],
            browse_path=item["browse_path"],
            irdi=item.get("irdi"),
            preferred_name=item.get("preferred_name"),
            status=status,
            consensus=consensus,
            data_type=item.get("data_type"),
            unit=item.get("unit"),
            node_id=item.get("node_id"),
        )
        tags.append(tag)

    return tags


def create_sample_tags() -> list[DiscoveredTag]:
    """Create sample tags for demonstration.

    Returns:
        List of sample DiscoveredTag objects.
    """
    return [
        DiscoveredTag(
            tag_name="TIC-101.PV",
            browse_path="/Objects/Plant/Area1/TIC-101/PV",
            irdi="0173-1#02-AAB663#001",
            preferred_name="Temperature",
            status=MappingStatus.MAPPED,
            consensus=ConsensusInfo(confidence=0.95, participating_agents=3),
        ),
        DiscoveredTag(
            tag_name="FIC-200.SP",
            browse_path="/Objects/Plant/Area1/FIC-200/SP",
            irdi="0173-1#02-AAD001#001",
            preferred_name="Flow setpoint",
            status=MappingStatus.VERIFIED,
            consensus=ConsensusInfo(confidence=0.98, participating_agents=5),
        ),
        DiscoveredTag(
            tag_name="PIC-300.PV",
            browse_path="/Objects/Plant/Area2/PIC-300/PV",
            status=MappingStatus.PENDING,
        ),
        DiscoveredTag(
            tag_name="LIC-400.OP",
            browse_path="/Objects/Plant/Area2/LIC-400/OP",
            irdi="0173-1#02-AAB664#001",
            preferred_name="Level output",
            status=MappingStatus.MAPPED,
            consensus=ConsensusInfo(confidence=0.87, participating_agents=3),
        ),
    ]


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Export AAS packages from tag mapping data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output file path (.aasx, .json, or .xml)",
    )

    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["json", "xml", "aasx"],
        help="Export format (inferred from extension if not specified)",
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        help="Input JSON file with tag data (uses sample data if not provided)",
    )

    parser.add_argument(
        "--aas-id",
        type=str,
        default="urn:noa:aas:tagmapping:default",
        help="AAS identifier (default: urn:noa:aas:tagmapping:default)",
    )

    parser.add_argument(
        "--asset-id",
        type=str,
        default="urn:noa:asset:plant:default",
        help="Asset identifier (default: urn:noa:asset:plant:default)",
    )

    parser.add_argument(
        "--submodel-id",
        type=str,
        default="urn:noa:submodel:tagmapping:default",
        help="Submodel identifier (default: urn:noa:submodel:tagmapping:default)",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Use pretty printing for JSON/XML output (default: True)",
    )

    parser.add_argument(
        "--no-pretty",
        action="store_false",
        dest="pretty",
        help="Disable pretty printing",
    )

    args = parser.parse_args()

    # Determine format from extension if not specified
    if args.format is None:
        ext = args.output.suffix.lower().lstrip(".")
        if ext in ["json", "xml", "aasx"]:
            args.format = ext
        else:
            print(f"Error: Cannot infer format from extension '{ext}'", file=sys.stderr)
            return 1

    export_format = ExportFormat(args.format)

    # Load tags
    if args.input:
        if not args.input.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            return 1
        tags = load_tags_from_json(args.input)
        print(f"Loaded {len(tags)} tags from {args.input}")
    else:
        tags = create_sample_tags()
        print(f"Using {len(tags)} sample tags (no input file provided)")

    # Create submodel
    submodel = TagMappingSubmodel(submodel_id=args.submodel_id)
    for tag in tags:
        submodel.add_tag(tag)

    # Create AAS
    aas, sm = create_tag_mapping_aas(
        submodel=submodel,
        aas_id=args.aas_id,
        asset_id=args.asset_id,
    )

    # Export
    config = ExportConfig(pretty_print=args.pretty)
    exporter = AASExporter(config=config)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    exporter.export_to_file(aas, sm, args.output, export_format)

    # Print statistics
    stats = submodel.get_statistics()
    print(f"\nExported to: {args.output}")
    print(f"Format: {export_format.value.upper()}")
    print(f"\nTag Statistics:")
    print(f"  Total tags: {stats.total_tags}")
    print(f"  Mapped: {stats.mapped_tags}")
    print(f"  Verified: {stats.verified_tags}")
    print(f"  Pending: {stats.pending_tags}")
    print(f"  Conflicts: {stats.conflict_tags}")
    print(f"  Mapping rate: {stats.mapping_rate:.1%}")
    print(f"  Avg confidence: {stats.average_confidence:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
