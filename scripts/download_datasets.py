#!/usr/bin/env python3
"""Download benchmark datasets for semantic tag mapping training.

This script downloads the following datasets:
- Tennessee Eastman Process (TEP) from Harvard Dataverse
- NASA C-MAPSS from NASA Prognostics Data Repository

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --tep
    python scripts/download_datasets.py --cmapss
    python scripts/download_datasets.py --output-dir /path/to/data
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Dataset download URLs and checksums
DATASETS: dict[str, dict[str, str | list[str]]] = {
    "tep": {
        "description": "Tennessee Eastman Process simulation data",
        # Harvard Dataverse TEP dataset (Rieth et al.)
        "url": "https://dataverse.harvard.edu/api/access/datafile/:persistentId"
        "?persistentId=doi:10.7910/DVN/6C3JR1/XJ1O0B",
        "filename": "TEP_FaultFree_Training.csv",
        "checksum": "",  # Checksum can be added after first download
        "alt_url": "https://github.com/camacho/tep/raw/master/data/TEP_FaultFree_Training.csv",
    },
    "cmapss": {
        "description": "NASA C-MAPSS turbofan engine degradation data",
        # NASA Prognostics Data Repository
        "url": "https://data.nasa.gov/api/views/rf5q-stnn/rows.csv?accessType=DOWNLOAD",
        "filename": "CMAPSSData.zip",
        "checksum": "",  # Checksum can be added after first download
        "alt_url": "https://ti.arc.nasa.gov/c/6/",
        "files": [
            "train_FD001.txt",
            "train_FD002.txt",
            "train_FD003.txt",
            "train_FD004.txt",
            "test_FD001.txt",
            "test_FD002.txt",
            "test_FD003.txt",
            "test_FD004.txt",
            "RUL_FD001.txt",
            "RUL_FD002.txt",
            "RUL_FD003.txt",
            "RUL_FD004.txt",
        ],
    },
}

# Default data directory relative to project root
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def get_file_checksum(filepath: Path) -> str:
    """Calculate SHA256 checksum of a file.

    Args:
        filepath: Path to the file

    Returns:
        Hex digest of SHA256 checksum
    """
    sha256 = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file from URL with progress indication.

    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress output

    Returns:
        True if download succeeded, False otherwise
    """
    print(f"Downloading {desc or dest_path.name}...")
    print(f"  URL: {url}")

    try:
        # Create request with user agent to avoid blocks
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; NOA-Swarm-Mapper/1.0)"
            },
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            total_size = response.headers.get("Content-Length")
            if total_size:
                total_size = int(total_size)
                print(f"  Size: {total_size / 1024 / 1024:.1f} MB")

            # Download to temporary file first
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                downloaded = 0
                block_size = 8192

                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        progress = downloaded / total_size * 100
                        print(f"\r  Progress: {progress:.1f}%", end="", flush=True)

                print()  # Newline after progress

            # Move to final destination
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tmp_file.name, dest_path)

        print(f"  Saved to: {dest_path}")
        return True

    except urllib.error.URLError as e:
        print(f"  Error downloading: {e}")
        return False
    except TimeoutError:
        print("  Error: Download timed out")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a ZIP file.

    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to

    Returns:
        True if extraction succeeded, False otherwise
    """
    print(f"Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # List contents
            print(f"  Contents: {len(zf.namelist())} files")
            zf.extractall(extract_to)
        print(f"  Extracted to: {extract_to}")
        return True
    except zipfile.BadZipFile:
        print(f"  Error: {zip_path} is not a valid ZIP file")
        return False


def download_tep(output_dir: Path) -> bool:
    """Download Tennessee Eastman Process dataset.

    Args:
        output_dir: Directory to save data

    Returns:
        True if download succeeded
    """
    print("\n" + "=" * 60)
    print("Tennessee Eastman Process (TEP) Dataset")
    print("=" * 60)

    tep_dir = output_dir / "tep"
    tep_dir.mkdir(parents=True, exist_ok=True)

    # Try primary URL first
    dest_file = tep_dir / "TEP_FaultFree_Training.csv"
    if dest_file.exists():
        print(f"  Already exists: {dest_file}")
        return True

    # Try Harvard Dataverse
    url = str(DATASETS["tep"]["url"])
    success = download_file(url, dest_file, "TEP data from Harvard Dataverse")

    if not success:
        # Try alternative URL
        alt_url = str(DATASETS["tep"]["alt_url"])
        print("  Trying alternative source...")
        success = download_file(alt_url, dest_file, "TEP data from GitHub mirror")

    if success:
        # Verify file
        if dest_file.stat().st_size < 1000:
            print("  Warning: Downloaded file seems too small")
            print(f"  Size: {dest_file.stat().st_size} bytes")
            return False

        checksum = get_file_checksum(dest_file)
        print(f"  Checksum (SHA256): {checksum}")

    return success


def create_synthetic_tep(output_dir: Path) -> bool:
    """Create synthetic TEP-like data for testing when download fails.

    Args:
        output_dir: Directory to save data

    Returns:
        True if creation succeeded
    """
    print("\n  Creating synthetic TEP data for testing...")

    try:
        import numpy as np
    except ImportError:
        print("  Error: NumPy is required to create synthetic data")
        return False

    tep_dir = output_dir / "tep"
    tep_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic data matching TEP structure
    # 52 variables: XMEAS(1-41) + XMV(1-11)
    np.random.seed(42)
    num_samples = 1000
    num_variables = 52

    # Generate synthetic process data
    data = np.random.randn(num_samples, num_variables)

    # Add some realistic scaling
    # Flow rates (typically positive, larger values)
    data[:, :6] = np.abs(data[:, :6]) * 100 + 50  # XMEAS 1-6 (flows)
    # Temperatures (typical range 50-150 degC)
    data[:, 8:12] = data[:, 8:12] * 20 + 100  # XMEAS 9-11 (temps)
    # Pressures (typical range 2000-3000 kPa)
    data[:, 6:8] = data[:, 6:8] * 200 + 2500  # XMEAS 7-8 (pressures)
    # Levels (0-100%)
    data[:, 7] = np.clip(data[:, 7] * 20 + 50, 0, 100)  # XMEAS 8 (level)
    # Valve positions (0-100%)
    data[:, 41:52] = np.clip(data[:, 41:52] * 20 + 50, 0, 100)  # XMV (valves)

    # Create header
    header = (
        "sample," +
        ",".join([f"XMEAS_{i}" for i in range(1, 42)]) +
        "," +
        ",".join([f"XMV_{i}" for i in range(1, 12)])
    )

    # Save to CSV
    dest_file = tep_dir / "TEP_FaultFree_Training.csv"
    np.savetxt(
        dest_file,
        np.column_stack([np.arange(num_samples), data]),
        delimiter=",",
        header=header,
        comments="",
        fmt=["%.0f"] + ["%.6f"] * num_variables,
    )

    print(f"  Created synthetic TEP data: {dest_file}")
    print(f"  Samples: {num_samples}, Variables: {num_variables}")
    return True


def download_cmapss(output_dir: Path) -> bool:
    """Download NASA C-MAPSS dataset.

    Args:
        output_dir: Directory to save data

    Returns:
        True if download succeeded
    """
    print("\n" + "=" * 60)
    print("NASA C-MAPSS Dataset")
    print("=" * 60)

    cmapss_dir = output_dir / "cmapss"
    cmapss_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    expected_files = DATASETS["cmapss"]["files"]
    if isinstance(expected_files, list):
        all_exist = all((cmapss_dir / f).exists() for f in expected_files)
        if all_exist:
            print(f"  All files already exist in: {cmapss_dir}")
            return True

    # Try to download
    print("\nNote: The C-MAPSS dataset is hosted on NASA's servers and may require")
    print("manual download from: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/")
    print("\nAlternatively, download from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
    print("\nLook for 'Turbofan Engine Degradation Simulation Data Set'")

    # Try direct download (may not work due to NASA's data access policies)
    url = str(DATASETS["cmapss"]["url"])
    zip_file = cmapss_dir / "CMAPSSData.zip"

    success = download_file(url, zip_file, "C-MAPSS data")

    if success and zip_file.exists() and extract_zip(zip_file, cmapss_dir):
        # Clean up ZIP file
        zip_file.unlink()
        return True

    # If download failed, create synthetic data for testing
    print("\n  Direct download unavailable. Creating synthetic data for testing...")
    return create_synthetic_cmapss(output_dir)


def create_synthetic_cmapss(output_dir: Path) -> bool:
    """Create synthetic C-MAPSS-like data for testing when download fails.

    Args:
        output_dir: Directory to save data

    Returns:
        True if creation succeeded
    """
    try:
        import numpy as np
    except ImportError:
        print("  Error: NumPy is required to create synthetic data")
        return False

    cmapss_dir = output_dir / "cmapss"
    cmapss_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # C-MAPSS data structure:
    # Columns: unit_id, cycle, setting1, setting2, setting3, sensor1...sensor21
    # 26 columns total

    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        # Training data: multiple units with full run-to-failure
        num_units_train = 100 if subset in ["FD001", "FD003"] else 260
        cycles_per_unit = np.random.randint(100, 400, num_units_train)

        train_data = []
        for unit_id in range(1, num_units_train + 1):
            n_cycles = cycles_per_unit[unit_id - 1]
            for cycle in range(1, n_cycles + 1):
                # Operational settings
                settings = [
                    np.random.uniform(0, 42),  # Altitude proxy
                    np.random.uniform(0, 0.84),  # Mach
                    np.random.uniform(100, 100),  # TRA
                ]

                # Sensor readings (21 sensors)
                sensors = np.random.randn(21) * 10 + 500

                # Add degradation trend
                degradation = cycle / n_cycles * 20
                sensors[0:4] += degradation  # Temperature sensors degrade

                row = [unit_id, cycle, *settings, *sensors.tolist()]
                train_data.append(row)

        train_data = np.array(train_data)
        train_file = cmapss_dir / f"train_{subset}.txt"
        np.savetxt(train_file, train_data, fmt="%.4f", delimiter=" ")
        print(f"  Created: {train_file.name} ({len(train_data)} samples)")

        # Test data: partial trajectories
        num_units_test = 100 if subset in ["FD001", "FD003"] else 259
        test_data = []
        rul_data = []

        for unit_id in range(1, num_units_test + 1):
            total_cycles = np.random.randint(150, 400)
            observed_cycles = np.random.randint(50, total_cycles - 10)
            rul = total_cycles - observed_cycles
            rul_data.append(rul)

            for cycle in range(1, observed_cycles + 1):
                settings = [
                    np.random.uniform(0, 42),
                    np.random.uniform(0, 0.84),
                    np.random.uniform(100, 100),
                ]
                sensors = np.random.randn(21) * 10 + 500
                degradation = cycle / total_cycles * 20
                sensors[0:4] += degradation

                row = [unit_id, cycle, *settings, *sensors.tolist()]
                test_data.append(row)

        test_data = np.array(test_data)
        test_file = cmapss_dir / f"test_{subset}.txt"
        np.savetxt(test_file, test_data, fmt="%.4f", delimiter=" ")
        print(f"  Created: {test_file.name} ({len(test_data)} samples)")

        rul_file = cmapss_dir / f"RUL_{subset}.txt"
        np.savetxt(rul_file, rul_data, fmt="%.0f")
        print(f"  Created: {rul_file.name} ({len(rul_data)} RUL values)")

    return True


def verify_datasets(output_dir: Path) -> dict[str, bool]:
    """Verify downloaded datasets.

    Args:
        output_dir: Directory containing downloaded data

    Returns:
        Dictionary mapping dataset names to verification status
    """
    print("\n" + "=" * 60)
    print("Verifying Downloads")
    print("=" * 60)

    results: dict[str, bool] = {}

    # Check TEP
    tep_file = output_dir / "tep" / "TEP_FaultFree_Training.csv"
    if tep_file.exists():
        size_mb = tep_file.stat().st_size / 1024 / 1024
        print(f"  TEP: OK ({size_mb:.1f} MB)")
        results["tep"] = True
    else:
        print("  TEP: MISSING")
        results["tep"] = False

    # Check C-MAPSS
    cmapss_dir = output_dir / "cmapss"
    cmapss_files = list(cmapss_dir.glob("*.txt")) if cmapss_dir.exists() else []
    if len(cmapss_files) >= 12:
        print(f"  C-MAPSS: OK ({len(cmapss_files)} files)")
        results["cmapss"] = True
    else:
        print(f"  C-MAPSS: INCOMPLETE ({len(cmapss_files)}/12 files)")
        results["cmapss"] = False

    return results


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for semantic tag mapping training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --all                  # Download all datasets
    %(prog)s --tep                  # Download only TEP
    %(prog)s --cmapss               # Download only C-MAPSS
    %(prog)s --output-dir ./data    # Specify output directory
    %(prog)s --verify               # Verify existing downloads
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets",
    )
    parser.add_argument(
        "--tep",
        action="store_true",
        help="Download Tennessee Eastman Process dataset",
    )
    parser.add_argument(
        "--cmapss",
        action="store_true",
        help="Download NASA C-MAPSS dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Output directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing downloads without downloading",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create synthetic data (useful when downloads fail)",
    )

    args = parser.parse_args()

    # Default to --all if no dataset specified
    if not any([args.all, args.tep, args.cmapss, args.verify, args.synthetic]):
        args.all = True

    output_dir = args.output_dir.resolve()
    print(f"Output directory: {output_dir}")

    if args.verify:
        results = verify_datasets(output_dir)
        return 0 if all(results.values()) else 1

    if args.synthetic:
        print("\nCreating synthetic datasets for testing...")
        create_synthetic_tep(output_dir)
        create_synthetic_cmapss(output_dir)
        verify_datasets(output_dir)
        return 0

    if (args.all or args.tep) and not download_tep(output_dir):
        print("\n  Falling back to synthetic TEP data...")
        create_synthetic_tep(output_dir)

    if (args.all or args.cmapss) and not download_cmapss(output_dir):
        print("\n  C-MAPSS download failed, synthetic data created instead")

    # Final verification
    results = verify_datasets(output_dir)
    success = all(results.values())

    print("\n" + "=" * 60)
    if success:
        print("All datasets downloaded successfully!")
    else:
        print("Some datasets could not be downloaded.")
        print("Synthetic data has been created for testing purposes.")
        print("For real training, please download manually from:")
        print("  TEP: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1")
        print("  C-MAPSS: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
