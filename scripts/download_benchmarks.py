#!/usr/bin/env python3
"""
Download MaxSAT Evaluation benchmarks.

Sources:
  - MaxSAT Evaluation 2024: https://maxsat-evaluations.github.io/2024/
  - Helsinki benchmark collection: https://www.cs.helsinki.fi/group/coreo/benchmarks/
"""

import os
import sys
import json
import hashlib
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmarks"


# Known benchmark sources
BENCHMARK_SOURCES = {
    "mse2024_regression": {
        "url": "https://maxsat-evaluations.github.io/2024/MaxSATRegressionSuiteWCNFs.zip",
        "description": "MaxSAT Evaluation 2024 Regression Suite",
        "category": "mixed",
    },
}


def download_file(url, dest):
    """Download a file with progress indication."""
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")

    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Done: {os.path.getsize(dest) / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def extract_archive(archive_path, dest_dir):
    """Extract zip or tar archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if str(archive_path).endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(dest_dir)
    elif str(archive_path).endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tf:
            tf.extractall(dest_dir)
    elif str(archive_path).endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tf:
            tf.extractall(dest_dir)

    print(f"  Extracted to: {dest_dir}")


def create_split(benchmark_dir, train_frac=0.70, val_frac=0.15, seed=42):
    """Create train/validation/test split of benchmark instances."""
    import random

    wcnf_files = sorted(
        str(p.relative_to(benchmark_dir))
        for p in benchmark_dir.rglob("*.wcnf")
    )

    if not wcnf_files:
        print("WARNING: No .wcnf files found for splitting.")
        return

    random.seed(seed)
    random.shuffle(wcnf_files)

    n = len(wcnf_files)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    split = {
        "train": wcnf_files[:n_train],
        "validation": wcnf_files[n_train:n_train + n_val],
        "test": wcnf_files[n_train + n_val:],
    }

    split_file = benchmark_dir / "split.json"
    with open(split_file, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\nSplit created ({n} total instances):")
    print(f"  Train: {len(split['train'])}")
    print(f"  Validation: {len(split['validation'])}")
    print(f"  Test: {len(split['test'])}")
    print(f"  Saved to: {split_file}")


def main():
    print("=" * 60)
    print("MaxSAT DAC — Benchmark Download")
    print("=" * 60)
    print(f"Target directory: {BENCHMARK_DIR}")
    print()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    for name, info in BENCHMARK_SOURCES.items():
        print(f"\n--- {info['description']} ---")

        # Download
        ext = info["url"].split(".")[-1]
        archive_path = BENCHMARK_DIR / f"{name}.{ext}"

        if not archive_path.exists():
            success = download_file(info["url"], archive_path)
            if not success:
                continue
        else:
            print(f"  Already downloaded: {archive_path}")

        # Extract
        dest = BENCHMARK_DIR / name
        if not dest.exists():
            extract_archive(archive_path, dest)

    # Create split
    print("\n--- Creating train/val/test split ---")
    create_split(BENCHMARK_DIR)

    # Summary
    wcnf_count = len(list(BENCHMARK_DIR.rglob("*.wcnf")))
    print(f"\n{'=' * 60}")
    print(f"Total WCNF instances: {wcnf_count}")
    print(f"Benchmark directory: {BENCHMARK_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
