#!/usr/bin/env python3
"""One-click setup: install deps, download solver, download benchmarks."""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def install_deps():
    """Install Python dependencies."""
    print("=== Installing Python dependencies ===")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r",
        str(PROJECT_ROOT / "requirements.txt")
    ])


def setup_solver():
    """Clone and compile NuWLS solver."""
    solver_dir = PROJECT_ROOT / "data" / "solvers"
    solver_dir.mkdir(parents=True, exist_ok=True)

    nuwls_dir = solver_dir / "NuWLS"
    if not nuwls_dir.exists():
        print("=== Cloning NuWLS solver ===")
        subprocess.check_call([
            "git", "clone", "https://github.com/filyouzicha/NuWLS.git",
            str(nuwls_dir)
        ])
    else:
        print("=== NuWLS already cloned ===")

    # Compile
    print("=== Compiling NuWLS ===")
    # Actual compile command depends on NuWLS Makefile structure
    # Will be updated after inspecting the repo
    print("NOTE: Manual compilation may be needed. Check NuWLS/README.md")


def setup_benchmarks():
    """Download MaxSAT Evaluation benchmarks."""
    bench_dir = PROJECT_ROOT / "data" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    print("=== Benchmark Setup ===")
    print("MaxSAT Evaluation benchmarks need to be downloaded from:")
    print("  https://maxsat-evaluations.github.io/2024/")
    print(f"Place WCNF files in: {bench_dir}/")
    print()
    print("Alternatively, run: python scripts/download_benchmarks.py")


def main():
    print("=" * 60)
    print("MaxSAT DAC — Project Setup")
    print("=" * 60)
    print()

    install_deps()
    print()
    setup_solver()
    print()
    setup_benchmarks()

    print()
    print("=" * 60)
    print("Setup complete!")
    print("Next steps:")
    print("  1. Compile NuWLS solver (check data/solvers/NuWLS/)")
    print("  2. Download benchmarks (python scripts/download_benchmarks.py)")
    print("  3. Run smoke test (python src/smoke_test.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
