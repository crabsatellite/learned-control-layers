"""Shared configuration, constants, and path utilities for all experiments."""

import os
import json
from pathlib import Path

from solver_wrapper import SolverParams

# ── Paths ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.environ.get("MAXSAT_RESULTS_DIR", str(PROJECT_ROOT / "data" / "results")))
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmarks"

# ── Solver / Environment defaults ──────────────────────────────────

CHECKPOINT_INTERVAL = 1000
SOLVER_TIMEOUT = 10.0
MAX_STEPS = 50

# ── PPO training defaults ──────────────────────────────────────────

SEEDS = [42, 123, 999]
LR = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10

# ── Static baseline configurations ────────────────────────────────

STATIC_CONFIGS = {
    "default":    SolverParams(h_inc=1,  smooth_prob=0.01,  noise_prob=0.01,  hard_weight_mult=1),
    "wpms":       SolverParams(h_inc=28, smooth_prob=0.01,  noise_prob=0.01,  hard_weight_mult=1),
    "high_h":     SolverParams(h_inc=5,  smooth_prob=0.01,  noise_prob=0.01,  hard_weight_mult=2),
    "low_noise":  SolverParams(h_inc=1,  smooth_prob=0.001, noise_prob=0.001, hard_weight_mult=1),
    "aggressive": SolverParams(h_inc=3,  smooth_prob=0.05,  noise_prob=0.1,   hard_weight_mult=3),
}

# ── Action-space parameter ranges (from MaxSATDACEnv._decode_action) ──

PARAM_RANGES = {
    "h_inc":            (0.1, 10.0),
    "smooth_prob":      (0.0, 0.1),
    "noise_prob":       (0.0, 0.2),
    "hard_weight_mult": (0.5, 5.0),
}

# ── Path helpers ───────────────────────────────────────────────────

def resolve_instance_path(filename: str) -> str:
    """Find the full path for a benchmark filename."""
    for subdir in ["generated", "generated_hard", "generated_large"]:
        p = BENCHMARK_DIR / subdir / filename
        if p.exists():
            return str(p)
    for p in BENCHMARK_DIR.rglob(filename):
        return str(p)
    raise FileNotFoundError(f"Cannot find benchmark: {filename}")


def load_instance_splits():
    """Load train/val/test splits. Returns (dict of path lists, raw instance names)."""
    exp_path = RESULTS_DIR / "experiment_train_test_split.json"
    with open(exp_path) as f:
        data = json.load(f)
    splits = {}
    for split_name in ["train", "val", "test"]:
        filenames = data["instances"][split_name]
        splits[split_name] = [resolve_instance_path(fn) for fn in filenames]
    return splits, data["instances"]
