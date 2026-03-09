#!/usr/bin/env python3
"""
Oracle schedule experiment: hand-crafted schedules vs PPO vs random.

Schedules: ppo_mimic (step function at 40%), linear, explore_exploit, optimal_static.
Produces: data/results/oracle_schedules.json
"""

import json
import time
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO

from configs import RESULTS_DIR, SOLVER_TIMEOUT, MAX_STEPS, CHECKPOINT_INTERVAL, load_instance_splits
from evaluation import params_to_action, evaluate_ppo, evaluate_random, evaluate_schedule

SEED = 42
MODEL_PATH = str(RESULTS_DIR / "ppo_csolver_500k" / "model_seed42")


# ── Schedule definitions ──────────────────────────────────────────

def oracle_ppo_mimic(step_frac):
    """Mimics the PPO trajectory: step function at 40%."""
    if step_frac < 0.4:
        return params_to_action(h_inc=0.1, smooth_prob=0.0, noise_prob=0.112, hard_weight_mult=0.5)
    return params_to_action(h_inc=9.63, smooth_prob=0.0, noise_prob=0.0, hard_weight_mult=0.5)


def oracle_linear(step_frac):
    """Linear exploration → exploitation."""
    return params_to_action(
        h_inc=0.1 + 9.9 * step_frac, smooth_prob=0.0,
        noise_prob=0.15 * (1.0 - step_frac), hard_weight_mult=0.5)


def oracle_explore_exploit(step_frac):
    """Simple 50/50 split: explore then exploit."""
    if step_frac < 0.5:
        return params_to_action(h_inc=1.0, smooth_prob=0.0, noise_prob=0.15, hard_weight_mult=0.5)
    return params_to_action(h_inc=5.0, smooth_prob=0.0, noise_prob=0.0, hard_weight_mult=0.5)


def oracle_optimal_static(step_frac):
    """Best static config (aggressive)."""
    return params_to_action(h_inc=3.0, smooth_prob=0.05, noise_prob=0.1, hard_weight_mult=3.0)


SCHEDULES = {
    "oracle_ppo_mimic": oracle_ppo_mimic,
    "oracle_linear": oracle_linear,
    "oracle_explore_exploit": oracle_explore_exploit,
    "oracle_optimal_static": oracle_optimal_static,
}


def main():
    print(f"MaxSAT DAC — Oracle Schedule Experiment  (seed={SEED})")
    splits, _ = load_instance_splits()
    test_paths = splits["test"]

    results = {
        "metadata": {"date": datetime.now().isoformat(), "seed": SEED,
                      "model_path": MODEL_PATH, "n_test": len(test_paths)},
        "schedules": {},
    }

    # PPO baseline
    model = PPO.load(MODEL_PATH)
    ppo_costs = evaluate_ppo(model, test_paths, SEED)
    results["schedules"]["ppo"] = {"mean": float(np.mean(ppo_costs)), "costs": ppo_costs}
    print(f"  PPO: {np.mean(ppo_costs):.1f}")

    # Random baseline
    random_costs = evaluate_random(test_paths, SEED)
    results["schedules"]["random"] = {"mean": float(np.mean(random_costs)), "costs": random_costs}
    print(f"  Random: {np.mean(random_costs):.1f}")

    # Oracle schedules
    for name, fn in SCHEDULES.items():
        costs = evaluate_schedule(fn, test_paths, SEED)
        results["schedules"][name] = {"mean": float(np.mean(costs)), "costs": costs}
        print(f"  {name}: {np.mean(costs):.1f}")

    with open(RESULTS_DIR / "oracle_schedules.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'oracle_schedules.json'}")


if __name__ == "__main__":
    main()
