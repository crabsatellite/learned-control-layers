#!/usr/bin/env python3
"""
Cross-solver zero-shot transfer: does NuWLS-trained PPO transfer to USW-LS?

Evaluates PPO (trained on NuWLS-DAC) on both NuWLS-DAC and USW-LS-DAC.
Produces: data/results/cross_solver_transfer.json
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from scipy.stats import wilcoxon

from stable_baselines3 import PPO

from configs import (
    RESULTS_DIR, BENCHMARK_DIR, CHECKPOINT_INTERVAL,
    SOLVER_TIMEOUT, MAX_STEPS, STATIC_CONFIGS, load_instance_splits,
)
from evaluation import solver_params_to_action
from gym_env import MaxSATDACEnv
from solver_wrapper import CSolver

_EXE = ".exe" if os.name == "nt" else ""
PROJECT_ROOT = RESULTS_DIR.parent.parent
NUWLS_BINARY = str(PROJECT_ROOT / "data" / "solvers" / "NuWLS" / "NuWLS-dac" / f"nuwls-dac{_EXE}")
USWLS_BINARY = str(PROJECT_ROOT / "data" / "solvers" / "USW-LS" / "USW-LS-dac" / f"usw-ls-dac{_EXE}")

MODEL_PATH = str(RESULTS_DIR / "ppo_csolver_500k" / "model_seed42")
SEED = 42


def make_env(instance_path, solver_binary):
    """Create env with a specific solver binary."""
    env = MaxSATDACEnv(
        instance_paths=[instance_path], use_csolver=True,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        solver_timeout=SOLVER_TIMEOUT, max_steps=MAX_STEPS,
        reward_type="shaped", seed=SEED,
    )
    env.solver.SOLVER_BINARY = solver_binary
    return env


def evaluate_on_solver(model, test_paths, solver_binary, policy="ppo"):
    """Run a policy on all test instances with a given solver. Returns per-instance results."""
    rng = np.random.default_rng(SEED)
    static_action = solver_params_to_action(STATIC_CONFIGS["default"])
    results = []
    for path in test_paths:
        env = make_env(path, solver_binary)
        obs, info = env.reset()
        best = info.get("initial_cost", float("inf"))
        done, step = False, 0
        while not done:
            if policy == "ppo":
                action, _ = model.predict(obs, deterministic=True)
            elif policy == "random":
                action = rng.uniform(-1, 1, size=4).astype(np.float32)
            else:
                action = static_action
            obs, _, term, trunc, info = env.step(action)
            if info.get("best_cost", float("inf")) < best:
                best = info["best_cost"]
            done = term or trunc
            step += 1
        env.close()
        results.append({"instance": os.path.basename(path), "best_cost": float(best), "steps": step})
    return results


def main():
    print("Cross-Solver Zero-Shot Transfer Experiment")
    splits, _ = load_instance_splits()
    test_paths = splits["test"]

    model = PPO.load(MODEL_PATH)
    solvers = {"NuWLS-DAC": NUWLS_BINARY, "USW-LS-DAC": USWLS_BINARY}
    all_results = {}

    for solver_name, binary in solvers.items():
        print(f"\n=== {solver_name} ===")
        sr = {}
        for policy in ["ppo", "random", "static_default"]:
            res = evaluate_on_solver(model, test_paths, binary, policy)
            mc = float(np.mean([r["best_cost"] for r in res]))
            sr[policy] = {"results": res, "mean_cost": mc}
            print(f"  {policy}: {mc:.1f}")
        all_results[solver_name] = sr

    # Transfer analysis
    usw = all_results["USW-LS-DAC"]
    ppo_costs = [r["best_cost"] for r in usw["ppo"]["results"]]
    rand_costs = [r["best_cost"] for r in usw["random"]["results"]]
    transfers = usw["ppo"]["mean_cost"] < usw["random"]["mean_cost"]
    print(f"\nTransfer: {'YES' if transfers else 'NO'}")

    output = {
        "metadata": {"date": datetime.now().isoformat(), "seed": SEED,
                      "model_path": MODEL_PATH, "n_test": len(test_paths)},
        "results": all_results,
        "transfer_analysis": {
            "usw_ppo_mean": usw["ppo"]["mean_cost"],
            "usw_random_mean": usw["random"]["mean_cost"],
            "transfers": transfers,
        },
    }
    with open(RESULTS_DIR / "cross_solver_transfer.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {RESULTS_DIR / 'cross_solver_transfer.json'}")


if __name__ == "__main__":
    main()
