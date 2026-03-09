#!/usr/bin/env python3
"""
Smoke test: Run MaxSATDACEnv with CSolver on ONE instance for 10 steps.
Verifies: reward varies, costs are non-negative, episodes have 10+ steps.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from gym_env import MaxSATDACEnv


def find_test_instance():
    """Find a small wcnf file to test with."""
    # Try generated first (small, fast)
    gen_dir = PROJECT_ROOT / "data" / "benchmarks" / "generated"
    if gen_dir.exists():
        files = sorted(gen_dir.glob("*.wcnf"))
        if files:
            return str(files[0])

    # Fallback: any wcnf
    bench_root = PROJECT_ROOT / "data" / "benchmarks"
    for wcnf in bench_root.rglob("*.wcnf"):
        return str(wcnf)

    raise FileNotFoundError("No .wcnf files found in data/benchmarks/")


def main():
    instance_path = find_test_instance()
    print(f"Test instance: {os.path.basename(instance_path)}")
    print(f"Full path: {instance_path}")
    print()

    env = MaxSATDACEnv(
        instance_paths=[instance_path],
        use_csolver=True,
        checkpoint_interval=1000,
        solver_timeout=10.0,
        max_steps=20,
        reward_type="shaped",
        seed=42,
    )

    try:
        obs, info = env.reset()
        print(f"Initial state:")
        print(f"  obs shape: {obs.shape}, obs: {obs}")
        print(f"  initial_cost: {info.get('initial_cost', 'N/A')}")
        print()

        rewards = []
        costs = []
        step = 0
        done = False

        while not done and step < 20:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            rewards.append(reward)
            cost = info.get("cost", float("inf"))
            best_cost = info.get("best_cost", float("inf"))
            costs.append(cost)

            print(f"  Step {step:2d}: reward={reward:+.6f}  cost={cost:.1f}  "
                  f"best={best_cost:.1f}  done={done}  "
                  f"obs[0:3]={obs[:3]}")

        print()
        print("=" * 60)
        print("SMOKE TEST RESULTS")
        print("=" * 60)

        # Check 1: Reward varies
        unique_rewards = len(set(f"{r:.6f}" for r in rewards))
        reward_varies = unique_rewards > 1
        print(f"  Reward varies:       {'PASS' if reward_varies else 'FAIL'} "
              f"({unique_rewards} unique values out of {len(rewards)} steps)")

        # Check 2: Costs are non-negative
        all_non_negative = all(c >= 0 for c in costs)
        print(f"  Costs non-negative:  {'PASS' if all_non_negative else 'FAIL'} "
              f"(min={min(costs):.1f}, max={max(costs):.1f})")

        # Check 3: Episode has 10+ steps
        enough_steps = step >= 10
        print(f"  Steps >= 10:         {'PASS' if enough_steps else 'FAIL'} "
              f"({step} steps)")

        # Check 4: Reward not constant 0.2
        all_02 = all(abs(r - 0.2) < 1e-6 for r in rewards)
        not_constant = not all_02
        print(f"  Not constant 0.2:    {'PASS' if not_constant else 'FAIL'}")

        print()
        all_pass = reward_varies and all_non_negative and enough_steps and not_constant
        print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
