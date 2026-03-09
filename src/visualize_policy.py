#!/usr/bin/env python3
"""
Visualize the learned PPO policy's behavior over time.

Loads the 500K model, runs it on all test instances, records action trajectories,
and computes binned summary statistics showing the learned parameter schedule.

Produces: data/results/policy_trajectories.json
"""

import json
import numpy as np
from collections import defaultdict

from stable_baselines3 import PPO

from configs import RESULTS_DIR, PARAM_RANGES, load_instance_splits
from evaluation import make_eval_env

SEED = 42
PARAM_NAMES = list(PARAM_RANGES.keys())


def linear_map(x, low, high):
    return low + (x + 1.0) * 0.5 * (high - low)


def run_policy_on_instance(model, instance_path):
    """Run PPO deterministically, recording full trajectory."""
    env = make_eval_env(instance_path, SEED)
    obs, info = env.reset()
    initial_cost = info.get("initial_cost", float("inf"))
    trajectory = [{"step": 0, "step_fraction": float(obs[0]),
                    "cost": float(initial_cost), "best_cost": float(initial_cost)}]

    done, best_cost, step = False, initial_cost, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step += 1
        clipped = np.clip(action, -1.0, 1.0)
        mapped = {name: float(linear_map(clipped[i], *PARAM_RANGES[name]))
                  for i, name in enumerate(PARAM_NAMES)}

        obs, _, terminated, truncated, info = env.step(action)
        if info.get("best_cost", float("inf")) < best_cost:
            best_cost = info["best_cost"]
        done = terminated or truncated

        trajectory.append({
            "step": step, "step_fraction": float(obs[0]),
            "raw_action": [float(a) for a in action],
            **mapped,
            "cost": float(info.get("cost", float("inf"))),
            "best_cost": float(best_cost),
        })

    env.close()
    return trajectory, float(best_cost)


def compute_binned_summary(all_trajectories, n_bins=10):
    """Compute mean action values binned by step_fraction."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(n_bins)]
    bin_data = {label: defaultdict(list) for label in bin_labels}

    for traj in all_trajectories:
        for point in traj[1:]:  # skip initial state
            bin_idx = min(int(point["step_fraction"] * n_bins), n_bins - 1)
            label = bin_labels[bin_idx]
            for param in PARAM_NAMES:
                if param in point:
                    bin_data[label][param].append(point[param])

    summary = {}
    for label in bin_labels:
        summary[label] = {}
        for key in PARAM_NAMES:
            vals = bin_data[label].get(key, [])
            if vals:
                summary[label][key] = {
                    "mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
            else:
                summary[label][key] = {"mean": None, "std": None, "n": 0}
    return summary


def main():
    print("MaxSAT DAC — Policy Trajectory Visualization")

    model_path = RESULTS_DIR / "ppo_csolver_500k" / "model_seed42"
    model = PPO.load(str(model_path))

    splits, raw = load_instance_splits()
    test_paths = splits["test"]
    filenames = raw["test"]
    print(f"Running on {len(test_paths)} test instances\n")

    all_trajectories = []
    instance_summaries = []

    for i, (fname, path) in enumerate(zip(filenames, test_paths)):
        trajectory, best_cost = run_policy_on_instance(model, path)
        all_trajectories.append(trajectory)
        instance_summaries.append({"instance": fname, "best_cost": best_cost,
                                    "n_steps": len(trajectory) - 1})
        print(f"  [{i+1:2d}/{len(test_paths)}] {fname}: cost={best_cost:.0f}")

    summary = compute_binned_summary(all_trajectories)

    # Print schedule table
    print(f"\n{'Bin':>10s}  {'h_inc':>8s}  {'smooth_p':>8s}  {'noise_p':>8s}  {'hw_mult':>8s}")
    print("-" * 50)
    noise_values = []
    for bin_label, data in summary.items():
        h = data["h_inc"]["mean"]
        if h is not None:
            print(f"{bin_label:>10s}  {h:8.3f}  {data['smooth_prob']['mean']:8.4f}  "
                  f"{data['noise_prob']['mean']:8.4f}  {data['hard_weight_mult']['mean']:8.3f}")
            noise_values.append(data["noise_prob"]["mean"])

    if len(noise_values) >= 2:
        first = np.mean(noise_values[:len(noise_values)//2])
        second = np.mean(noise_values[len(noise_values)//2:])
        print(f"\nnoise_prob: first half {first:.4f} → second half {second:.4f} "
              f"({'decreasing' if second < first else 'increasing'})")

    costs = [s["best_cost"] for s in instance_summaries]
    print(f"\nMean cost: {np.mean(costs):.1f}, Median: {np.median(costs):.1f}")

    output = {
        "metadata": {"model": "ppo_csolver_500k/model_seed42", "seed": SEED,
                      "n_instances": len(test_paths)},
        "instance_summaries": instance_summaries,
        "trajectories": {fname: traj for fname, traj in zip(filenames, all_trajectories)},
        "binned_summary": summary,
    }
    output_path = RESULTS_DIR / "policy_trajectories.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
