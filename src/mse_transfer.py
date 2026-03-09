#!/usr/bin/env python3
"""
MSE Benchmark Zero-Shot Transfer Experiment.

Evaluates the 500K PPO model (trained on v50-v100 generated instances)
on MSE22/MSE23 benchmark instances in the 100-1000 clause range.

Produces: data/results/mse_transfer.json
"""

import json
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO

from configs import RESULTS_DIR, BENCHMARK_DIR, CHECKPOINT_INTERVAL, SOLVER_TIMEOUT
from evaluation import make_eval_env

MSE_DIR = BENCHMARK_DIR / "mse2024_regression"
MODEL_PATH = RESULTS_DIR / "ppo_csolver_500k" / "model_seed42"
SEED = 42


def find_medium_instances(min_clauses=100, max_clauses=1000):
    """Find MSE instances in the 100-1000 clause range."""
    instances = []
    for subdir in ["MSE22Big", "MSE23Big"]:
        dirpath = MSE_DIR / subdir
        if not dirpath.exists():
            continue
        for f in sorted(dirpath.glob("*.wcnf")):
            n_lines = sum(1 for _ in open(f))
            if min_clauses <= n_lines <= max_clauses:
                instances.append({
                    "path": str(f), "source": subdir,
                    "n_clauses": n_lines, "name": f.name[:12] + "...",
                })
    return instances


def evaluate_single(model, path, seed, use_model=True):
    """Run one instance, return detailed results."""
    try:
        env = make_eval_env(path, seed)
        obs, info = env.reset()
        initial_cost = info.get("initial_cost", float("inf"))
        best_cost = initial_cost
        total_reward, steps = 0, 0
        done = False
        rng = np.random.default_rng(seed)

        while not done:
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = rng.uniform(-1, 1, size=4).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if info.get("best_cost", float("inf")) < best_cost:
                best_cost = info["best_cost"]
            done = terminated or truncated
        env.close()

        return {
            "initial_cost": float(initial_cost),
            "best_cost": float(best_cost),
            "improvement_pct": float((initial_cost - best_cost) / max(initial_cost, 1) * 100),
            "steps": steps,
            "trivial": best_cost == 0 or initial_cost == 0,
            "no_progress": best_cost == initial_cost and initial_cost > 0,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("MaxSAT DAC — MSE Benchmark Zero-Shot Transfer\n")

    instances = find_medium_instances(100, 1000)
    print(f"Found {len(instances)} MSE instances with 100-1000 clauses")
    if not instances:
        print("ERROR: No suitable instances found")
        return

    step = max(1, len(instances) // 30)
    selected = instances[::step][:30]
    print(f"Selected {len(selected)} instances for evaluation\n")

    model = PPO.load(str(MODEL_PATH))

    results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "model": str(MODEL_PATH), "n_instances": len(selected),
            "clause_range": [100, 1000], "timeout": SOLVER_TIMEOUT,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
        },
        "instances": [],
    }

    ppo_costs, random_costs = [], []
    trivial_count = no_progress_count = error_count = dac_benefit_count = 0

    for i, inst in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] {inst['name']} ({inst['n_clauses']} clauses)")

        ppo_result = evaluate_single(model, inst["path"], SEED, use_model=True)
        random_result = evaluate_single(None, inst["path"], SEED, use_model=False)

        if ppo_result.get("error") or random_result.get("error"):
            error_count += 1
            results["instances"].append({"name": inst["name"], "error": ppo_result.get("error") or random_result.get("error")})
            continue

        is_trivial = ppo_result["trivial"] or random_result["trivial"]
        is_no_progress = ppo_result["no_progress"] and random_result["no_progress"]

        if is_trivial:
            trivial_count += 1
            category = "TRIVIAL"
        elif is_no_progress:
            no_progress_count += 1
            category = "NO_PROGRESS"
        else:
            category = "INTERESTING"
            ppo_costs.append(ppo_result["best_cost"])
            random_costs.append(random_result["best_cost"])
            if ppo_result["best_cost"] < random_result["best_cost"]:
                dac_benefit_count += 1

        print(f"  [{category}] PPO: {ppo_result['improvement_pct']:.1f}%, "
              f"Random: {random_result['improvement_pct']:.1f}%")

        results["instances"].append({
            "name": inst["name"], "source": inst["source"],
            "n_clauses": inst["n_clauses"], "category": category,
            "ppo": ppo_result, "random": random_result,
        })

    n_tested = len(selected) - error_count
    n_interesting = len(ppo_costs)

    results["summary"] = {
        "total_tested": n_tested, "trivial": trivial_count,
        "no_progress": no_progress_count, "interesting": n_interesting,
        "errors": error_count,
        "ppo_mean": float(np.mean(ppo_costs)) if n_interesting > 0 else None,
        "random_mean": float(np.mean(random_costs)) if n_interesting > 0 else None,
        "dac_benefit_count": dac_benefit_count,
    }

    with open(RESULTS_DIR / "mse_transfer.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nSUMMARY — MSE Benchmark Transfer\n{'='*60}")
    print(f"Tested: {n_tested}, Trivial: {trivial_count}, No progress: {no_progress_count}")
    print(f"Interesting: {n_interesting}")
    if n_interesting > 0:
        print(f"  PPO mean: {results['summary']['ppo_mean']:.1f}")
        print(f"  Random mean: {results['summary']['random_mean']:.1f}")
        print(f"  DAC better: {dac_benefit_count}/{n_interesting} instances")
    print(f"\nSaved to {RESULTS_DIR / 'mse_transfer.json'}")


if __name__ == "__main__":
    main()
