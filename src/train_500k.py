#!/usr/bin/env python3
"""
Multi-seed 500K PPO training and evaluation for MaxSAT DAC.

Trains 3 PPO models (seeds 42, 123, 999) for 500K steps each,
evaluates on the same 18 test instances, and runs statistical tests.

Produces: data/results/csolver_500k_multiseed.json
"""

import json
import time
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from configs import (
    RESULTS_DIR, SEEDS, CHECKPOINT_INTERVAL, SOLVER_TIMEOUT,
    MAX_STEPS, LR, N_STEPS, BATCH_SIZE, N_EPOCHS, STATIC_CONFIGS,
    load_instance_splits,
)
from evaluation import (
    evaluate_ppo, evaluate_static, evaluate_random,
    run_statistical_tests, RewardLogger,
)
from gym_env import MaxSATDACEnv

TOTAL_TIMESTEPS = 500_000


def make_env(instance_paths, seed):
    """Create a monitored MaxSATDACEnv for training."""
    env = MaxSATDACEnv(
        instance_paths=instance_paths,
        use_csolver=True,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        solver_timeout=SOLVER_TIMEOUT,
        max_steps=MAX_STEPS,
        reward_type="shaped",
        seed=seed,
    )
    return Monitor(env)


def train_model(train_paths, seed, save_dir):
    """Train PPO for TOTAL_TIMESTEPS with given seed."""
    print(f"\n{'='*60}")
    print(f"Training PPO with seed={seed}, timesteps={TOTAL_TIMESTEPS}")
    print(f"{'='*60}")

    env = make_env(train_paths, seed)
    logger = RewardLogger()

    model = PPO(
        "MlpPolicy", env,
        learning_rate=LR, n_steps=N_STEPS,
        batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        seed=seed, verbose=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger.callback)
    train_time = time.time() - t0
    print(f"  Done in {train_time:.1f}s ({logger.episode_count} episodes)")

    model.save(str(save_dir / f"model_seed{seed}"))
    env.close()
    return model, train_time, logger.episode_rewards


def main():
    print(f"MaxSAT DAC — Multi-seed 500K PPO Experiment")
    print(f"Date: {datetime.now().isoformat()}")

    splits, _ = load_instance_splits()
    train_paths, test_paths = splits["train"], splits["test"]
    print(f"Train: {len(train_paths)}, Test: {len(test_paths)} instances")

    save_dir = RESULTS_DIR / "ppo_csolver_500k"
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "seeds": SEEDS,
            "timesteps": TOTAL_TIMESTEPS,
            "n_train": len(train_paths),
            "n_test": len(test_paths),
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "solver_timeout": SOLVER_TIMEOUT,
            "max_steps": MAX_STEPS,
        },
        "per_seed": {},
        "aggregate": {},
        "statistical_tests": {},
    }

    all_ppo, all_random = [], []
    all_static = {name: [] for name in STATIC_CONFIGS}
    total_start = time.time()

    for seed in SEEDS:
        print(f"\n{'#'*60}\n# SEED {seed}\n{'#'*60}")
        sr = {}

        # Train
        model, train_time, rewards = train_model(train_paths, seed, save_dir)
        sr["train_time"] = train_time
        sr["n_episodes"] = len(rewards)

        # Evaluate PPO
        ppo_costs = evaluate_ppo(model, test_paths, seed)
        sr["ppo_costs"] = ppo_costs
        sr["ppo_mean"] = float(np.mean(ppo_costs))
        print(f"  PPO mean cost: {sr['ppo_mean']:.1f}")
        all_ppo.extend(ppo_costs)

        # Evaluate random
        random_costs = evaluate_random(test_paths, seed)
        sr["random_costs"] = random_costs
        sr["random_mean"] = float(np.mean(random_costs))
        print(f"  Random mean cost: {sr['random_mean']:.1f}")
        all_random.extend(random_costs)

        # Evaluate static configs
        for cfg_name, cfg_params in STATIC_CONFIGS.items():
            costs = evaluate_static(cfg_params, test_paths, seed)
            sr[f"static_{cfg_name}_costs"] = costs
            sr[f"static_{cfg_name}_mean"] = float(np.mean(costs))
            print(f"  static_{cfg_name}: {sr[f'static_{cfg_name}_mean']:.1f}")
            all_static[cfg_name].extend(costs)

        # Per-seed stats
        sr["statistical_tests"] = {
            "ppo_vs_random": run_statistical_tests(ppo_costs, random_costs),
        }
        results["per_seed"][str(seed)] = sr

        # Save intermediate
        with open(RESULTS_DIR / "csolver_500k_multiseed.json", "w") as f:
            json.dump(results, f, indent=2)

    # Aggregate
    for label, costs in [("ppo", all_ppo), ("random", all_random)]:
        results["aggregate"][label] = {
            "mean": float(np.mean(costs)), "std": float(np.std(costs)),
            "costs": costs,
        }
    for cfg_name, costs in all_static.items():
        results["aggregate"][f"static_{cfg_name}"] = {
            "mean": float(np.mean(costs)), "std": float(np.std(costs)),
            "costs": costs,
        }

    # Aggregate tests
    agg = {"ppo_vs_random": run_statistical_tests(all_ppo, all_random)}
    for cfg_name in STATIC_CONFIGS:
        agg[f"ppo_vs_static_{cfg_name}"] = run_statistical_tests(all_ppo, all_static[cfg_name])
    results["statistical_tests"] = agg
    results["metadata"]["total_time_seconds"] = time.time() - total_start

    with open(RESULTS_DIR / "csolver_500k_multiseed.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"PPO:    {np.mean(all_ppo):.1f} ± {np.std(all_ppo):.1f}")
    print(f"Random: {np.mean(all_random):.1f} ± {np.std(all_random):.1f}")
    for name, costs in all_static.items():
        print(f"  {name:12s}: {np.mean(costs):.1f} ± {np.std(costs):.1f}")
    for test_name, tr in agg.items():
        p = tr.get("wilcoxon_p", tr.get("mann_whitney_p", "N/A"))
        sig = "***" if isinstance(p, float) and p < 0.001 else ""
        print(f"  {test_name}: diff={tr['diff']:.1f} p={p} {sig}")


if __name__ == "__main__":
    main()
