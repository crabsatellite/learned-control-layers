#!/usr/bin/env python3
"""
Scale experiment: Train PPO on larger instances (v200, v500) and compare vs random.
Also evaluates zero-shot transfer of existing 500K model trained on v50/v75/v100.

Produces: data/results/scale_experiment.json
"""

import json
import time
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from configs import (
    RESULTS_DIR, BENCHMARK_DIR, CHECKPOINT_INTERVAL, MAX_STEPS,
    LR, N_STEPS, BATCH_SIZE, N_EPOCHS,
)
from evaluation import evaluate_ppo, evaluate_random, run_statistical_tests, RewardLogger
from gym_env import MaxSATDACEnv

SEED = 42
TOTAL_TIMESTEPS = 100_000

# Per-size instance splits (generated_large directory)
_LARGE_DIR = BENCHMARK_DIR / "generated_large"
SIZES = {
    200: {
        "train": [str(_LARGE_DIR / f"v200_{i:03d}.wcnf") for i in range(20)],
        "test":  [str(_LARGE_DIR / f"v200_{i:03d}.wcnf") for i in range(25, 30)],
    },
    500: {
        "train": [str(_LARGE_DIR / f"v500_{i:03d}.wcnf") for i in range(20)],
        "test":  [str(_LARGE_DIR / f"v500_{i:03d}.wcnf") for i in range(25, 30)],
    },
}

# Solver timeout: 10s for v200, 30s for v500
TIMEOUTS = {200: 10.0, 500: 30.0}


def main():
    print(f"MaxSAT DAC — Scale Experiment")
    print(f"Sizes: {list(SIZES.keys())}, Timesteps: {TOTAL_TIMESTEPS}, Seed: {SEED}\n")

    save_dir = RESULTS_DIR / "scale_experiment"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "metadata": {
            "date": datetime.now().isoformat(), "seed": SEED,
            "timesteps": TOTAL_TIMESTEPS, "checkpoint_interval": CHECKPOINT_INTERVAL,
            "max_steps": MAX_STEPS, "sizes": list(SIZES.keys()),
        },
        "per_size": {},
        "zero_shot_transfer": {},
    }

    total_start = time.time()

    for n_vars in SIZES:
        timeout = TIMEOUTS[n_vars]
        splits = SIZES[n_vars]
        print(f"\n{'='*60}\nSIZE: v{n_vars} (timeout={timeout}s)\n{'='*60}")

        sr = {"n_vars": n_vars, "timeout": timeout}

        # Train PPO
        print(f"  Training PPO (100K steps)...")
        env = Monitor(MaxSATDACEnv(
            instance_paths=splits["train"], use_csolver=True,
            checkpoint_interval=CHECKPOINT_INTERVAL, solver_timeout=timeout,
            max_steps=MAX_STEPS, reward_type="shaped", seed=SEED,
        ))
        logger = RewardLogger()
        model = PPO("MlpPolicy", env, learning_rate=LR, n_steps=N_STEPS,
                     batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, seed=SEED, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger.callback)
        sr["train_time"] = time.time() - t0
        sr["n_episodes"] = logger.episode_count
        env.close()
        model.save(str(save_dir / f"model_v{n_vars}_seed{SEED}"))

        # Evaluate PPO and Random
        ppo_costs = evaluate_ppo(model, splits["test"], SEED, solver_timeout=timeout)
        random_costs = evaluate_random(splits["test"], SEED, solver_timeout=timeout)

        sr["ppo_costs"] = ppo_costs
        sr["ppo_mean"] = float(np.mean(ppo_costs))
        sr["random_costs"] = random_costs
        sr["random_mean"] = float(np.mean(random_costs))
        sr["ppo_vs_random"] = run_statistical_tests(ppo_costs, random_costs)
        print(f"  PPO: {sr['ppo_mean']:.1f}  Random: {sr['random_mean']:.1f}  "
              f"Δ={sr['ppo_vs_random']['diff']:.1f}")

        all_results["per_size"][str(n_vars)] = sr

    # Zero-shot transfer from existing 500K model
    print(f"\n{'='*60}\nZERO-SHOT TRANSFER: 500K model -> large instances\n{'='*60}")
    existing_model_path = RESULTS_DIR / "ppo_csolver_500k" / "model_seed42"
    if existing_model_path.with_suffix(".zip").exists():
        existing_model = PPO.load(str(existing_model_path))
        for n_vars in SIZES:
            timeout = TIMEOUTS[n_vars]
            test_paths = SIZES[n_vars]["test"]
            transfer_costs = evaluate_ppo(existing_model, test_paths, SEED, solver_timeout=timeout)
            random_costs = all_results["per_size"][str(n_vars)]["random_costs"]
            size_ppo_costs = all_results["per_size"][str(n_vars)]["ppo_costs"]

            tr = {
                "costs": transfer_costs,
                "mean": float(np.mean(transfer_costs)),
                "vs_random": run_statistical_tests(transfer_costs, random_costs),
                "vs_size_specific_ppo": run_statistical_tests(transfer_costs, size_ppo_costs),
            }
            print(f"  v{n_vars}: transfer={tr['mean']:.1f}, "
                  f"vs_random Δ={tr['vs_random']['diff']:.1f}")
            all_results["zero_shot_transfer"][str(n_vars)] = tr
    else:
        print(f"  WARNING: No existing 500K model found")
        all_results["zero_shot_transfer"]["error"] = "Model not found"

    all_results["metadata"]["total_time_seconds"] = time.time() - total_start

    with open(RESULTS_DIR / "scale_experiment.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'scale_experiment.json'}")


if __name__ == "__main__":
    main()
