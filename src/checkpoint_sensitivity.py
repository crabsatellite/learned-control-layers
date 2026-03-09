#!/usr/bin/env python3
"""
Checkpoint frequency sensitivity: tests intervals [250, 500, 1000, 2000].

Each interval: train PPO 100K (seed=42), evaluate on 18 test instances.
Produces: data/results/checkpoint_sensitivity.json
"""

import json
import time
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from configs import (
    RESULTS_DIR, SOLVER_TIMEOUT, MAX_STEPS,
    LR, N_STEPS, BATCH_SIZE, N_EPOCHS, load_instance_splits,
)
from evaluation import evaluate_ppo, evaluate_random, RewardLogger
from gym_env import MaxSATDACEnv

CHECKPOINT_INTERVALS = [250, 500, 1000, 2000]
SEED = 42
TOTAL_TIMESTEPS = 100_000


def main():
    print(f"MaxSAT DAC — Checkpoint Sensitivity  (seed={SEED})")
    splits, _ = load_instance_splits()
    train_paths, test_paths = splits["train"], splits["test"]

    results = {
        "metadata": {"date": datetime.now().isoformat(), "seed": SEED,
                      "timesteps": TOTAL_TIMESTEPS, "intervals": CHECKPOINT_INTERVALS},
        "results": {},
    }

    for interval in CHECKPOINT_INTERVALS:
        print(f"\n=== Interval = {interval} flips ===")

        # Train
        env = Monitor(MaxSATDACEnv(
            instance_paths=train_paths, use_csolver=True,
            checkpoint_interval=interval, solver_timeout=SOLVER_TIMEOUT,
            max_steps=MAX_STEPS, reward_type="shaped", seed=SEED,
        ))
        logger = RewardLogger()
        model = PPO("MlpPolicy", env, learning_rate=LR, n_steps=N_STEPS,
                     batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, seed=SEED, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger.callback)
        train_time = time.time() - t0
        env.close()

        # Evaluate
        ppo_costs = evaluate_ppo(model, test_paths, SEED, checkpoint_interval=interval)
        random_costs = evaluate_random(test_paths, SEED, checkpoint_interval=interval)

        adv = float(np.mean(random_costs) - np.mean(ppo_costs))
        results["results"][str(interval)] = {
            "ppo_mean": float(np.mean(ppo_costs)), "random_mean": float(np.mean(random_costs)),
            "ppo_advantage": adv, "ppo_costs": ppo_costs, "random_costs": random_costs,
            "train_time": round(train_time, 1),
        }
        print(f"  PPO={np.mean(ppo_costs):.1f}  Random={np.mean(random_costs):.1f}  Δ={adv:.1f}")

    with open(RESULTS_DIR / "checkpoint_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'checkpoint_sensitivity.json'}")


if __name__ == "__main__":
    main()
