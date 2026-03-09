#!/usr/bin/env python3
"""
Ablation study for MaxSAT DAC: state features and action space.

Each variant trained 100K steps (seed=42), evaluated on 18 test instances.
Produces: data/results/ablation_study.json
"""

import json
import time
import numpy as np
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from configs import (
    RESULTS_DIR, CHECKPOINT_INTERVAL, SOLVER_TIMEOUT,
    MAX_STEPS, LR, N_STEPS, BATCH_SIZE, N_EPOCHS,
    load_instance_splits,
)
from evaluation import RewardLogger
from gym_env import MaxSATDACEnv

SEED = 42
TOTAL_TIMESTEPS = 100_000

# ── Ablation definitions ──────────────────────────────────────────

STATE_ABLATIONS = {
    "full":      {"desc": "All 10 features",        "keep": list(range(10))},
    "no_weights":{"desc": "Drop weight_mean/std",    "keep": [0,1,2,3,4,5,6,9]},
    "cost_only": {"desc": "step/cost/improve/plateau","keep": [0,1,2,6]},
    "minimal":   {"desc": "step + cost (2 features)","keep": [0,1]},
}

ACTION_ABLATIONS = {
    "full_4d":    {"desc": "All 4 actions",                "free": [0,1,2,3], "fixed": {}},
    "weight_only":{"desc": "Only h_inc, others default",   "free": [0],       "fixed": {1:-0.8, 2:-0.9, 3:-0.778}},
    "no_noise":   {"desc": "h_inc+smooth+hwm, noise fixed","free": [0,1,3],   "fixed": {2:-0.9}},
}


# ── Gymnasium wrappers ────────────────────────────────────────────

class StateFeatureMaskWrapper(gym.Wrapper):
    """Keep only selected observation indices."""
    def __init__(self, env, keep_indices):
        super().__init__(env)
        self.keep_indices = np.array(keep_indices, dtype=int)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(keep_indices),), dtype=np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return obs[self.keep_indices], info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs[self.keep_indices], r, term, trunc, info


class ActionMaskWrapper(gym.Wrapper):
    """Agent controls only free dimensions; fixed dims are spliced in."""
    def __init__(self, env, free_indices, fixed_values):
        super().__init__(env)
        self.free_indices = np.array(free_indices, dtype=int)
        self.fixed_values = fixed_values
        self.full_dim = env.action_space.shape[0]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(free_indices),), dtype=np.float32)

    def step(self, action):
        full = np.zeros(self.full_dim, dtype=np.float32)
        for idx, val in self.fixed_values.items():
            full[idx] = val
        for i, idx in enumerate(self.free_indices):
            full[idx] = action[i]
        return self.env.step(full)


# ── Helpers ───────────────────────────────────────────────────────

def make_env(paths, seed, state_keep=None, action_free=None, action_fixed=None):
    env = MaxSATDACEnv(
        instance_paths=paths, use_csolver=True,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        solver_timeout=SOLVER_TIMEOUT, max_steps=MAX_STEPS,
        reward_type="shaped", seed=seed,
    )
    if state_keep is not None and len(state_keep) < 10:
        env = StateFeatureMaskWrapper(env, state_keep)
    if action_free is not None and len(action_free) < 4:
        env = ActionMaskWrapper(env, action_free, action_fixed or {})
    return Monitor(env)


def evaluate_ablation(model, test_paths, seed, state_keep=None, action_free=None, action_fixed=None):
    costs = []
    for path in test_paths:
        env = MaxSATDACEnv(
            instance_paths=[path], use_csolver=True,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            solver_timeout=SOLVER_TIMEOUT, max_steps=MAX_STEPS,
            reward_type="shaped", seed=seed,
        )
        if state_keep is not None and len(state_keep) < 10:
            env = StateFeatureMaskWrapper(env, state_keep)
        if action_free is not None and len(action_free) < 4:
            env = ActionMaskWrapper(env, action_free, action_fixed or {})

        obs, info = env.reset()
        best = info.get("initial_cost", float("inf"))
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            if info.get("best_cost", float("inf")) < best:
                best = info["best_cost"]
            done = term or trunc
        env.close()
        costs.append(float(best))
    return costs


def train_variant(name, env, seed):
    logger = RewardLogger()
    model = PPO(
        "MlpPolicy", env, learning_rate=LR, n_steps=N_STEPS,
        batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, seed=seed, verbose=0,
    )
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger.callback)
    t = time.time() - t0
    print(f"  {name}: {t:.0f}s, {logger.episode_count} episodes")
    env.close()
    return model, t, logger.episode_rewards


# ── Main ──────────────────────────────────────────────────────────

def main():
    print(f"MaxSAT DAC — Ablation Study  (seed={SEED}, {TOTAL_TIMESTEPS} steps)")
    splits, _ = load_instance_splits()
    train_paths, test_paths = splits["train"], splits["test"]

    results = {
        "metadata": {"date": datetime.now().isoformat(), "seed": SEED,
                      "timesteps": TOTAL_TIMESTEPS, "n_test": len(test_paths)},
        "state_ablations": {}, "action_ablations": {},
    }

    # State feature ablation
    print(f"\n=== State Feature Ablation ===")
    for name, cfg in STATE_ABLATIONS.items():
        keep = cfg["keep"]
        env = make_env(train_paths, SEED, state_keep=keep)
        model, t, rews = train_variant(name, env, SEED)
        costs = evaluate_ablation(model, test_paths, SEED, state_keep=keep)
        results["state_ablations"][name] = {
            "description": cfg["desc"], "n_features": len(keep),
            "test_mean_cost": float(np.mean(costs)),
            "test_std_cost": float(np.std(costs)), "test_costs": costs,
            "train_time_s": round(t, 1),
        }
        print(f"    {name}: mean={np.mean(costs):.1f}")

    # Action space ablation
    print(f"\n=== Action Space Ablation ===")
    for name, cfg in ACTION_ABLATIONS.items():
        free, fixed = cfg["free"], cfg["fixed"]
        env = make_env(train_paths, SEED, action_free=free, action_fixed=fixed)
        model, t, rews = train_variant(name, env, SEED)
        costs = evaluate_ablation(model, test_paths, SEED, action_free=free, action_fixed=fixed)
        results["action_ablations"][name] = {
            "description": cfg["desc"], "n_free_actions": len(free),
            "test_mean_cost": float(np.mean(costs)),
            "test_std_cost": float(np.std(costs)), "test_costs": costs,
            "train_time_s": round(t, 1),
        }
        print(f"    {name}: mean={np.mean(costs):.1f}")

    with open(RESULTS_DIR / "ablation_study.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'ablation_study.json'}")


if __name__ == "__main__":
    main()
