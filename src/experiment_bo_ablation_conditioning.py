#!/usr/bin/env python3
"""
Extended experiments for reviewer requests:
1. SMAC-equivalent Bayesian optimization baseline
2. 500K ablation with matched seeds (controlled feature importance)
3. Learning curves during training
4. Reward function ablation
5. State-conditioning evidence (policy behavior stratified by state)

Produces: data/results/experiment_bo_ablation_conditioning.json
"""

import os
import json
import time
import numpy as np
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from scipy.stats import mannwhitneyu

from configs import (
    RESULTS_DIR, SEEDS, CHECKPOINT_INTERVAL, SOLVER_TIMEOUT, MAX_STEPS,
    LR, N_STEPS, BATCH_SIZE, N_EPOCHS, load_instance_splits,
)
from evaluation import evaluate_ppo, evaluate_static, make_eval_env
from gym_env import MaxSATDACEnv
from solver_wrapper import SolverParams


# ── Wrappers ─────────────────────────────────────────────────────

class StateFeatureMaskWrapper(gym.Wrapper):
    """Mask out state features for ablation studies."""
    def __init__(self, env, keep_indices):
        super().__init__(env)
        self.keep_indices = np.array(keep_indices, dtype=int)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self.keep_indices),), dtype=np.float32)

    def _filter_obs(self, obs):
        return obs[self.keep_indices]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._filter_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._filter_obs(obs), reward, terminated, truncated, info


class LearningCurveCallback(BaseCallback):
    """Records episode rewards and periodic evaluation checkpoints."""
    def __init__(self, eval_paths, eval_seed, eval_interval=10000, verbose=0):
        super().__init__(verbose)
        self.eval_paths = eval_paths
        self.eval_seed = eval_seed
        self.eval_interval = eval_interval
        self.checkpoints = []
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_count += 1

        if self.num_timesteps % self.eval_interval == 0 and self.num_timesteps > 0:
            costs = evaluate_ppo(self.model, self.eval_paths[:6], self.eval_seed)
            mean_cost = float(np.mean(costs))
            self.checkpoints.append({
                "timestep": self.num_timesteps, "mean_cost": mean_cost,
                "mean_reward": float(np.mean(self.episode_rewards[-50:])) if self.episode_rewards else 0,
            })
            print(f"    [{self.num_timesteps}] eval cost={mean_cost:.1f}")
        return True


# ── Experiment 1: Bayesian Optimization Baseline ─────────────────

def experiment_bayesian_optimization(splits):
    """SMAC-equivalent: Bayesian optimization over static configs."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Bayesian Optimization Static Configuration Baseline")
    print("="*70)

    from skopt import gp_minimize
    from skopt.space import Real

    train_paths = splits["train"][:20]
    test_paths = splits["test"]

    search_space = [
        Real(0.1, 10.0, name="h_inc"),
        Real(0.0, 0.1, name="smooth_prob"),
        Real(0.0, 0.2, name="noise_prob"),
        Real(0.5, 5.0, name="hard_weight_mult"),
    ]

    eval_count = [0]
    eval_log = []

    def objective(x):
        params = SolverParams(h_inc=x[0], smooth_prob=x[1],
                             noise_prob=x[2], hard_weight_mult=x[3])
        costs = evaluate_static(params, train_paths, seed=42)
        mean_cost = float(np.mean(costs))
        eval_count[0] += 1
        eval_log.append({"params": [float(v) for v in x], "train_mean": mean_cost})
        if eval_count[0] % 20 == 0:
            print(f"  [{eval_count[0]}] best so far: {min(e['train_mean'] for e in eval_log):.1f}")
        return mean_cost

    t0 = time.time()
    result = gp_minimize(objective, search_space, n_calls=100, n_initial_points=20,
                         random_state=42, verbose=False)
    bo_time = time.time() - t0

    best_params = SolverParams(
        h_inc=result.x[0], smooth_prob=result.x[1],
        noise_prob=result.x[2], hard_weight_mult=result.x[3])

    print(f"\nBO complete in {bo_time:.0f}s, {eval_count[0]} evaluations")
    print(f"Best config: h={best_params.h_inc:.3f}, s={best_params.smooth_prob:.4f}, "
          f"n={best_params.noise_prob:.4f}, w={best_params.hard_weight_mult:.3f}")

    # Evaluate best BO config on test set with all 3 seeds
    all_test_costs = []
    per_seed = {}
    for seed in SEEDS:
        costs = evaluate_static(best_params, test_paths, seed)
        all_test_costs.extend(costs)
        per_seed[str(seed)] = {"costs": costs, "mean": float(np.mean(costs))}
        print(f"  Test seed {seed}: mean={np.mean(costs):.1f}")

    # Random search with same budget for comparison
    print("\nRandom search with same budget (100 configs)...")
    rng = np.random.default_rng(42)
    random_configs = []
    for _ in range(100):
        params = SolverParams(
            h_inc=rng.uniform(0.1, 10.0), smooth_prob=rng.uniform(0.0, 0.1),
            noise_prob=rng.uniform(0.0, 0.2), hard_weight_mult=rng.uniform(0.5, 5.0))
        costs = evaluate_static(params, train_paths, seed=42)
        random_configs.append({"params": params, "train_mean": float(np.mean(costs))})

    best_random = min(random_configs, key=lambda c: c["train_mean"])
    random_test_costs = []
    for seed in SEEDS:
        costs = evaluate_static(best_random["params"], test_paths, seed)
        random_test_costs.extend(costs)

    return {
        "bo_config": {k: float(getattr(best_params, k)) for k in ["h_inc", "smooth_prob", "noise_prob", "hard_weight_mult"]},
        "bo_train_mean": float(result.fun),
        "bo_test_mean": float(np.mean(all_test_costs)),
        "bo_test_std": float(np.std(all_test_costs)),
        "bo_test_costs": all_test_costs,
        "bo_per_seed": per_seed,
        "bo_n_evals": eval_count[0],
        "bo_time_s": bo_time,
        "random_100_test_mean": float(np.mean(random_test_costs)),
        "random_100_test_costs": random_test_costs,
        "convergence": [{"iteration": i+1, "best_train": float(min(result.func_vals[:i+1]))}
                        for i in range(len(result.func_vals))],
    }


# ── Experiment 2: 500K Ablation with Matched Seeds ──────────────

def experiment_matched_ablation(splits):
    """Train full vs no_weights at 500K with SAME seeds."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Matched-Seed 500K Feature Ablation")
    print("="*70)

    train_paths, test_paths = splits["train"], splits["test"]
    results = {}

    for variant_name, keep_indices in [("full_10feat", list(range(10))),
                                         ("no_weights_8feat", [0,1,2,3,4,5,6,9])]:
        print(f"\n--- {variant_name} ---")
        all_costs = []
        per_seed = {}

        for seed in SEEDS:
            print(f"  Seed {seed}: Training 500K steps...")
            env = MaxSATDACEnv(
                instance_paths=train_paths, use_csolver=True,
                checkpoint_interval=CHECKPOINT_INTERVAL,
                solver_timeout=SOLVER_TIMEOUT, max_steps=MAX_STEPS,
                reward_type="shaped", seed=seed,
            )
            if len(keep_indices) < 10:
                env = StateFeatureMaskWrapper(env, keep_indices)
            env = Monitor(env)

            model = PPO("MlpPolicy", env, learning_rate=LR, n_steps=N_STEPS,
                        batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, seed=seed, verbose=0)
            t0 = time.time()
            model.learn(total_timesteps=500_000)
            train_time = time.time() - t0
            env.close()
            print(f"    Trained in {train_time:.0f}s")

            # Evaluate (must wrap eval env with same mask)
            costs = []
            for path in test_paths:
                eval_env = make_eval_env(path, seed)
                if len(keep_indices) < 10:
                    eval_env = StateFeatureMaskWrapper(eval_env, keep_indices)
                obs, info = eval_env.reset()
                best_cost = info.get("initial_cost", float("inf"))
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = eval_env.step(action)
                    if info.get("best_cost", float("inf")) < best_cost:
                        best_cost = info["best_cost"]
                    done = terminated or truncated
                eval_env.close()
                costs.append(float(best_cost))

            all_costs.extend(costs)
            per_seed[str(seed)] = {"costs": costs, "mean": float(np.mean(costs)), "train_time": train_time}
            print(f"    Test mean: {np.mean(costs):.1f}")

        results[variant_name] = {
            "keep_indices": keep_indices, "n_features": len(keep_indices),
            "overall_mean": float(np.mean(all_costs)),
            "overall_std": float(np.std(all_costs)),
            "all_costs": all_costs, "per_seed": per_seed,
        }
    return results


# ── Experiment 3: Learning Curves ────────────────────────────────

def experiment_learning_curves(splits):
    """Train with periodic evaluation checkpoints."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Learning Curves")
    print("="*70)

    seed = 42
    env = Monitor(MaxSATDACEnv(
        instance_paths=splits["train"], use_csolver=True,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        solver_timeout=SOLVER_TIMEOUT, max_steps=MAX_STEPS,
        reward_type="shaped", seed=seed,
    ))

    callback = LearningCurveCallback(
        eval_paths=splits["test"], eval_seed=seed, eval_interval=25000)

    model = PPO("MlpPolicy", env, learning_rate=LR, n_steps=N_STEPS,
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, seed=seed, verbose=0)

    print(f"Training 500K with eval every 25K steps...")
    t0 = time.time()
    model.learn(total_timesteps=500_000, callback=callback)
    train_time = time.time() - t0
    env.close()

    final_costs = evaluate_ppo(model, splits["test"], seed)
    return {
        "seed": seed, "train_time_s": train_time,
        "checkpoints": callback.checkpoints,
        "final_mean_cost": float(np.mean(final_costs)), "final_costs": final_costs,
    }


# ── Experiment 4: Reward Function Ablation ───────────────────────

def experiment_reward_ablation(splits):
    """Compare shaped reward vs pure cost improvement vs final cost."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Reward Function Ablation")
    print("="*70)

    seed = 42
    timesteps = 100_000
    results = {}

    for rtype in ["shaped", "cost_improvement", "final_cost"]:
        print(f"\n--- reward_type={rtype} ---")
        env = Monitor(MaxSATDACEnv(
            instance_paths=splits["train"], use_csolver=True,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            solver_timeout=SOLVER_TIMEOUT, max_steps=MAX_STEPS,
            reward_type=rtype, seed=seed,
        ))
        model = PPO("MlpPolicy", env, learning_rate=LR, n_steps=N_STEPS,
                    batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, seed=seed, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=timesteps)
        train_time = time.time() - t0
        env.close()

        costs = evaluate_ppo(model, splits["test"], seed)
        print(f"  Test mean: {np.mean(costs):.1f}")
        results[rtype] = {"train_time_s": train_time, "test_costs": costs,
                          "test_mean": float(np.mean(costs)), "test_std": float(np.std(costs))}
    return results


# ── Experiment 5: State-Conditioning Evidence ────────────────────

def experiment_state_conditioning(splits):
    """Show that the policy conditions on state features beyond step_fraction."""
    print("\n" + "="*70)
    print("EXPERIMENT 5: State-Conditioning Evidence")
    print("="*70)

    test_paths = splits["test"]
    model_path = RESULTS_DIR / "ppo_csolver_500k" / "model_seed42"
    model = PPO.load(str(model_path))

    # Collect (state, action) pairs from all test instances
    all_records = []
    for path in test_paths:
        env = make_eval_env(path, 42)
        obs, info = env.reset()
        done, step = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            all_records.append({
                "instance": os.path.basename(path), "step": step,
                "step_fraction": float(obs[0]), "plateau_fraction": float(obs[6]),
                "noise_prob_action": float(action[2]), "h_inc_action": float(action[0]),
            })
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        env.close()

    print(f"Collected {len(all_records)} (state, action) pairs from {len(test_paths)} instances")

    # Analysis: at same step_fraction, does noise_prob differ by plateau_fraction?
    step_bins = {}
    for r in all_records:
        sf_bin = min(round(r["step_fraction"] * 10) / 10, 0.9)
        step_bins.setdefault(sf_bin, []).append(r)

    conditioning_evidence = []
    for sf_bin in sorted(step_bins.keys()):
        records = step_bins[sf_bin]
        if len(records) < 5:
            continue
        median_plateau = np.median([r["plateau_fraction"] for r in records])
        low = [r for r in records if r["plateau_fraction"] <= median_plateau]
        high = [r for r in records if r["plateau_fraction"] > median_plateau]
        if len(low) < 3 or len(high) < 3:
            continue
        noise_low = [r["noise_prob_action"] for r in low]
        noise_high = [r["noise_prob_action"] for r in high]
        try:
            _, p_val = mannwhitneyu(noise_high, noise_low, alternative="two-sided")
            p_val = float(p_val)
        except Exception:
            p_val = 1.0
        conditioning_evidence.append({
            "step_fraction_bin": sf_bin, "n_low": len(low), "n_high": len(high),
            "noise_mean_low": float(np.mean(noise_low)),
            "noise_mean_high": float(np.mean(noise_high)),
            "noise_diff": float(np.mean(noise_high) - np.mean(noise_low)),
            "mann_whitney_p": p_val,
        })

    # Pairwise correlation of per-instance action sequences
    per_inst = {}
    for r in all_records:
        per_inst.setdefault(r["instance"], []).append(r["noise_prob_action"])

    instances = list(per_inst.keys())
    min_len = min(len(per_inst[i]) for i in instances)
    seqs = np.array([per_inst[i][:min_len] for i in instances])
    corrs = []
    for i in range(len(instances)):
        for j in range(i+1, len(instances)):
            c = np.corrcoef(seqs[i], seqs[j])[0, 1]
            if not np.isnan(c):
                corrs.append(float(c))

    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    print(f"Mean pairwise action correlation: {mean_corr:.4f}")

    return {
        "n_records": len(all_records), "n_instances": len(test_paths),
        "conditioning_by_plateau": conditioning_evidence,
        "pairwise_correlation": {"mean": mean_corr, "std": float(np.std(corrs)) if corrs else 0.0,
                                  "n_pairs": len(corrs)},
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    print(f"MaxSAT DAC — Extended Experiments\nDate: {datetime.now().isoformat()}\n")

    splits, _ = load_instance_splits()
    all_results = {"metadata": {"date": datetime.now().isoformat()}}
    total_t0 = time.time()

    # Fast experiments first
    all_results["state_conditioning"] = experiment_state_conditioning(splits)
    all_results["bayesian_optimization"] = experiment_bayesian_optimization(splits)
    all_results["reward_ablation"] = experiment_reward_ablation(splits)
    all_results["learning_curves"] = experiment_learning_curves(splits)
    all_results["matched_ablation"] = experiment_matched_ablation(splits)

    total_time = time.time() - total_t0
    all_results["metadata"]["total_time_s"] = total_time

    with open(RESULTS_DIR / "experiment_bo_ablation_conditioning.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}\nALL EXPERIMENTS COMPLETE — {total_time:.0f}s\n{'='*70}")
    print(f"Saved to {RESULTS_DIR / 'experiment_bo_ablation_conditioning.json'}")


if __name__ == "__main__":
    main()
