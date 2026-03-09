"""Shared evaluation and statistical testing utilities for all experiments."""

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, ttest_ind

from gym_env import MaxSATDACEnv
from configs import CHECKPOINT_INTERVAL, SOLVER_TIMEOUT, MAX_STEPS, PARAM_RANGES


# ── Environment creation ──────────────────────────────────────────

def make_eval_env(instance_path, seed, checkpoint_interval=None,
                  solver_timeout=None, max_steps=None, reward_type="shaped"):
    """Create a single-instance MaxSATDACEnv for evaluation."""
    return MaxSATDACEnv(
        instance_paths=[instance_path],
        use_csolver=True,
        checkpoint_interval=checkpoint_interval or CHECKPOINT_INTERVAL,
        solver_timeout=solver_timeout or SOLVER_TIMEOUT,
        max_steps=max_steps or MAX_STEPS,
        reward_type=reward_type,
        seed=seed,
    )


# ── Parameter encoding ────────────────────────────────────────────

def params_to_action(h_inc, smooth_prob, noise_prob, hard_weight_mult):
    """Convert real-valued solver parameters to action array in [-1, 1]^4."""
    ranges = PARAM_RANGES
    return np.array([
        2.0 * (h_inc - ranges["h_inc"][0]) / (ranges["h_inc"][1] - ranges["h_inc"][0]) - 1.0,
        2.0 * (smooth_prob - ranges["smooth_prob"][0]) / (ranges["smooth_prob"][1] - ranges["smooth_prob"][0]) - 1.0,
        2.0 * (noise_prob - ranges["noise_prob"][0]) / (ranges["noise_prob"][1] - ranges["noise_prob"][0]) - 1.0,
        2.0 * (hard_weight_mult - ranges["hard_weight_mult"][0]) / (ranges["hard_weight_mult"][1] - ranges["hard_weight_mult"][0]) - 1.0,
    ], dtype=np.float32)


def solver_params_to_action(params):
    """Convert a SolverParams instance to action array in [-1, 1]^4."""
    return params_to_action(
        params.h_inc, params.smooth_prob,
        params.noise_prob, params.hard_weight_mult,
    )


# ── Evaluation runners ────────────────────────────────────────────

def _run_episode(env):
    """Run one episode, return best cost."""
    obs, info = env.reset()
    best_cost = info.get("initial_cost", float("inf"))
    done = False
    while not done:
        yield obs, info  # caller sends action via .send()
        action = yield   # receive action
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("best_cost", float("inf")) < best_cost:
            best_cost = info["best_cost"]
        done = terminated or truncated
    env.close()
    return best_cost


def evaluate_ppo(model, test_paths, seed, **env_kwargs):
    """Evaluate PPO model on test instances. Returns list of best costs."""
    costs = []
    for path in test_paths:
        env = make_eval_env(path, seed, **env_kwargs)
        obs, info = env.reset()
        best_cost = info.get("initial_cost", float("inf"))
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("best_cost", float("inf")) < best_cost:
                best_cost = info["best_cost"]
            done = terminated or truncated
        env.close()
        costs.append(float(best_cost))
    return costs


def evaluate_static(params, test_paths, seed, **env_kwargs):
    """Evaluate a fixed SolverParams on test instances. Returns list of best costs."""
    action = solver_params_to_action(params)
    costs = []
    for path in test_paths:
        env = make_eval_env(path, seed, **env_kwargs)
        obs, info = env.reset()
        best_cost = info.get("initial_cost", float("inf"))
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("best_cost", float("inf")) < best_cost:
                best_cost = info["best_cost"]
            done = terminated or truncated
        env.close()
        costs.append(float(best_cost))
    return costs


def evaluate_random(test_paths, seed, **env_kwargs):
    """Evaluate random actions on test instances. Returns list of best costs."""
    rng = np.random.default_rng(seed)
    costs = []
    for path in test_paths:
        env = make_eval_env(path, seed, **env_kwargs)
        obs, info = env.reset()
        best_cost = info.get("initial_cost", float("inf"))
        done = False
        while not done:
            action = rng.uniform(-1, 1, size=4).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("best_cost", float("inf")) < best_cost:
                best_cost = info["best_cost"]
            done = terminated or truncated
        env.close()
        costs.append(float(best_cost))
    return costs


def evaluate_schedule(schedule_fn, test_paths, seed, **env_kwargs):
    """Evaluate a step-fraction-based schedule. schedule_fn(step_frac) -> action.
    Returns list of best costs."""
    max_steps = env_kwargs.get("max_steps", MAX_STEPS)
    costs = []
    for path in test_paths:
        env = make_eval_env(path, seed, **env_kwargs)
        obs, info = env.reset()
        best_cost = info.get("initial_cost", float("inf"))
        done = False
        step = 0
        while not done:
            step_frac = step / max_steps
            action = schedule_fn(step_frac)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("best_cost", float("inf")) < best_cost:
                best_cost = info["best_cost"]
            done = terminated or truncated
            step += 1
        env.close()
        costs.append(float(best_cost))
    return costs


# ── Statistical tests ─────────────────────────────────────────────

def run_statistical_tests(costs_a, costs_b):
    """Run Mann-Whitney U, Wilcoxon signed-rank, and Welch's t-test.
    Tests whether costs_a < costs_b (one-sided 'less').
    Returns dict with test statistics and p-values."""
    result = {
        "mean_a": float(np.mean(costs_a)),
        "mean_b": float(np.mean(costs_b)),
        "diff": float(np.mean(costs_b) - np.mean(costs_a)),
        "diff_pct": float(
            (np.mean(costs_b) - np.mean(costs_a))
            / max(np.mean(costs_b), 1) * 100
        ),
    }
    try:
        u_stat, u_p = mannwhitneyu(costs_a, costs_b, alternative="less")
        result["mann_whitney_u"] = float(u_stat)
        result["mann_whitney_p"] = float(u_p)
    except Exception as e:
        result["mann_whitney_error"] = str(e)

    try:
        w_stat, w_p = wilcoxon(costs_a, costs_b, alternative="less")
        result["wilcoxon_stat"] = float(w_stat)
        result["wilcoxon_p"] = float(w_p)
    except Exception as e:
        result["wilcoxon_error"] = str(e)

    try:
        t_stat, t_p = ttest_ind(costs_a, costs_b)
        result["ttest_stat"] = float(t_stat)
        result["ttest_p"] = float(t_p)
    except Exception as e:
        result["ttest_error"] = str(e)

    return result


# ── PPO training helper ───────────────────────────────────────────

class RewardLogger:
    """SB3 callback that logs episode rewards periodically."""
    def __init__(self):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self_cb, verbose=0):
                super().__init__(verbose)
                self_cb.episode_rewards = []
                self_cb.episode_count = 0

            def _on_step(self_cb):
                for info in self_cb.locals.get("infos", []):
                    if "episode" in info:
                        self_cb.episode_rewards.append(info["episode"]["r"])
                        self_cb.episode_count += 1
                        if self_cb.episode_count % 200 == 0:
                            recent = self_cb.episode_rewards[-200:]
                            print(f"  Episode {self_cb.episode_count}: "
                                  f"mean_reward={np.mean(recent):.4f} "
                                  f"(last 200), step={self_cb.num_timesteps}")
                return True

        self._cb = _Callback()

    @property
    def callback(self):
        return self._cb

    @property
    def episode_rewards(self):
        return self._cb.episode_rewards

    @property
    def episode_count(self):
        return self._cb.episode_count
