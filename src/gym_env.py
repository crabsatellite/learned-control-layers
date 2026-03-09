"""
Gymnasium environment for Dynamic Algorithm Configuration of MaxSAT local search.

This is the core contribution: a Gymnasium-compatible environment that allows
an RL agent to dynamically adjust MaxSAT solver parameters during solving.

Compatible with:
  - Stable-Baselines3 (PPO, A2C, SAC)
  - DACBench (via AbstractBenchmark wrapper)
  - Any Gymnasium-compatible RL framework
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
from typing import Optional, List

from solver_wrapper import (
    SolverConfig, SolverParams, SolverState,
    CheckpointSolver, SimulatedSolver, CSolver
)
try:
    from maxsat_solver import NativeSolver
except ImportError:
    NativeSolver = None  # Optional: only needed if use_native=True


class MaxSATDACEnv(gym.Env):
    """
    MaxSAT Dynamic Algorithm Configuration Environment.

    Observation: Search state features (normalized to [0, 1])
    Action: Parameter adjustments for clause weighting
    Reward: Improvement in objective value

    Episode = one solver run on one MaxSAT instance.
    """

    metadata = {"render_modes": ["human"]}

    # Feature names for interpretability
    FEATURE_NAMES = [
        "step_fraction",       # Current step / max steps
        "cost_normalized",     # Current cost / initial cost
        "cost_improvement",    # (prev_cost - cost) / initial cost
        "hard_unsat_frac",     # Hard clause violations / total hard clauses
        "soft_sat_frac",       # Soft clause satisfaction ratio
        "flip_rate_norm",      # Normalized flip rate
        "plateau_fraction",    # Plateau length / max plateau
        "weight_mean_norm",    # Normalized mean weight
        "weight_std_norm",     # Normalized weight std
        "best_cost_norm",      # Best cost found / initial cost
    ]

    def __init__(
        self,
        instance_paths: List[str],
        solver_config: Optional[SolverConfig] = None,
        max_steps: int = 100,
        use_simulated: bool = True,
        use_native: bool = False,
        use_csolver: bool = False,
        reward_type: str = "cost_improvement",
        normalize_obs: bool = True,
        seed: int = 42,
        checkpoint_interval: int = 1000,
        solver_timeout: float = 10.0,
    ):
        super().__init__()

        self.instance_paths = instance_paths
        self.solver_config = solver_config or SolverConfig()
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.normalize_obs = normalize_obs
        self._seed = seed

        # Create solver (priority: csolver > native > simulated/checkpoint)
        if use_csolver:
            self.solver = CSolver(
                checkpoint_interval=checkpoint_interval,
                timeout=solver_timeout, seed=seed
            )
        elif use_native:
            self.solver = NativeSolver(
                checkpoint_interval=checkpoint_interval,
                timeout=solver_timeout, seed=seed
            )
        elif use_simulated:
            self.solver = SimulatedSolver(self.solver_config)
        else:
            self.solver = CheckpointSolver(self.solver_config)

        # State dimension
        self.n_features = len(self.FEATURE_NAMES)

        # Observation space: normalized features in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_features,),
            dtype=np.float32
        )

        # Action space: 4 continuous parameters
        # Each mapped to a meaningful range via _decode_action
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Episode state
        self._current_instance_idx = 0
        self._step_count = 0
        self._initial_cost = 1.0
        self._prev_cost = 1.0
        self._best_cost = float('inf')
        self._state: Optional[SolverState] = None
        self._rng = np.random.default_rng(seed)
        self._episode_costs = []

    def reset(self, seed=None, options=None):
        """Reset environment: pick a new instance and start solver."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Pick next instance (round-robin with shuffle)
        if self._current_instance_idx >= len(self.instance_paths):
            self._current_instance_idx = 0
            self._rng.shuffle(self.instance_paths)

        instance_path = self.instance_paths[self._current_instance_idx]
        self._current_instance_idx += 1

        # Start solver
        self.solver.close() if hasattr(self.solver, 'close') else None
        self._state = self.solver.start(instance_path)

        # Initialize episode tracking
        self._step_count = 0
        self._initial_cost = max(self._state.cost, 1.0)
        self._prev_cost = self._state.cost
        self._best_cost = self._state.cost
        self._episode_costs = [self._state.cost]

        obs = self._extract_features(self._state)
        info = {"instance": instance_path, "initial_cost": self._initial_cost}

        return obs, info

    def step(self, action):
        """Execute one step: send params to solver, get new state."""
        self._step_count += 1

        # Decode action to solver parameters
        params = self._decode_action(action)

        # Step solver
        self._state = self.solver.step(params)

        # Track costs
        self._episode_costs.append(self._state.cost)

        # Use best_cost as effective cost for done states to avoid inf
        effective_cost = self._state.cost if self._state.cost < float('inf') else self._best_cost
        self._state.cost = effective_cost

        # Compute reward BEFORE updating best_cost so best_bonus works
        is_new_best = self._state.cost < self._best_cost
        reward = self._compute_reward(self._state, is_new_best=is_new_best)

        if self._state.cost < self._best_cost:
            self._best_cost = self._state.cost
        self._prev_cost = self._state.cost

        # Check termination
        terminated = self._state.done
        truncated = self._step_count >= self.max_steps

        if terminated or truncated:
            self.solver.close() if hasattr(self.solver, 'close') else None

        obs = self._extract_features(self._state)
        info = {
            "cost": self._state.cost,
            "best_cost": self._best_cost,
            "step": self._step_count,
            "params": {
                "h_inc": params.h_inc,
                "smooth_prob": params.smooth_prob,
                "noise_prob": params.noise_prob,
                "hard_weight_mult": params.hard_weight_mult,
            }
        }

        return obs, reward, terminated, truncated, info

    def _decode_action(self, action: np.ndarray) -> SolverParams:
        """
        Map continuous action [-1, 1]^4 to solver parameter ranges.

        Ranges chosen based on NuWLS paper and MaxSAT solver literature:
          h_inc:            [0.1, 10.0]   (hard clause weight increment)
          smooth_prob:      [0.0, 0.1]    (weight smoothing probability)
          noise_prob:       [0.0, 0.2]    (random walk probability)
          hard_weight_mult: [0.5, 5.0]    (hard/soft weight ratio)
        """
        action = np.clip(action, -1.0, 1.0)

        # Linear mapping from [-1, 1] to target range
        h_inc = self._linear_map(action[0], 0.1, 10.0)
        smooth_prob = self._linear_map(action[1], 0.0, 0.1)
        noise_prob = self._linear_map(action[2], 0.0, 0.2)
        hard_weight_mult = self._linear_map(action[3], 0.5, 5.0)

        return SolverParams(
            h_inc=h_inc,
            smooth_prob=smooth_prob,
            noise_prob=noise_prob,
            hard_weight_mult=hard_weight_mult,
        )

    @staticmethod
    def _linear_map(x: float, low: float, high: float) -> float:
        """Map x from [-1, 1] to [low, high]."""
        return low + (x + 1.0) * 0.5 * (high - low)

    def _extract_features(self, state: SolverState) -> np.ndarray:
        """Extract normalized feature vector from solver state."""
        import math
        features = np.zeros(self.n_features, dtype=np.float32)

        # Use log-scale for costs to handle huge MaxSAT weights
        init_log = math.log1p(max(0, self._initial_cost))

        features[0] = self._step_count / max(self.max_steps, 1)
        features[1] = math.log1p(max(0, state.cost)) / max(init_log, 1.0)
        features[2] = max(0, (math.log1p(max(0, self._prev_cost)) - math.log1p(max(0, state.cost))) / max(init_log, 1.0))
        features[3] = min(state.hard_unsat / 100.0, 1.0)
        features[4] = state.soft_sat_frac
        features[5] = min(state.flip_rate / 100000.0, 1.0)
        features[6] = min(state.plateau_length / max(self.max_steps, 1), 1.0)
        features[7] = min(state.weight_mean / 100.0, 1.0)
        features[8] = min(state.weight_std / 50.0, 1.0)
        features[9] = math.log1p(max(0, self._best_cost)) / max(init_log, 1.0)

        return np.clip(features, 0.0, 1.0)

    def _compute_reward(self, state: SolverState, is_new_best: bool = False) -> float:
        """Compute reward signal."""
        import math

        if self.reward_type == "cost_improvement":
            # Log-scale cost improvement to handle huge MaxSAT weights
            prev_log = math.log1p(max(0, self._prev_cost))
            curr_log = math.log1p(max(0, state.cost))
            init_log = math.log1p(max(0, self._initial_cost))
            improvement = (prev_log - curr_log) / max(init_log, 1.0)
            return float(np.clip(improvement, -1.0, 1.0))

        elif self.reward_type == "final_cost":
            # Sparse reward: only at episode end
            if state.done or self._step_count >= self.max_steps:
                best_log = math.log1p(max(0, self._best_cost))
                init_log = math.log1p(max(0, self._initial_cost))
                return float(-best_log / max(init_log, 1.0))
            return 0.0

        elif self.reward_type == "shaped":
            # Shaped: log improvement + feasibility bonus + best-improvement bonus
            prev_log = math.log1p(max(0, self._prev_cost))
            curr_log = math.log1p(max(0, state.cost))
            init_log = math.log1p(max(0, self._initial_cost))
            improvement = (prev_log - curr_log) / max(init_log, 1.0)
            # Feasibility bonus (smaller to avoid dominating signal)
            feasibility_bonus = 0.02 if state.hard_unsat == 0 else -0.01
            # New-best bonus: extra reward when we find a new best cost
            best_bonus = 0.05 if is_new_best else 0.0
            return float(np.clip(improvement + feasibility_bonus + best_bonus, -1.0, 1.0))

        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def render(self):
        """Print current state."""
        if self._state is not None:
            print(f"Step {self._step_count}/{self.max_steps} | "
                  f"Cost: {self._state.cost:.1f} | "
                  f"Best: {self._best_cost:.1f} | "
                  f"Hard unsat: {self._state.hard_unsat} | "
                  f"Soft sat: {self._state.soft_sat_frac:.3f}")

    def close(self):
        """Clean up solver process."""
        if hasattr(self.solver, 'close'):
            self.solver.close()


def make_maxsat_env(
    benchmark_dir: str,
    split: str = "train",
    use_simulated: bool = True,
    **kwargs
) -> MaxSATDACEnv:
    """Factory function to create MaxSAT DAC environment."""
    bench_path = Path(benchmark_dir)

    # Load instance paths
    split_file = bench_path / "split.json"
    if split_file.exists():
        import json
        with open(split_file) as f:
            splits = json.load(f)
        instance_paths = [str(bench_path / p) for p in splits[split]]
    else:
        # Use all WCNF files
        instance_paths = [str(p) for p in bench_path.glob("**/*.wcnf")]

    if not instance_paths:
        # Fallback: create dummy instances for testing
        instance_paths = ["dummy.wcnf"]

    return MaxSATDACEnv(
        instance_paths=instance_paths,
        use_simulated=use_simulated,
        **kwargs
    )
