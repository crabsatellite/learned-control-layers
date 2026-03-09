"""
Wrapper for MaxSAT local search solvers (NuWLS / USW-LS).

Two modes of integration:
  Mode A (subprocess): Run solver as subprocess, parse final output only.
                       Used for baseline evaluation and final benchmarking.
  Mode B (checkpoint): Modified solver reports state periodically, receives
                       parameter updates via stdin/stdout protocol.
                       Used for RL training (DAC).

The checkpoint protocol:
  Solver prints to stdout:  STATE <step> <cost> <hard_unsat> <soft_sat_frac> <flip_rate> <plateau> <w_mean> <w_std>
  Controller writes to stdin: PARAMS <h_inc> <smooth_prob> <noise_prob> <hard_weight_mult>
  Solver reads new params and continues for N more flips.
  At termination: DONE <final_cost> <total_flips>
"""

import subprocess
import os
import json
import time
import signal
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class SolverConfig:
    """Configuration for the MaxSAT solver."""
    solver_binary: str = ""
    timeout_seconds: int = 60
    checkpoint_interval: int = 10000  # flips between checkpoints
    seed: int = 1

    def __post_init__(self):
        if not self.solver_binary:
            # Default: look for compiled solver
            self.solver_binary = str(PROJECT_ROOT / "data" / "solvers" / "nuwls-dac")


@dataclass
class SolverState:
    """State reported by solver at each checkpoint."""
    step: int = 0
    cost: float = float('inf')
    hard_unsat: int = 0
    soft_sat_frac: float = 0.0
    flip_rate: float = 0.0
    plateau_length: int = 0
    weight_mean: float = 0.0
    weight_std: float = 0.0
    done: bool = False
    final_cost: float = float('inf')
    total_flips: int = 0


@dataclass
class SolverParams:
    """Parameters that the RL agent can adjust."""
    h_inc: float = 1.0           # Hard clause weight increment
    smooth_prob: float = 0.01    # Weight smoothing probability
    noise_prob: float = 0.01     # Random walk probability
    hard_weight_mult: float = 1.0  # Hard vs soft weight multiplier


class SubprocessSolver:
    """Run solver as a subprocess for baseline evaluation (Mode A)."""

    def __init__(self, config: SolverConfig):
        self.config = config

    def solve(self, wcnf_path: str) -> dict:
        """Run solver on a single WCNF instance. Returns result dict."""
        cmd = [
            self.config.solver_binary,
            wcnf_path,
            str(self.config.seed),
            str(self.config.timeout_seconds)
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds + 5
            )
            elapsed = time.time() - start_time

            return self._parse_output(result.stdout, result.stderr, elapsed)
        except subprocess.TimeoutExpired:
            return {
                "solved": False,
                "cost": float('inf'),
                "time": self.config.timeout_seconds,
                "timeout": True
            }

    def _parse_output(self, stdout: str, stderr: str, elapsed: float) -> dict:
        """Parse MaxSAT solver output (standard MSE format)."""
        cost = float('inf')
        solved = False

        for line in stdout.strip().split('\n'):
            line = line.strip()
            if line.startswith('o '):
                # Objective value line
                try:
                    cost = int(line[2:])
                    solved = True
                except ValueError:
                    pass
            elif line.startswith('s '):
                # Status line
                if 'OPTIMUM' in line or 'SATISFIABLE' in line:
                    solved = True

        return {
            "solved": solved,
            "cost": cost,
            "time": elapsed,
            "timeout": False
        }


class CheckpointSolver:
    """
    Interactive solver with checkpoint-based parameter control (Mode B).

    The solver is a modified version that:
    1. Runs for checkpoint_interval flips
    2. Reports search state via stdout
    3. Reads new parameters from stdin
    4. Repeats until timeout or convergence
    """

    def __init__(self, config: SolverConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._wcnf_path: Optional[str] = None

    def start(self, wcnf_path: str) -> SolverState:
        """Start solver process and return initial state."""
        self._wcnf_path = wcnf_path

        cmd = [
            self.config.solver_binary,
            "--checkpoint-mode",
            "--checkpoint-interval", str(self.config.checkpoint_interval),
            wcnf_path,
            str(self.config.seed),
            str(self.config.timeout_seconds)
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        # Read initial state
        return self._read_state()

    def step(self, params: SolverParams) -> SolverState:
        """Send parameters and get next state."""
        if self.process is None or self.process.poll() is not None:
            state = SolverState(done=True)
            return state

        # Send parameters
        param_line = f"PARAMS {params.h_inc} {params.smooth_prob} {params.noise_prob} {params.hard_weight_mult}\n"
        try:
            self.process.stdin.write(param_line)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            return SolverState(done=True)

        # Read next state
        return self._read_state()

    def _read_state(self) -> SolverState:
        """Read a state line from solver stdout."""
        try:
            line = self.process.stdout.readline().strip()
        except (BrokenPipeError, OSError):
            return SolverState(done=True)

        if not line:
            return SolverState(done=True)

        if line.startswith("STATE"):
            parts = line.split()
            return SolverState(
                step=int(parts[1]),
                cost=float(parts[2]),
                hard_unsat=int(parts[3]),
                soft_sat_frac=float(parts[4]),
                flip_rate=float(parts[5]),
                plateau_length=int(parts[6]),
                weight_mean=float(parts[7]),
                weight_std=float(parts[8]),
            )
        elif line.startswith("DONE"):
            parts = line.split()
            final_cost = float(parts[1])
            return SolverState(
                done=True,
                cost=final_cost,
                final_cost=final_cost,
                total_flips=int(parts[2])
            )
        else:
            return SolverState(done=True)

    def close(self):
        """Terminate solver process."""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None


class CSolver:
    """
    C solver wrapper using the NuWLS-DAC checkpoint protocol.

    Protocol:
      1. Solver emits: DAC_READY <num_vars> <num_clauses> <num_hard> <num_soft> <total_soft_weight>
      2. Every N flips: DAC_STATE <step> <hard_unsat> <soft_unsat_weight> <opt_unsat_weight>
                        <total_step> <weight_mean> <weight_std> <feasible> <time>
                        <hard_large_count> <soft_large_count> <goodvar_count>
      3. Controller responds: "CONTINUE" | "STOP" | "<h_inc> <s_inc> <sp> <ssp> <rwp> <rdp>"
    """

    _SOLVER_NAME = "nuwls-dac.exe" if os.name == "nt" else "nuwls-dac"
    SOLVER_BINARY = str(PROJECT_ROOT / "data" / "solvers" / "NuWLS" / "NuWLS-dac" / _SOLVER_NAME)

    def __init__(self, checkpoint_interval=10000, timeout=60.0, seed=1):
        self.checkpoint_interval = checkpoint_interval
        self.timeout = timeout
        self.seed = seed
        self.process: Optional[subprocess.Popen] = None
        self._instance_info = {}
        self._best_cost = float('inf')
        self._total_soft_weight = 1
        self._num_sclauses = 0

    def start(self, wcnf_path: str) -> SolverState:
        """Start solver and return initial state."""
        self.close()

        cmd = [
            self.SOLVER_BINARY,
            wcnf_path,
            str(self.seed),
            "--dac", str(self.checkpoint_interval),
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        self._best_cost = float('inf')

        # Read lines until we get DAC_READY and first DAC_STATE
        ready_line = self._read_until_prefix("DAC_READY")
        if ready_line:
            parts = ready_line.split()
            self._instance_info = {
                "num_vars": int(parts[1]),
                "num_clauses": int(parts[2]),
                "num_hard": int(parts[3]),
                "num_soft": int(parts[4]),
                "total_soft_weight": int(parts[5]),
            }
            self._total_soft_weight = max(1, self._instance_info["total_soft_weight"])
            self._num_sclauses = self._instance_info["num_soft"]

        state = self._read_dac_state()
        return state

    def step(self, params: SolverParams) -> SolverState:
        """Send parameters, get next state."""
        if self.process is None or self.process.poll() is not None:
            best = self._best_cost if self._best_cost < float('inf') else 0.0
            return SolverState(done=True, cost=best, final_cost=best)

        # Map 4 gym params to 6 solver params
        h_inc = params.h_inc
        s_inc = max(1.0, params.h_inc / 3)
        smooth_prob = params.smooth_prob
        soft_smooth_prob = params.smooth_prob / 100
        rwprob = params.noise_prob
        rdprob = params.noise_prob / 2

        param_line = f"{h_inc} {s_inc} {smooth_prob} {soft_smooth_prob} {rwprob} {rdprob}\n"
        try:
            self.process.stdin.write(param_line)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            return SolverState(done=True, cost=self._best_cost)

        state = self._read_dac_state()
        return state

    def _read_until_prefix(self, prefix: str, max_lines=50) -> Optional[str]:
        """Read lines until one starts with prefix. Collect 'o' lines along the way."""
        for _ in range(max_lines):
            try:
                line = self.process.stdout.readline()
                if not line:
                    return None
                line = line.strip()
                if line.startswith("o "):
                    parts = line.split()
                    try:
                        cost = int(parts[1])
                        if cost < self._best_cost:
                            self._best_cost = cost
                    except (ValueError, IndexError):
                        pass
                if line.startswith(prefix):
                    return line
            except (BrokenPipeError, OSError):
                return None
        return None

    def _read_dac_state(self) -> SolverState:
        """Read lines until DAC_STATE, collecting 'o' lines."""
        state_line = self._read_until_prefix("DAC_STATE")
        if state_line is None:
            best = self._best_cost if self._best_cost < float('inf') else 0.0
            return SolverState(done=True, cost=best, final_cost=best)

        # DAC_STATE <step> <hard_unsat> <soft_unsat_weight> <opt_unsat_weight>
        #           <total_step> <weight_mean> <weight_std> <feasible> <time>
        #           <hard_large_count> <soft_large_count> <goodvar_count>
        parts = state_line.split()
        try:
            step = int(parts[1])
            hard_unsat = int(parts[2])
            soft_unsat_weight = int(parts[3])
            opt_unsat_weight = int(parts[4])
            total_step = int(parts[5])
            weight_mean = float(parts[6])
            weight_std = float(parts[7])
            feasible = int(parts[8])
            runtime = float(parts[9])
            goodvar_count = int(parts[12]) if len(parts) > 12 else 0
        except (ValueError, IndexError):
            best = self._best_cost if self._best_cost < float('inf') else 0.0
            return SolverState(done=True, cost=best, final_cost=best)

        # Use opt_unsat_weight as cost (best found so far)
        # opt_unsat_weight can be LLONG_MAX (9.2e18) when no feasible solution found yet
        # soft_unsat_weight can also be very large. Clamp both to avoid negative/overflow.
        cost = opt_unsat_weight if 0 <= opt_unsat_weight < 9e18 else soft_unsat_weight
        cost = max(0, cost)  # Never negative
        if cost < self._best_cost:
            self._best_cost = cost

        # soft_sat_frac: 1 - (soft_unsat_weight / total_soft_weight)
        soft_sat_frac = max(0, 1.0 - soft_unsat_weight / self._total_soft_weight) if self._total_soft_weight > 0 else 0

        # Check if done (timeout or solver exited)
        done = runtime >= self.timeout or (self.process.poll() is not None)

        return SolverState(
            step=step,
            cost=float(cost),
            hard_unsat=hard_unsat,
            soft_sat_frac=soft_sat_frac,
            flip_rate=float(total_step) / max(runtime, 0.001),
            plateau_length=0,
            weight_mean=weight_mean,
            weight_std=weight_std,
            done=done,
            final_cost=float(self._best_cost),
            total_flips=total_step,
        )

    def close(self):
        """Terminate solver process."""
        if self.process is not None:
            try:
                self.process.stdin.write("STOP\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.stdout.close()
            except OSError:
                pass
            try:
                self.process.stderr.close()
            except OSError:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self.process.kill()
                    self.process.wait(timeout=2)
                except OSError:
                    pass
            self.process = None


class SimulatedSolver:
    """
    Simulated solver for development/testing when real solver is not available.
    Mimics the checkpoint protocol with a simple local search simulation.
    """

    def __init__(self, config: SolverConfig):
        self.config = config
        self._step = 0
        self._cost = 1000.0
        self._best_cost = 1000.0
        self._plateau = 0
        self._done = False
        self._max_steps = 100
        self._rng = None

    def start(self, wcnf_path: str) -> SolverState:
        """Initialize simulation."""
        import numpy as np
        self._rng = np.random.default_rng(self.config.seed)
        self._step = 0
        self._cost = 1000.0
        self._best_cost = 1000.0
        self._plateau = 0
        self._done = False
        return self._get_state()

    def step(self, params: SolverParams) -> SolverState:
        """Simulate one checkpoint interval."""
        if self._done:
            return SolverState(done=True, final_cost=self._best_cost, total_flips=self._step * self.config.checkpoint_interval)

        self._step += 1

        # Simulate cost improvement based on parameters
        # Better params → more improvement
        base_improvement = 5.0
        param_bonus = (params.h_inc * 0.5 + params.smooth_prob * 100 +
                       params.noise_prob * 50 - params.hard_weight_mult * 0.3)
        noise = self._rng.normal(0, 10)

        improvement = base_improvement + param_bonus + noise
        self._cost = max(0, self._cost - improvement)

        if self._cost < self._best_cost:
            self._best_cost = self._cost
            self._plateau = 0
        else:
            self._plateau += 1

        if self._step >= self._max_steps or self._cost <= 0:
            self._done = True
            return SolverState(
                done=True,
                cost=self._cost,
                final_cost=self._best_cost,
                total_flips=self._step * self.config.checkpoint_interval
            )

        return self._get_state()

    def _get_state(self) -> SolverState:
        return SolverState(
            step=self._step,
            cost=self._cost,
            hard_unsat=max(0, int(self._cost / 100)),
            soft_sat_frac=max(0, 1.0 - self._cost / 1000.0),
            flip_rate=self._rng.uniform(1000, 50000),
            plateau_length=self._plateau,
            weight_mean=1.0 + self._step * 0.1,
            weight_std=0.5 + self._step * 0.05,
        )

    def close(self):
        pass
