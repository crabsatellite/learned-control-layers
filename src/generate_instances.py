#!/usr/bin/env python3
"""
Generate random partial MaxSAT instances at the phase transition.

Format: simplified WCNF
  - Lines starting with 'h' are hard clauses
  - Lines starting with a number are soft clauses with that weight
  - Each clause ends with '0'
  - All clauses are 3-literal (3-SAT style)

Parameters derived from existing instances:
  v50:  315 clauses (215 hard, 100 soft) -> ratio=6.3, hard_frac=0.683
  v75:  472 clauses (322 hard, 150 soft) -> ratio=6.29, hard_frac=0.682
  v100: 600 clauses (400 hard, 200 soft) -> ratio=6.0, hard_frac=0.667
  Average: ~6.2 clauses/var, ~68% hard, ~32% soft
  Soft clause weights: uniform [1, 100]
"""

import argparse
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmarks" / "generated_large"

# Parameters matching existing generated instances
CLAUSE_VAR_RATIO = 6.2    # total clauses per variable
HARD_FRACTION = 0.68      # fraction of clauses that are hard
CLAUSE_LENGTH = 3         # 3-SAT style
WEIGHT_MIN = 1
WEIGHT_MAX = 100


def generate_clause(n_vars: int, rng: random.Random) -> list:
    """Generate a random 3-literal clause over n_vars variables."""
    # Pick 3 distinct variables
    variables = rng.sample(range(1, n_vars + 1), CLAUSE_LENGTH)
    # Randomly negate each
    literals = [v if rng.random() < 0.5 else -v for v in variables]
    return literals


def generate_instance(n_vars: int, seed: int) -> str:
    """Generate a random partial MaxSAT instance in simplified WCNF format."""
    rng = random.Random(seed)

    n_total = int(n_vars * CLAUSE_VAR_RATIO)
    n_hard = int(n_total * HARD_FRACTION)
    n_soft = n_total - n_hard

    lines = []

    # Hard clauses
    for _ in range(n_hard):
        lits = generate_clause(n_vars, rng)
        lines.append("h " + " ".join(str(l) for l in lits) + " 0")

    # Soft clauses with random weights
    for _ in range(n_soft):
        lits = generate_clause(n_vars, rng)
        weight = rng.randint(WEIGHT_MIN, WEIGHT_MAX)
        lines.append(f"{weight} " + " ".join(str(l) for l in lits) + " 0")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate random partial MaxSAT instances")
    parser.add_argument("--n_vars", type=int, nargs="+", default=[200, 500],
                        help="Number of variables per instance size")
    parser.add_argument("--n_instances", type=int, default=30,
                        help="Number of instances per size")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--base_seed", type=int, default=12345,
                        help="Base seed for reproducibility")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for n_vars in args.n_vars:
        print(f"\nGenerating {args.n_instances} instances with {n_vars} variables...")
        n_total = int(n_vars * CLAUSE_VAR_RATIO)
        n_hard = int(n_total * HARD_FRACTION)
        n_soft = n_total - n_hard
        print(f"  {n_total} clauses ({n_hard} hard, {n_soft} soft), 3-literal")

        for i in range(args.n_instances):
            seed = args.base_seed + n_vars * 1000 + i
            content = generate_instance(n_vars, seed)
            filename = f"v{n_vars}_{i:03d}.wcnf"
            filepath = output_dir / filename
            with open(filepath, "w") as f:
                f.write(content)

        print(f"  Saved to {output_dir}/v{n_vars}_*.wcnf")

    # Print split info
    for n_vars in args.n_vars:
        print(f"\nv{n_vars} split (20 train / 5 val / 5 test):")
        print(f"  Train: v{n_vars}_000.wcnf - v{n_vars}_019.wcnf")
        print(f"  Val:   v{n_vars}_020.wcnf - v{n_vars}_024.wcnf")
        print(f"  Test:  v{n_vars}_025.wcnf - v{n_vars}_029.wcnf")


if __name__ == "__main__":
    main()
