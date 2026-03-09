# Learned Control Layers for MaxSAT Local Search

This repository contains the code, data, and solver modifications for the first RL-based Dynamic Algorithm Configuration (DAC) system for MaxSAT local search.

## Key Results

| Comparison | PPO Mean Cost | Baseline Mean Cost | Improvement | Wilcoxon p |
|------------|--------------|-------------------|-------------|-----------|
| PPO vs Random | 372.1 | 459.6 | **−19.0%** | 2.4 × 10⁻⁵ |
| PPO vs Best Static | 372.1 | 415.1 | **−10.4%** | 0.0069 |
| PPO vs BO Static | 372.1 | 321.1 | +15.9% | — |
| PPO vs BO (v500) | — | — | **7/10 wins** | 0.019 |

All results from 3 seeds × 18 test instances = 54 evaluation points. BO static outperforms PPO on small instances, but PPO overtakes at 500-variable scale.

## What This Is

- **DAC (Dynamic Algorithm Configuration)**: Adjust solver parameters *during* solving based on search state, not just before solving
- **Control Layer**: An external PPO controller observes NuWLS search state every 1,000 flips and adjusts 4 clause-weighting parameters
- **Key Finding**: The learned policy discovers an explore-then-exploit noise schedule, with `noise_prob` as the dominant control channel

## Repository Structure

```
learned-control-layers/
├── src/
│   ├── gym_env.py              # Gymnasium DAC environment
│   ├── solver_wrapper.py       # NuWLS-DAC subprocess interface
│   ├── configs.py              # Shared configuration and path utilities
│   ├── evaluation.py           # Shared evaluation and statistical testing
│   ├── train_500k.py           # Main training script (3 seeds × 500K steps)
│   ├── generate_instances.py   # Phase-transition benchmark generator
│   ├── ablation_study.py       # State/action feature ablation
│   ├── scale_experiment.py     # Scale transfer experiments (v200, v500)
│   └── ...                     # Additional experiment scripts
├── data/
│   ├── solvers/
│   │   ├── NuWLS/NuWLS-dac/    # Modified NuWLS with DAC protocol (C++)
│   │   └── USW-LS/USW-LS-dac/  # Modified USW-LS for cross-solver transfer
│   ├── benchmarks/
│   │   ├── generated/          # 100 phase-transition instances (v50/v75/v100)
│   │   ├── generated_hard/     # 100 hard partial MaxSAT instances (pt75)
│   │   └── generated_large/    # Scale transfer instances (v200/v500)
│   └── results/                # All experimental results (JSON)
├── configs/default.json        # Project configuration
├── requirements.txt            # Python dependencies
└── REPRODUCE.md                # Step-by-step reproduction guide
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Compile NuWLS-DAC solver
cd data/solvers/NuWLS/NuWLS-dac
make              # Linux/Mac (g++)
# or on Windows with MSVC:
# cl.exe /O2 /EHsc pms.cpp /Fe:nuwls-dac.exe
cd ../../../..

# 3. Train PPO (3 seeds × 500K steps, ~4 hours)
python src/train_500k.py

# 4. Run smoke test
python src/smoke_test.py
```

See [REPRODUCE.md](REPRODUCE.md) for detailed step-by-step instructions.

## NuWLS-DAC Protocol

The solver modification adds a `--dac N` flag that pauses every N variable flips:

1. **Solver → Controller**: Emits 12-dimensional state vector (cost, improvement rate, plateau length, clause weight statistics, etc.)
2. **Controller → Solver**: Sends 6 parameter values (weight increment, smoothing probability, noise probability, etc.) or `CONTINUE`/`STOP`

This creates a standard Gymnasium interface compatible with any RL algorithm from Stable-Baselines3.

## Five Structural Insights

1. **Exploration parameter dominance**: `noise_prob` alone accounts for nearly all DAC benefit
2. **Scale-dependent DAC advantage**: PPO overtakes optimal static at 5× training scale
3. **Automatic explore-then-exploit discovery**: Policy independently discovers noise annealing
4. **Scale-dependent feature importance**: Clause weight features hurt at 100K but help at 500K
5. **Decision frequency sensitivity**: Finer checkpoints (250 vs 1000 flips) disproportionately benefit learned policies

## Requirements

- Python 3.10+
- g++ (Linux/Mac) or MSVC (Windows) for solver compilation
- 4+ CPU cores recommended
- ~4 hours for full training (500K steps × 3 seeds)

## Paper

*Learned Control Layers for MaxSAT Local Search*
Alex Chengyu Li, 2026

[![Zenodo](https://img.shields.io/badge/Preprint-Zenodo-blue?style=flat-square&logo=zenodo)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Citation

Please cite as: Li, Alex Chengyu (2026). "Learned Control Layers for MaxSAT Local Search: Dynamic Algorithm Configuration of Clause Weighting Parameters." Preprint.

```bibtex
@misc{li2026maxsatdac,
  author    = {Li, Alex Chengyu},
  title     = {Learned Control Layers for {MaxSAT} Local Search:
               Dynamic Algorithm Configuration of Clause Weighting Parameters},
  year      = {2026},
  note      = {Preprint}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- NuWLS solver by Yi Chu et al. (AAAI 2023): [GitHub](https://github.com/filyouzicha/NuWLS)
- USW-LS solver from MaxSAT Evaluation
- DACBench framework: [GitHub](https://github.com/automl/DACBench)
