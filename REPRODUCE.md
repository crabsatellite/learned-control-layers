# Reproduction Guide

Step-by-step instructions to reproduce all results from the paper.

## Prerequisites

- Python 3.10+
- C++ compiler: g++ (Linux/Mac) or MSVC cl.exe (Windows)
- 4+ CPU cores (i7-12700K used in paper)
- ~4 hours total compute time for main results
- ~2 hours additional for ablation/transfer experiments

```bash
pip install -r requirements.txt
```

## Stage 1: Compile NuWLS-DAC Solver

The modified NuWLS solver with DAC protocol is included in `data/solvers/NuWLS/NuWLS-dac/`.

### Linux/Mac

```bash
cd data/solvers/NuWLS/NuWLS-dac
make
cd ../../../..
```

### Windows (MSVC)

```powershell
# Open Developer Command Prompt or use Enter-VsDevShell
cd data\solvers\NuWLS\NuWLS-dac
cl.exe /O2 /EHsc pms.cpp /Fe:nuwls-dac.exe
cd ..\..\..\..
```

Verify: `./data/solvers/NuWLS/NuWLS-dac/nuwls-dac` (or `.exe` on Windows) should print usage info.

## Stage 2: Verify Benchmarks

The generated benchmark instances are included in the repository:
- `data/benchmarks/generated/` — 100 instances (v50, v75, v100)
- `data/benchmarks/generated_hard/` — 100 instances (pt75, partial MaxSAT)

The train/validation/test split is recorded in `data/results/experiment_train_test_split.json` (84 train / 18 val / 18 test).

```bash
python -c "
import json
with open('data/results/experiment_train_test_split.json') as f:
    d = json.load(f)
print(f'Train: {len(d[\"instances\"][\"train\"])} instances')
print(f'Val:   {len(d[\"instances\"][\"val\"])} instances')
print(f'Test:  {len(d[\"instances\"][\"test\"])} instances')
"
# Expected: Train: 84, Val: 18, Test: 18
```

## Stage 3: Quick Smoke Test (~5 minutes)

```bash
python src/smoke_test.py
```

This verifies the solver binary works, the DAC protocol functions, and the Gymnasium environment can run episodes.

## Stage 4: Main Training (Table 5, ~4 hours)

Train 3 PPO models (seeds 42, 123, 999) for 500K timesteps each:

```bash
python src/train_500k.py
```

This produces:
- Trained models in `data/results/ppo_csolver_500k/`
- Multi-seed evaluation results in `data/results/csolver_500k_multiseed.json`

**Expected results** (Table 5 in paper):
- PPO mean cost: ~372 (±20 across runs)
- Random mean cost: ~460
- Wilcoxon PPO vs Random: p < 0.001
- All 3 seeds individually significant (p < 0.05)

## Stage 5: Ablation Studies (Tables 7–12, ~2 hours)

```bash
# State feature ablation (Table 7a)
python src/ablation_study.py

# Checkpoint sensitivity (Table 9)
python src/checkpoint_sensitivity.py

# Oracle schedules (Table 10)
python src/oracle_schedule.py
```

## Stage 6: Scale Transfer (Table 13, ~1 hour)

Generate large instances and test zero-shot transfer:

```bash
# Generate v200 and v500 instances (if not already present)
python src/generate_instances.py --num-vars 200 --count 30
python src/generate_instances.py --num-vars 500 --count 30

# Run scale experiment
python src/scale_experiment.py
```

**Expected**: PPO trained on v50–v100 transfers to v200 (−24.2%, p=0.028 one-sided) and v500 (−37.4%, p=0.004 one-sided).

## Stage 7: Bayesian Optimization Baseline (Table 14, ~30 minutes)

```bash
python src/experiment_bo_ablation_conditioning.py
```

**Expected**: BO static finds configuration with mean cost ~321 on test set (better than PPO's 372 at this scale).

## Stage 8: Cross-Solver Transfer (Table 16, ~10 minutes)

Compile USW-LS-DAC first:

```bash
cd data/solvers/USW-LS/USW-LS-dac
make    # or cl.exe /O2 /EHsc pms.cpp /Fe:usw-ls-dac.exe
cd ../../../..

python src/cross_solver_transfer.py
```

**Expected**: PPO policy trained on NuWLS does NOT transfer to USW-LS (no significant improvement over random).

## Validation Criteria

Results are valid if:
1. PPO achieves statistically significant improvement over random control (Wilcoxon p < 0.05)
2. All 3 training seeds produce individually significant results
3. Scale transfer shows improvement at v200 and v500
4. Cross-solver transfer fails (policy is solver-specific)

## Hardware Notes

- **CPU**: Paper uses i7-12700K (12 cores). Training time scales roughly linearly with core count.
- **Memory**: ~4 GB RAM sufficient.
- **GPU**: Not required (PPO MLP policy is CPU-only).
- **OS**: Developed on Windows 11. Linux/macOS should work with g++ compilation.
- **Non-determinism**: Exact costs may vary by ±20 across machines due to solver timing sensitivity, but statistical significance should hold.

## Result Files

All experimental evidence is stored as JSON in `data/results/`:

| File | Paper Table |
|------|------------|
| `csolver_500k_multiseed.json` | Table 5, 6 |
| `ablation_study.json` | Table 7a, 7b |
| `checkpoint_sensitivity.json` | Table 9 |
| `oracle_schedules.json` | Table 10 |
| `difficulty_analysis.json` | Table 11 |
| `scale_experiment.json` | Table 13 |
| `experiment_bo_ablation_conditioning.json` | Table 14 |
| `cross_solver_transfer.json` | Table 16 |
| `mse_transfer.json` | Section 5.11 |
| `policy_trajectories.json` | Figure 1 |
