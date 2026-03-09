"""
Difficulty-stratified analysis of MaxSAT DAC 500K results.

Key question: Does DAC (PPO) help more on medium-difficulty instances
where search phases exist, compared to easy (trivial) or hard (timeout) instances?
"""

import json
import numpy as np
from scipy import stats

from configs import RESULTS_DIR

RESULTS_FILE = RESULTS_DIR / "csolver_500k_multiseed.json"
OUTPUT_FILE = RESULTS_DIR / "difficulty_analysis.json"

def main():
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    seeds = data["metadata"]["seeds"]
    n_instances = 18
    methods = ["ppo", "random", "static_default", "static_wpms",
               "static_high_h", "static_low_noise", "static_aggressive"]

    # Collect per-instance costs across all methods and seeds
    # Shape: instance_costs[i] = list of all costs for instance i
    instance_costs_all = {i: [] for i in range(n_instances)}

    for seed_str in [str(s) for s in seeds]:
        seed_data = data["per_seed"][seed_str]
        for method in methods:
            key = f"{method}_costs" if method != "ppo" else "ppo_costs"
            if method == "random":
                key = "random_costs"
            costs = seed_data[key]
            for i in range(n_instances):
                instance_costs_all[i].append(costs[i])

    # Difficulty score = average cost across ALL methods and seeds
    difficulty_scores = {}
    for i in range(n_instances):
        difficulty_scores[i] = np.mean(instance_costs_all[i])

    # Sort by difficulty and split into 3 groups
    sorted_indices = sorted(range(n_instances), key=lambda i: difficulty_scores[i])
    easy_idx = sorted_indices[:6]
    medium_idx = sorted_indices[6:12]
    hard_idx = sorted_indices[12:]

    groups = {
        "easy": easy_idx,
        "medium": medium_idx,
        "hard": hard_idx,
    }

    # For each instance, collect PPO/Random/BestStatic costs across 3 seeds
    # best_static = per-instance minimum across all static methods
    ppo_by_instance = {}  # i -> [cost_seed1, cost_seed2, cost_seed3]
    random_by_instance = {}
    best_static_by_instance = {}

    static_methods = ["static_default", "static_wpms", "static_high_h",
                      "static_low_noise", "static_aggressive"]

    for i in range(n_instances):
        ppo_by_instance[i] = []
        random_by_instance[i] = []
        best_static_by_instance[i] = []

    for seed_str in [str(s) for s in seeds]:
        seed_data = data["per_seed"][seed_str]
        ppo_costs = seed_data["ppo_costs"]
        random_costs = seed_data["random_costs"]

        static_all = []
        for sm in static_methods:
            static_all.append(seed_data[f"{sm}_costs"])

        for i in range(n_instances):
            ppo_by_instance[i].append(ppo_costs[i])
            random_by_instance[i].append(random_costs[i])
            # Best static for this instance and seed = min across static methods
            best_static_by_instance[i].append(
                min(static_all[m][i] for m in range(len(static_methods)))
            )

    # Analyze each group
    results = {"groups": {}, "instance_details": {}}

    for group_name, indices in groups.items():
        # Collect all PPO/Random/BestStatic values for this group (instances x seeds)
        ppo_vals = []
        random_vals = []
        best_static_vals = []

        for i in indices:
            ppo_vals.extend(ppo_by_instance[i])
            random_vals.extend(random_by_instance[i])
            best_static_vals.extend(best_static_by_instance[i])

        ppo_arr = np.array(ppo_vals)
        random_arr = np.array(random_vals)
        bs_arr = np.array(best_static_vals)

        ppo_mean = float(np.mean(ppo_arr))
        random_mean = float(np.mean(random_arr))
        bs_mean = float(np.mean(bs_arr))

        improvement_vs_random = float((random_mean - ppo_mean) / random_mean * 100)
        improvement_vs_best_static = float((bs_mean - ppo_mean) / bs_mean * 100)

        # Wilcoxon signed-rank (paired: same instance-seed pairs)
        diff_random = random_arr - ppo_arr
        diff_bs = bs_arr - ppo_arr

        # Only run Wilcoxon if there are non-zero differences
        n_nonzero_random = np.count_nonzero(diff_random)
        n_nonzero_bs = np.count_nonzero(diff_bs)

        if n_nonzero_random >= 2:
            wilcoxon_random = stats.wilcoxon(ppo_arr, random_arr, alternative='less')
            w_random_stat = float(wilcoxon_random.statistic)
            w_random_p = float(wilcoxon_random.pvalue)
        else:
            w_random_stat = None
            w_random_p = None

        if n_nonzero_bs >= 2:
            wilcoxon_bs = stats.wilcoxon(ppo_arr, bs_arr, alternative='less')
            w_bs_stat = float(wilcoxon_bs.statistic)
            w_bs_p = float(wilcoxon_bs.pvalue)
        else:
            w_bs_stat = None
            w_bs_p = None

        # Effect size (Cohen's d)
        pooled_std_random = float(np.sqrt((np.std(ppo_arr, ddof=1)**2 + np.std(random_arr, ddof=1)**2) / 2))
        cohens_d_random = float((random_mean - ppo_mean) / pooled_std_random) if pooled_std_random > 0 else 0.0

        pooled_std_bs = float(np.sqrt((np.std(ppo_arr, ddof=1)**2 + np.std(bs_arr, ddof=1)**2) / 2))
        cohens_d_bs = float((bs_mean - ppo_mean) / pooled_std_bs) if pooled_std_bs > 0 else 0.0

        # Instance-level: how many instances does PPO win on average?
        ppo_wins_vs_random = 0
        ppo_wins_vs_bs = 0
        ties_random = 0
        ties_bs = 0
        for i in indices:
            ppo_mean_i = np.mean(ppo_by_instance[i])
            rand_mean_i = np.mean(random_by_instance[i])
            bs_mean_i = np.mean(best_static_by_instance[i])
            if ppo_mean_i < rand_mean_i:
                ppo_wins_vs_random += 1
            elif ppo_mean_i == rand_mean_i:
                ties_random += 1
            if ppo_mean_i < bs_mean_i:
                ppo_wins_vs_bs += 1
            elif ppo_mean_i == bs_mean_i:
                ties_bs += 1

        difficulty_range = [float(difficulty_scores[i]) for i in indices]

        results["groups"][group_name] = {
            "instance_indices": indices,
            "difficulty_score_range": [float(min(difficulty_range)), float(max(difficulty_range))],
            "difficulty_score_mean": float(np.mean(difficulty_range)),
            "n_observations": len(ppo_vals),
            "ppo_mean": round(ppo_mean, 1),
            "random_mean": round(random_mean, 1),
            "best_static_mean": round(bs_mean, 1),
            "improvement_vs_random_pct": round(improvement_vs_random, 1),
            "improvement_vs_best_static_pct": round(improvement_vs_best_static, 1),
            "cohens_d_vs_random": round(cohens_d_random, 3),
            "cohens_d_vs_best_static": round(cohens_d_bs, 3),
            "wilcoxon_ppo_vs_random": {
                "statistic": w_random_stat,
                "p_value": w_random_p,
                "significant_005": w_random_p < 0.05 if w_random_p is not None else None,
                "n_nonzero_diffs": int(n_nonzero_random),
            },
            "wilcoxon_ppo_vs_best_static": {
                "statistic": w_bs_stat,
                "p_value": w_bs_p,
                "significant_005": w_bs_p < 0.05 if w_bs_p is not None else None,
                "n_nonzero_diffs": int(n_nonzero_bs),
            },
            "ppo_wins_vs_random": f"{ppo_wins_vs_random}/{6 - ties_random} (ties: {ties_random})",
            "ppo_wins_vs_best_static": f"{ppo_wins_vs_bs}/{6 - ties_bs} (ties: {ties_bs})",
        }

    # Per-instance details
    for i in sorted_indices:
        group = "easy" if i in easy_idx else ("medium" if i in medium_idx else "hard")
        results["instance_details"][f"instance_{i}"] = {
            "group": group,
            "difficulty_score": round(float(difficulty_scores[i]), 1),
            "ppo_costs": ppo_by_instance[i],
            "ppo_mean": round(float(np.mean(ppo_by_instance[i])), 1),
            "random_costs": random_by_instance[i],
            "random_mean": round(float(np.mean(random_by_instance[i])), 1),
            "best_static_costs": best_static_by_instance[i],
            "best_static_mean": round(float(np.mean(best_static_by_instance[i])), 1),
        }

    # Print summary
    print("=" * 70)
    print("DIFFICULTY-STRATIFIED ANALYSIS OF MAXSAT DAC 500K RESULTS")
    print("=" * 70)
    print()

    for group_name in ["easy", "medium", "hard"]:
        g = results["groups"][group_name]
        print(f"--- {group_name.upper()} (difficulty {g['difficulty_score_range'][0]:.0f}-{g['difficulty_score_range'][1]:.0f}) ---")
        print(f"  PPO mean:         {g['ppo_mean']}")
        print(f"  Random mean:      {g['random_mean']}")
        print(f"  Best static mean: {g['best_static_mean']}")
        print(f"  PPO vs Random:    {g['improvement_vs_random_pct']:+.1f}% (d={g['cohens_d_vs_random']:.3f})")
        print(f"  PPO vs BestStat:  {g['improvement_vs_best_static_pct']:+.1f}% (d={g['cohens_d_vs_best_static']:.3f})")
        w_r = g["wilcoxon_ppo_vs_random"]
        w_b = g["wilcoxon_ppo_vs_best_static"]
        p_r = f"p={w_r['p_value']:.4f}" if w_r['p_value'] is not None else "N/A"
        p_b = f"p={w_b['p_value']:.4f}" if w_b['p_value'] is not None else "N/A"
        sig_r = " *" if w_r.get("significant_005") else ""
        sig_b = " *" if w_b.get("significant_005") else ""
        print(f"  Wilcoxon vs Rand: {p_r}{sig_r}")
        print(f"  Wilcoxon vs BSt:  {p_b}{sig_b}")
        print(f"  PPO wins vs Rand: {g['ppo_wins_vs_random']}")
        print(f"  PPO wins vs BSt:  {g['ppo_wins_vs_best_static']}")
        print()

    # Key finding
    improvements = {g: results["groups"][g]["improvement_vs_random_pct"] for g in ["easy", "medium", "hard"]}
    best_group = max(improvements, key=improvements.get)
    print(f"KEY FINDING: DAC helps most on {best_group.upper()} instances "
          f"({improvements[best_group]:+.1f}% vs random)")
    print()

    # Instance-level table
    print("Per-instance breakdown (sorted by difficulty):")
    print(f"{'Idx':>4} {'Group':>6} {'Diff':>6} {'PPO':>7} {'Rand':>7} {'BStat':>7} {'vs Rand':>8} {'vs BSt':>8}")
    print("-" * 62)
    for i in sorted_indices:
        det = results["instance_details"][f"instance_{i}"]
        vs_rand = (det["random_mean"] - det["ppo_mean"]) / det["random_mean"] * 100 if det["random_mean"] > 0 else 0
        vs_bs = (det["best_static_mean"] - det["ppo_mean"]) / det["best_static_mean"] * 100 if det["best_static_mean"] > 0 else 0
        print(f"{i:>4} {det['group']:>6} {det['difficulty_score']:>6.0f} "
              f"{det['ppo_mean']:>7.0f} {det['random_mean']:>7.0f} {det['best_static_mean']:>7.0f} "
              f"{vs_rand:>+7.1f}% {vs_bs:>+7.1f}%")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
