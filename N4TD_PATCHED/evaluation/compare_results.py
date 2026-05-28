"""
evaluation/compare_results.py
------------------------------
Đọc file JSON kết quả từ training/train_mas.py và tạo:
  1. Bảng so sánh Fixed / Independent RL / Cooperative MAS
  2. Biểu đồ học tập (learning curve)
  3. Xuất CSV và hình PNG

Chạy:
    python evaluation/compare_results.py --all
    python evaluation/compare_results.py --scenario peak_hour
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "mas_results"
EVAL_DIR   = ROOT / "output" / "evaluation"

MODES     = ["fixed", "independent", "mas"]
SCENARIOS = ["peak_hour", "low_demand", "medium_demand", "asymmetric", "mixed_vehicle"]

METRICS = {
    "avg_waiting":  "Avg Waiting Time (s)",
    "avg_speed":    "Avg Speed (m/s)",
    "throughput":   "Throughput (vehicles)",
    "reward_gl1":   "Total Reward GL1",
    "reward_gl2":   "Total Reward GL2",
}

MODE_LABELS = {
    "fixed":       "Fixed-Time",
    "independent": "Independent RL",
    "mas":         "Cooperative MAS",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_results(mode: str, scenario: str) -> Optional[List[List[Dict]]]:
    """Load JSON kết quả. Returns None nếu file chưa tồn tại."""
    path = OUTPUT_DIR / f"{mode}_{scenario}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_last_n(results: List[List[Dict]], metric: str, n: int = 20) -> np.ndarray:
    """Lấy giá trị metric từ n episode cuối của mỗi run, flatten thành 1D array."""
    vals = []
    for run in results:
        last = run[-n:] if len(run) >= n else run
        vals.extend([ep.get(metric, 0.0) or 0.0 for ep in last])
    return np.array(vals, dtype=np.float64)


def extract_curve(results: List[List[Dict]], metric: str,
                  window: int = 10) -> np.ndarray:
    """Tính learning curve (mean qua các run, smoothed)."""
    min_len = min(len(run) for run in results)
    arr = np.array(
        [[ep.get(metric, 0.0) or 0.0 for ep in run[:min_len]]
         for run in results]
    )  # shape: (n_runs, n_episodes)
    mean_curve = arr.mean(axis=0)

    # Moving average
    kernel = np.ones(window) / window
    if len(mean_curve) >= window:
        mean_curve = np.convolve(mean_curve, kernel, mode="valid")
    return mean_curve


# ──────────────────────────────────────────────────────────────────────────────
# Table builder
# ──────────────────────────────────────────────────────────────────────────────

def build_comparison_table(scenario: str) -> Dict:
    """
    Tạo dict kết quả so sánh cho 1 scenario.

    Returns:
        {
          metric_key: {
            mode: {"mean": float, "std": float}
          }
        }
    """
    table = {}
    for metric in METRICS:
        table[metric] = {}
        for mode in MODES:
            results = load_results(mode, scenario)
            if results is None:
                table[metric][mode] = {"mean": None, "std": None}
                continue
            vals = extract_last_n(results, metric, n=20)
            table[metric][mode] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
            }
    return table


def print_table(scenario: str, table: Dict):
    """In bảng so sánh ra terminal."""
    col_w = 22
    header = f"\n{'─'*80}\n  Scenario: {scenario.upper()}\n{'─'*80}"
    print(header)

    # Header row
    mode_header = "".join(f"{MODE_LABELS[m]:<{col_w}}" for m in MODES)
    print(f"  {'Metric':<28}{mode_header}")
    print(f"  {'─'*28}{'─'*col_w*len(MODES)}")

    for metric, label in METRICS.items():
        row = f"  {label:<28}"
        for mode in MODES:
            v = table[metric][mode]
            if v["mean"] is None:
                row += f"{'N/A':<{col_w}}"
            else:
                cell = f"{v['mean']:.2f} ±{v['std']:.2f}"
                row += f"{cell:<{col_w}}"
        print(row)

    print(f"{'─'*80}")


def export_csv(scenario: str, table: Dict):
    """Xuất bảng so sánh ra CSV."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"comparison_{scenario}.csv"

    lines = ["metric," + ",".join(
        f"{MODE_LABELS[m]}_mean,{MODE_LABELS[m]}_std" for m in MODES
    )]
    for metric, label in METRICS.items():
        row = [label]
        for mode in MODES:
            v = table[metric][mode]
            if v["mean"] is None:
                row.extend(["N/A", "N/A"])
            else:
                row.extend([f"{v['mean']:.4f}", f"{v['std']:.4f}"])
        lines.append(",".join(row))

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  → CSV saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Learning curves (matplotlib, optional)
# ──────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(scenario: str):
    """Vẽ learning curve reward_gl1 cho 3 mode."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [INFO] matplotlib không có — bỏ qua plot.")
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Learning Curves — {scenario}", fontsize=13)

    colors = {"fixed": "gray", "independent": "steelblue", "mas": "darkorange"}

    for ax, metric in zip(axes, ["reward_gl1", "avg_waiting"]):
        for mode in MODES:
            results = load_results(mode, scenario)
            if results is None:
                continue
            curve = extract_curve(results, metric, window=10)
            x = np.arange(len(curve))
            ax.plot(x, curve, label=MODE_LABELS[mode],
                    color=colors[mode], linewidth=2)

        ax.set_xlabel("Episode")
        ax.set_ylabel(METRICS.get(metric, metric))
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = EVAL_DIR / f"curves_{scenario}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → Plot saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary across all scenarios
# ──────────────────────────────────────────────────────────────────────────────

def print_overall_summary():
    """In bảng tóm tắt tổng hợp qua tất cả scenario."""
    print(f"\n{'='*80}")
    print(f"  OVERALL SUMMARY — avg_waiting (lower is better)")
    print(f"{'='*80}")

    col_w = 18
    hdr = "".join(f"{MODE_LABELS[m]:<{col_w}}" for m in MODES)
    print(f"  {'Scenario':<20}{hdr}")
    print(f"  {'─'*20}{'─'*col_w*len(MODES)}")

    for sc in SCENARIOS:
        row = f"  {sc:<20}"
        for mode in MODES:
            results = load_results(mode, sc)
            if results is None:
                row += f"{'N/A':<{col_w}}"
                continue
            vals = extract_last_n(results, "avg_waiting", n=20)
            row += f"{np.mean(vals):.1f}±{np.std(vals):.1f}{'':>{col_w - 12}}"
        print(row)

    print(f"{'='*80}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    scenarios = SCENARIOS if args.all else [args.scenario]

    for sc in scenarios:
        table = build_comparison_table(sc)
        print_table(sc, table)
        export_csv(sc, table)
        if args.plot:
            plot_learning_curves(sc)

    if args.all or args.summary:
        print_overall_summary()


def parse_args():
    p = argparse.ArgumentParser(description="Compare Fixed / IndepRL / MAS results")
    p.add_argument("--scenario", choices=SCENARIOS, default="peak_hour")
    p.add_argument("--all",     action="store_true",
                   help="Chạy tất cả scenarios")
    p.add_argument("--summary", action="store_true",
                   help="In overall summary table")
    p.add_argument("--plot",    action="store_true",
                   help="Vẽ learning curves (cần matplotlib)")
    return p.parse_args()


if __name__ == "__main__":
    main()
