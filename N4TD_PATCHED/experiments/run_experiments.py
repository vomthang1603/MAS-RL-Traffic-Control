"""
experiments/run_experiments.py
--------------------------------
Chạy tự động toàn bộ thực nghiệm:
    3 mode × 5 scenario × N runs × 200 episodes

Ghi log tiến trình, ước tính thời gian còn lại,
tự động retry nếu 1 job thất bại.

Chạy:
    python experiments/run_experiments.py
    python experiments/run_experiments.py --dry-run     # kiểm tra pipeline
    python experiments/run_experiments.py --resume      # tiếp tục nếu bị gián đoạn
    python experiments/run_experiments.py --only-mas    # chỉ chạy MAS mode
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

ROOT       = Path(__file__).resolve().parent.parent
LOG_DIR    = ROOT / "output" / "experiment_logs"
STATUS_FILE = LOG_DIR / "experiment_status.json"

MODES     = ["fixed", "independent", "mas"]
SCENARIOS = ["peak_hour", "low_demand", "medium_demand", "asymmetric", "mixed_vehicle"]

# Cấu hình thực nghiệm đề xuất cho đồ án
DEFAULT_EPISODES = 200
DEFAULT_RUNS     = 3
MAX_RETRIES      = 2


# ──────────────────────────────────────────────────────────────────────────────
# Job definition
# ──────────────────────────────────────────────────────────────────────────────

def build_job_list(modes: List[str], scenarios: List[str],
                   episodes: int, runs: int, seed: int) -> List[dict]:
    """Tạo danh sách tất cả jobs cần chạy."""
    jobs = []
    for mode in modes:
        for scenario in scenarios:
            jobs.append({
                "mode":     mode,
                "scenario": scenario,
                "episodes": episodes,
                "runs":     runs,
                "seed":     seed,
                "status":   "pending",   # pending | done | failed
                "retries":  0,
                "elapsed":  None,
            })
    return jobs


# ──────────────────────────────────────────────────────────────────────────────
# Status persistence (để resume)
# ──────────────────────────────────────────────────────────────────────────────

def save_status(jobs: List[dict]):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)


def load_status() -> List[dict]:
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Job runner
# ──────────────────────────────────────────────────────────────────────────────

def run_job(job: dict, dry_run: bool = False) -> Tuple[bool, float]:
    """
    Chạy 1 job. Returns (success, elapsed_seconds).
    """
    cmd = [
        sys.executable,
        str(ROOT / "training" / "train_mas.py"),
        "--mode",     job["mode"],
        "--scenario", job["scenario"],
        "--episodes", str(job["episodes"]),
        "--runs",     str(job["runs"]),
        "--seed",     str(job["seed"]),
    ]

    if dry_run:
        print(f"    [DRY-RUN] {' '.join(cmd)}")
        time.sleep(0.1)
        return True, 0.1

    log_path = LOG_DIR / f"{job['mode']}_{job['scenario']}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    return result.returncode == 0, elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Progress display
# ──────────────────────────────────────────────────────────────────────────────

def format_eta(elapsed_per_job: float, remaining: int) -> str:
    if elapsed_per_job <= 0:
        return "N/A"
    eta_sec = elapsed_per_job * remaining
    return str(timedelta(seconds=int(eta_sec)))


def print_progress(jobs: List[dict], current_idx: int, elapsed: float):
    done    = sum(1 for j in jobs if j["status"] == "done")
    failed  = sum(1 for j in jobs if j["status"] == "failed")
    total   = len(jobs)
    pending = total - done - failed

    avg_time = elapsed / max(done, 1)
    eta = format_eta(avg_time, pending)

    bar_len  = 30
    filled   = int(bar_len * done / total)
    bar      = "█" * filled + "░" * (bar_len - filled)

    print(f"\n  Progress: [{bar}] {done}/{total}")
    print(f"  Done: {done}  Failed: {failed}  Pending: {pending}")
    print(f"  ETA: {eta}  (avg {avg_time:.0f}s/job)")


def print_final_report(jobs: List[dict], total_elapsed: float):
    done   = [j for j in jobs if j["status"] == "done"]
    failed = [j for j in jobs if j["status"] == "failed"]

    print(f"\n{'='*65}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total time: {timedelta(seconds=int(total_elapsed))}")
    print(f"  Jobs done:  {len(done)}/{len(jobs)}")
    if failed:
        print(f"  Jobs failed ({len(failed)}):")
        for j in failed:
            print(f"    - {j['mode']}/{j['scenario']} (retries={j['retries']})")
    print(f"{'='*65}\n")

    print("  Next step:")
    print("    python evaluation/compare_results.py --all --plot\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Xác định modes cần chạy
    if args.only_mas:
        modes = ["mas"]
    elif args.only_rl:
        modes = ["independent", "mas"]
    else:
        modes = MODES

    # Load hoặc tạo job list
    if args.resume and STATUS_FILE.exists():
        jobs = load_status()
        pending = [j for j in jobs if j["status"] != "done"]
        print(f"[resume] Loaded {len(jobs)} jobs, {len(pending)} pending.")
    else:
        jobs = build_job_list(
            modes=modes,
            scenarios=SCENARIOS,
            episodes=args.episodes,
            runs=args.runs,
            seed=args.seed,
        )
        save_status(jobs)

    total     = len(jobs)
    t_start   = time.time()
    elapsed_done = 0.0

    print(f"\n{'='*65}")
    print(f"  N4TD Experiments — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Jobs total:  {total}")
    print(f"  Modes:       {modes}")
    print(f"  Episodes:    {args.episodes}  Runs: {args.runs}")
    if args.dry_run:
        print(f"  *** DRY-RUN MODE — no actual training ***")
    print(f"{'='*65}\n")

    for i, job in enumerate(jobs):
        if job["status"] == "done":
            continue

        mode, sc = job["mode"], job["scenario"]
        print(f"\n[{i+1}/{total}] {mode.upper()} / {sc}")

        success = False
        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                print(f"  ↺ Retry {attempt}/{MAX_RETRIES}...")
                time.sleep(5)

            ok, elapsed = run_job(job, dry_run=args.dry_run)

            if ok:
                job["status"]  = "done"
                job["elapsed"] = elapsed
                elapsed_done  += elapsed
                print(f"  ✓ {elapsed:.0f}s")
                success = True
                break
            else:
                job["retries"] += 1

        if not success:
            job["status"] = "failed"
            print(f"  ✗ Failed after {MAX_RETRIES} retries")
            if args.fail_fast:
                save_status(jobs)
                sys.exit(1)

        save_status(jobs)
        print_progress(jobs, i, elapsed_done)

    # Sinh báo cáo tự động
    if not args.dry_run and not args.no_report:
        print("\n[run_experiments] Generating comparison report...")
        subprocess.run([
            sys.executable,
            str(ROOT / "evaluation" / "compare_results.py"),
            "--all", "--plot", "--summary",
        ])

    total_elapsed = time.time() - t_start
    print_final_report(jobs, total_elapsed)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run all N4TD MAS experiments automatically"
    )
    p.add_argument("--episodes",   type=int, default=DEFAULT_EPISODES)
    p.add_argument("--runs",       type=int, default=DEFAULT_RUNS)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--dry-run",    action="store_true",
                   help="Test pipeline mà không chạy SUMO thật")
    p.add_argument("--resume",     action="store_true",
                   help="Tiếp tục từ lần chạy bị gián đoạn")
    p.add_argument("--only-mas",   action="store_true",
                   help="Chỉ chạy MAS mode (bỏ fixed và independent)")
    p.add_argument("--only-rl",    action="store_true",
                   help="Chỉ chạy independent + mas (bỏ fixed)")
    p.add_argument("--fail-fast",  action="store_true",
                   help="Dừng ngay khi có job thất bại")
    p.add_argument("--no-report",  action="store_true",
                   help="Không sinh báo cáo sau khi xong")
    return p.parse_args()


if __name__ == "__main__":
    main()
