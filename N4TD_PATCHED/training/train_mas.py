"""
training/train_mas.py
----------------------
Vòng lặp training MAS hoàn chỉnh.

Chạy:
    python training/train_mas.py --scenario peak_hour --episodes 200 --runs 3

So sánh 3 chế độ (--mode):
    fixed       → FixedTimeAgent (baseline)
    independent → 2 CooperativeAgent riêng lẻ (không bus, không shaping)
    mas         → 2 CooperativeAgent + MessageBus + RewardShaper (MAS đầy đủ)
"""

import argparse
import os
import json
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ── Project imports ────────────────────────────────────────────────────────────
# Điều chỉnh sys.path nếu chạy từ thư mục gốc dự án
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from communication.message_bus import MessageBus, AgentMessage
from communication.reward_shaper import RewardShaper, NetworkState
from agents.cooperative_agent import CooperativeAgent
from baselines.fixed_time import FixedTimeAgent, make_gl1_fixed_plan, make_gl2_fixed_plan

# SUMO / TraCI
try:
    import traci
    import sumolib
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("[WARNING] TraCI not found. Running in dry-run mode.")


# ──────────────────────────────────────────────────────────────────────────────
# Hằng số
# ──────────────────────────────────────────────────────────────────────────────

DECISION_FREQ   = 10        # số bước SUMO giữa 2 lần ra quyết định
SIM_STEPS       = 3600      # tổng bước / episode (= 1 giờ với step=1s)
STATE_SIZE_GL1  = 25        # 18 local + 7 từ GL2
STATE_SIZE_GL2  = 18
ACTION_SIZE_GL1 = 12
ACTION_SIZE_GL2 = 8

SCENARIO_CFGS = {
    "peak_hour":    "cfg/peak_hour.sumocfg",
    "low_demand":   "cfg/low_demand.sumocfg",
    "medium_demand":"cfg/medium_demand.sumocfg",
    "asymmetric":   "cfg/asymmetric.sumocfg",
    "mixed_vehicle":"cfg/mixed_vehicle.sumocfg",
}

OUTPUT_DIR = ROOT / "output" / "mas_results"


# ──────────────────────────────────────────────────────────────────────────────
# SUMO Environment helpers
# ──────────────────────────────────────────────────────────────────────────────

def start_sumo(cfg_path: str, gui: bool = False, seed: int = 42) -> None:
    sumo_bin = "sumo-gui" if gui else "sumo"
    cmd = [
        sumo_bin,
        "-c", cfg_path,
        "--seed", str(seed),
        "--no-step-log", "true",
        "--waiting-time-memory", "100",
        "--time-to-teleport", "-1",
    ]
    traci.start(cmd)


def get_state_gl1() -> np.ndarray:
    """Lấy 18 chiều state cục bộ của GL1."""
    tls_id = "GL1"
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    unique_lanes = list(dict.fromkeys(lanes))[:9]   # lấy tối đa 9 làn

    state = []
    for lane in unique_lanes:
        state.append(min(traci.lane.getLastStepHaltingNumber(lane) / 30.0, 1.0))
        state.append(min(traci.lane.getLastStepOccupancy(lane), 1.0))
    # pad nếu ít hơn 9 làn
    while len(state) < 18:
        state.append(0.0)
    return np.array(state[:18], dtype=np.float32)


def get_state_gl2() -> np.ndarray:
    """Lấy 18 chiều state cục bộ của GL2."""
    tls_id = "GL2"
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    unique_lanes = list(dict.fromkeys(lanes))[:9]

    state = []
    for lane in unique_lanes:
        state.append(min(traci.lane.getLastStepHaltingNumber(lane) / 30.0, 1.0))
        state.append(min(traci.lane.getLastStepOccupancy(lane), 1.0))
    while len(state) < 18:
        state.append(0.0)
    return np.array(state[:18], dtype=np.float32)


def build_network_state() -> NetworkState:
    """Tổng hợp NetworkState từ TraCI."""
    def lane_queues(tls_id):
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        return {l: float(traci.lane.getLastStepHaltingNumber(l)) for l in lanes}

    def avg_wait(tls_id):
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        waits = [traci.lane.getWaitingTime(l) for l in lanes]
        return float(np.mean(waits)) if waits else 0.0

    return NetworkState(
        gl1_queues=lane_queues("GL1"),
        gl2_queues=lane_queues("GL2"),
        gl1_waiting=avg_wait("GL1"),
        gl2_waiting=avg_wait("GL2"),
        sim_time=traci.simulation.getTime(),
    )


def set_phase(tls_id: str, phase: int) -> None:
    traci.trafficlight.setPhase(tls_id, phase)


def collect_episode_stats() -> Dict[str, float]:
    """Thu thập metrics cuối episode."""
    vehicles = traci.vehicle.getIDList()
    if not vehicles:
        return {"avg_waiting": 0.0, "avg_speed": 0.0, "throughput": 0}
    waiting = [traci.vehicle.getWaitingTime(v) for v in vehicles]
    speeds  = [traci.vehicle.getSpeed(v) for v in vehicles]
    return {
        "avg_waiting":  float(np.mean(waiting)),
        "avg_speed":    float(np.mean(speeds)),
        "throughput":   len(vehicles),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(
    agent_gl1,
    agent_gl2,
    bus: Optional[MessageBus],
    shaper: Optional[RewardShaper],
    mode: str,
    training: bool = True,
    sim_time_limit: int = SIM_STEPS,
) -> Dict:
    """
    Chạy một episode simulation.

    Returns dict với: total_reward_gl1, total_reward_gl2, losses, stats
    """
    if shaper:
        shaper.reset()

    total_reward = {"GL1": 0.0, "GL2": 0.0}
    losses = {"GL1": [], "GL2": []}
    step = 0

    while step < sim_time_limit:
        # Bước SUMO
        traci.simulationStep()
        step += 1

        # Ra quyết định mỗi DECISION_FREQ bước
        if step % DECISION_FREQ != 0:
            continue

        sim_time = traci.simulation.getTime()

        # ── Lấy state ──────────────────────────────────────────────────
        s2_local = get_state_gl2()
        s1_local = get_state_gl1()

        # GL2 publish lên bus
        if bus is not None:
            lanes2 = list(dict.fromkeys(
                traci.trafficlight.getControlledLanes("GL2")
            ))
            msg = AgentMessage(
                sender_id="GL2",
                timestamp=sim_time,
                queue_lengths={l: traci.lane.getLastStepHaltingNumber(l)
                               for l in lanes2},
                avg_waiting_time=float(np.mean(
                    [traci.lane.getWaitingTime(l) for l in lanes2]
                )),
                current_phase=traci.trafficlight.getPhase("GL2"),
                occupancy={l: traci.lane.getLastStepOccupancy(l)
                           for l in lanes2},
            )
            bus.publish("GL2", msg)

        # GL1 augment state nếu là MAS
        s1 = agent_gl1.augment_state(s1_local) if hasattr(agent_gl1, "augment_state") \
             else s1_local

        # ── Chọn action ────────────────────────────────────────────────
        if mode == "fixed":
            a1 = agent_gl1.act(s1, sim_time=sim_time)
            a2 = agent_gl2.act(s2_local, sim_time=sim_time)
        else:
            a1 = agent_gl1.act(s1, training=training)
            a2 = agent_gl2.act(s2_local, training=training)

        set_phase("GL1", a1)
        set_phase("GL2", a2)

        # Bước tiếp theo (lấy next state sau khi đã set phase)
        traci.simulationStep()
        step += 1

        ns2_local = get_state_gl2()
        ns1_local = get_state_gl1()
        ns1 = agent_gl1.augment_state(ns1_local) if hasattr(agent_gl1, "augment_state") \
              else ns1_local

        done = step >= sim_time_limit

        # ── Tính reward ────────────────────────────────────────────────
        if shaper is not None:
            net_state = build_network_state()
            rewards = shaper.compute_both(net_state)
            r1, r2 = rewards["GL1"], rewards["GL2"]
        else:
            # Independent: reward = -local_queue
            r1 = -sum(
                traci.lane.getLastStepHaltingNumber(l)
                for l in dict.fromkeys(
                    traci.trafficlight.getControlledLanes("GL1")
                )
            ) / 30.0
            r2 = -sum(
                traci.lane.getLastStepHaltingNumber(l)
                for l in dict.fromkeys(
                    traci.trafficlight.getControlledLanes("GL2")
                )
            ) / 30.0

        total_reward["GL1"] += r1
        total_reward["GL2"] += r2

        # ── Learn ──────────────────────────────────────────────────────
        if training and mode != "fixed":
            agent_gl1.remember(s1, a1, r1, ns1, done)
            agent_gl2.remember(s2_local, a2, r2, ns2_local, done)

            loss1 = agent_gl1.learn()
            loss2 = agent_gl2.learn()
            if loss1 is not None:
                losses["GL1"].append(loss1)
            if loss2 is not None:
                losses["GL2"].append(loss2)

    stats = collect_episode_stats()
    return {
        "reward_gl1":   total_reward["GL1"],
        "reward_gl2":   total_reward["GL2"],
        "loss_gl1":     float(np.mean(losses["GL1"])) if losses["GL1"] else None,
        "loss_gl2":     float(np.mean(losses["GL2"])) if losses["GL2"] else None,
        **stats,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_path = str(ROOT / SCENARIO_CFGS[args.scenario])
    all_runs_results = []

    for run in range(args.runs):
        seed = args.seed + run
        random.seed(seed)
        np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"  Mode: {args.mode.upper()} | Scenario: {args.scenario} | Run {run+1}/{args.runs}")
        print(f"{'='*60}")

        # ── Khởi tạo components ────────────────────────────────────────
        bus    = MessageBus()    if args.mode == "mas"         else None
        shaper = RewardShaper()  if args.mode == "mas"         else None

        if args.mode == "fixed":
            agent_gl1 = FixedTimeAgent("GL1")
            agent_gl2 = FixedTimeAgent("GL2")
        else:
            state_gl1 = STATE_SIZE_GL1 if args.mode == "mas" else STATE_SIZE_GL2
            agent_gl1 = CooperativeAgent(
                "GL1", state_gl1, ACTION_SIZE_GL1,
                message_bus=bus if args.mode == "mas" else None,
            )
            agent_gl2 = CooperativeAgent(
                "GL2", STATE_SIZE_GL2, ACTION_SIZE_GL2,
                message_bus=None,
            )

        run_results = []

        for ep in range(args.episodes):
            if not TRACI_AVAILABLE:
                # Dry-run: sinh dữ liệu giả để test pipeline
                result = {
                    "reward_gl1": -random.uniform(50, 200),
                    "reward_gl2": -random.uniform(30, 120),
                    "loss_gl1": random.uniform(0.01, 0.5) if args.mode != "fixed" else None,
                    "loss_gl2": random.uniform(0.01, 0.5) if args.mode != "fixed" else None,
                    "avg_waiting": random.uniform(10, 60),
                    "avg_speed": random.uniform(2, 10),
                    "throughput": random.randint(300, 700),
                }
            else:
                start_sumo(cfg_path, gui=args.gui, seed=seed + ep)
                try:
                    result = run_episode(
                        agent_gl1, agent_gl2, bus, shaper,
                        mode=args.mode,
                        training=True,
                        sim_time_limit=SIM_STEPS,
                    )
                finally:
                    traci.close()

            run_results.append(result)

            # Log
            ep_str = f"Ep {ep+1:>3}/{args.episodes}"
            r_str  = f"R_GL1={result['reward_gl1']:>8.1f}  R_GL2={result['reward_gl2']:>8.1f}"
            w_str  = f"AvgWait={result['avg_waiting']:>5.1f}s"
            l_str  = (f"Loss={result['loss_gl1']:.4f}"
                      if result.get("loss_gl1") else "Loss=N/A")
            print(f"  {ep_str} | {r_str} | {w_str} | {l_str}")

            # Lưu checkpoint mỗi 50 episode
            if args.mode != "fixed" and (ep + 1) % 50 == 0:
                ckpt_dir = OUTPUT_DIR / args.mode / args.scenario / f"run{run+1}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                agent_gl1.save(str(ckpt_dir / f"gl1_ep{ep+1}.weights.h5"))
                agent_gl2.save(str(ckpt_dir / f"gl2_ep{ep+1}.weights.h5"))
                print(f"  → Checkpoint saved: {ckpt_dir}")

        all_runs_results.append(run_results)

    # ── Lưu kết quả JSON ──────────────────────────────────────────────
    out_file = OUTPUT_DIR / f"{args.mode}_{args.scenario}_results.json"
    with open(out_file, "w") as f:
        json.dump(all_runs_results, f, indent=2)
    print(f"\n✓ Results saved → {out_file}")

    # ── In tóm tắt ────────────────────────────────────────────────────
    print_summary(all_runs_results, args)


def print_summary(results: List[List[Dict]], args):
    """In bảng tóm tắt kết quả qua các run."""
    print(f"\n{'─'*60}")
    print(f"  SUMMARY | Mode: {args.mode} | Scenario: {args.scenario}")
    print(f"{'─'*60}")

    for metric in ["avg_waiting", "avg_speed", "throughput"]:
        all_vals = [ep[metric] for run in results for ep in run[-20:]]  # 20 ep cuối
        print(f"  {metric:<15}: mean={np.mean(all_vals):.2f}  std={np.std(all_vals):.2f}")
    print(f"{'─'*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train MAS Traffic Control")
    p.add_argument("--mode",     choices=["fixed", "independent", "mas"],
                   default="mas", help="Chế độ chạy")
    p.add_argument("--scenario", choices=list(SCENARIO_CFGS.keys()),
                   default="peak_hour", help="Kịch bản traffic")
    p.add_argument("--episodes", type=int, default=200, help="Số episode / run")
    p.add_argument("--runs",     type=int, default=3,   help="Số lần chạy lặp")
    p.add_argument("--seed",     type=int, default=42,  help="Random seed gốc")
    p.add_argument("--gui",      action="store_true",   help="Bật SUMO GUI")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
