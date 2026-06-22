#!/usr/bin/env python3
# =============================================================================
# run_sim.py - CHAY MO PHONG SAU TRAIN (replay chinh sach da hoc trong SUMO)
#
#   Nap model da train (.save tu train_q/train_dqn) roi chay 1 (hoac vai) tap,
#   THAM LAM (epsilon=0, khong explore, khong train), de xem/ket den hoc duoc gi.
#   Dung dung API cua SumoEnv / QLearningAgent / DQNAgent trong project.
#
# DAT FILE NAY O GOC PROJECT (cung cap config.py).
#
# Chay co GUI (may local, da cai sumo-gui, KHONG chay cell S6 libsumo):
#   python run_sim.py --method dqn --scenario peak_hour --gui
#
# Chay khong GUI (Colab headless / chi muon so):
#   python run_sim.py --method q --scenario peak_hour
# =============================================================================
import os, sys, time, glob, argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # goc project
import numpy as np

from config import (SCENARIOS, DECISION_FREQ, GL1_STATE_SIZE, GL2_STATE_SIZE)
from environment.sumo_env import SumoEnv
from environment import sumo_env            # lay dung traci/libsumo ma SumoEnv dung
traci = sumo_env.traci


def sample_traffic_stats():
    veh = traci.vehicle.getIDList()
    if not veh:
        return None, None
    wait = float(np.mean([traci.vehicle.getWaitingTime(v) for v in veh]))
    spd  = float(np.mean([traci.vehicle.getSpeed(v) for v in veh]))
    return wait, spd


ROOT = os.path.dirname(os.path.abspath(__file__))


def model_path(method, name, scenario):
    """Dung dung quy uoc cua save():
       Q  -> models/q_tables/qtable_{name}_{scenario}.pkl
       DQN-> models/dqn_weights/dqn_{name}_{scenario}.keras
    """
    if method == 'q':
        return os.path.join(ROOT, 'models', 'q_tables',   f'qtable_{name}_{scenario}.pkl')
    return os.path.join(ROOT, 'models', 'dqn_weights', f'dqn_{name}_{scenario}.keras')


def build_agents(method, scenario):
    """Tao 2 agent GL1/GL2 va NAP model da train. LUU Y: load() nhan PATH, khong phai scenario."""
    if method == 'q':
        from agents.q_learning_agent import QLearningAgent as A
    else:
        from agents.dqn_agent import DQNAgent as A
    a1 = A(state_size=GL1_STATE_SIZE, name='GL1', epsilon=0.0)
    a2 = A(state_size=GL2_STATE_SIZE, name='GL2', epsilon=0.0)
    a1.epsilon = 0.0; a2.epsilon = 0.0   # tham lam tuyet doi: chi khai thac chinh sach da hoc
    for a in (a1, a2):
        p = model_path(method, a.name, scenario)
        if not os.path.exists(p):
            sys.exit(f"[loi] Khong thay file model:\n      {p}\n"
                     f"      -> Kiem tra da tai dung file tu Drive vao models/ chua, "
                     f"va scenario/method co khop ten file khong.")
        a.load(p)
    return a1, a2


def run_one(env, a1, a2, method, gui=False, delay_ms=0):
    s1, s2 = env.reset()
    if method == 'q':
        k1, k2 = a1.discretize_state(s1), a2.discretize_state(s2)
    queue_sum = wait_sum = spd_sum = 0.0
    queue_n = wait_n = spd_n = arrived = 0
    step, done = 0, False
    while not done:
        decide = (step % DECISION_FREQ == 0)
        if decide:
            if method == 'q':
                a1act = a1.get_action_from_key(k1, 0.0)
                a2act = a2.get_action_from_key(k2, 0.0)
            else:
                a1act = a1.get_action(s1); a2act = a2.get_action(s2)
        else:
            a1act, a2act = 0, 0
        (ns1, ns2), _, done, info = env.step(a1act, a2act)
        if method == 'q':
            k1, k2 = a1.discretize_state(ns1), a2.discretize_state(ns2)
        else:
            s1, s2 = ns1, ns2
        arrived   += int(traci.simulation.getArrivedNumber())
        queue_sum += info['gl1']['total_queue'] + info['gl2']['total_queue']; queue_n += 1
        if decide:
            w, sp = sample_traffic_stats()
            if w is not None:
                wait_sum += w; wait_n += 1; spd_sum += sp; spd_n += 1
        if gui and delay_ms:               # phach nhip de quay video cho muot
            time.sleep(delay_ms / 1000.0)
        step += 1
    env.close()
    return {
        'avg_waiting': wait_sum / wait_n if wait_n else 0.0,
        'avg_queue':   queue_sum / queue_n if queue_n else 0.0,
        'avg_speed':   spd_sum / spd_n if spd_n else 0.0,
        'throughput':  arrived,
    }


def find_sumocfg(scenario, override=None):
    """Tu tim file .sumocfg cua kich ban (de chay den co dinh goc cua SUMO)."""
    if override:
        p = override if os.path.isabs(override) else os.path.join(ROOT, override)
        if os.path.exists(p):
            return p
        sys.exit(f"[loi] Khong thay sumocfg: {p}")
    cfg = SCENARIOS[scenario]
    cands = []
    vals = cfg.values() if isinstance(cfg, dict) else getattr(cfg, '__dict__', {}).values()
    for v in vals:
        if isinstance(v, str) and v.lower().endswith('.sumocfg'):
            cands.append(v if os.path.isabs(v) else os.path.join(ROOT, v))
    if not cands:                          # du phong: glob trong scenarios/ va cfg/
        for d in ('scenarios', 'cfg', '.'):
            cands += glob.glob(os.path.join(ROOT, d, '**', f'*{scenario}*.sumocfg'), recursive=True)
    cands = [c for c in cands if os.path.exists(c)]
    if not cands:
        sys.exit(f"[loi] Khong tu tim duoc .sumocfg cho '{scenario}'.\n"
                 f"      Chay:  dir /s /b *.sumocfg   roi them  --sumocfg <duong_dan>")
    return cands[0]


def run_fixed(scenario, gui=False, delay_ms=0, sumocfg=None):
    """Baseline DEN CO DINH: chay chuong trinh den GOC trong .net.xml, khong dieu khien bang RL."""
    cfgfile = find_sumocfg(scenario, sumocfg)
    binname = 'sumo-gui' if gui else 'sumo'
    cmd = [binname, '-c', cfgfile, '--start', '--quit-on-end', 'true', '--no-step-log', 'true']
    if gui and delay_ms:
        cmd += ['--delay', str(delay_ms)]
    if traci.isLoaded():
        traci.close()
    traci.start(cmd)
    wait_sum = q_sum = 0.0
    wait_n = q_n = arrived = step = 0
    lanes = None
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        arrived += int(traci.simulation.getArrivedNumber())
        if step % DECISION_FREQ == 0:
            w, _ = sample_traffic_stats()
            if w is not None:
                wait_sum += w; wait_n += 1
            if lanes is None:
                lanes = traci.lane.getIDList()
            q_sum += sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes); q_n += 1
        step += 1
    traci.close()
    return {
        'avg_waiting': wait_sum / wait_n if wait_n else 0.0,
        # luu y: queue nay cong tren MOI lane (khac detector cua env) -> chi tham khao;
        # so chuan cho fixed lay tu bao cao / fixed_*.json
        'avg_queue':   q_sum / q_n if q_n else 0.0,
        'avg_speed':   0.0,
        'throughput':  arrived,
    }


def main():
    p = argparse.ArgumentParser(description='Chay mo phong sau train (fixed / replay chinh sach)')
    p.add_argument('--method',   choices=['fixed', 'q', 'dqn'], required=True)
    p.add_argument('--scenario', choices=list(SCENARIOS.keys()), required=True)
    p.add_argument('--gui',      action='store_true', help='Mo sumo-gui (can traci that + co man hinh)')
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--delay-ms', type=int, default=60, help='Do tre moi buoc khi co GUI (ms) de quay video cho muot')
    p.add_argument('--sumocfg',  default=None, help='(fixed) duong dan .sumocfg neu khong tu tim duoc')
    args = p.parse_args()

    def show(ep, r):
        print(f"  Tap {ep+1}/{args.episodes} | AvgWait={r['avg_waiting']:6.2f}s | "
              f"AvgQueue={r['avg_queue']:6.2f} | Throughput={r['throughput']}")

    if args.method == 'fixed':
        print(f"Chay FIXED (den goc SUMO) / {args.scenario} ...")
        for ep in range(args.episodes):
            r = run_fixed(args.scenario, gui=args.gui, delay_ms=args.delay_ms, sumocfg=args.sumocfg)
            show(ep, r)
        return

    print(f"Nap model {args.method.upper()} / {args.scenario} ...")
    a1, a2 = build_agents(args.method, args.scenario)
    cfg = SCENARIOS[args.scenario]
    for ep in range(args.episodes):
        env = SumoEnv(cfg, use_gui=args.gui)
        r = run_one(env, a1, a2, args.method, gui=args.gui, delay_ms=args.delay_ms)
        show(ep, r)


if __name__ == '__main__':
    main()