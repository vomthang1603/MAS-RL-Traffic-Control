# =============================================================================
# evaluate.py — So sánh Fixed-Time vs Q-Learning vs DQN
#
# Chạy:
#   python -m training.evaluate --scenarios low_demand medium_demand peak_hour
#   python -m training.evaluate --scenarios all
# =============================================================================

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from config import (
    SCENARIOS, PLOTS_DIR, QTABLES_DIR, DQN_WEIGHTS_DIR,
    GL1_STATE_SIZE, GL2_STATE_SIZE,
    DECISION_FREQ,
)
from environment.sumo_env import SumoEnv


# ---------------------------------------------------------------------------
# Fixed-Time baseline
# ---------------------------------------------------------------------------

def run_fixed_time(scenario: str, use_gui: bool = False) -> dict:
    """Chạy 1 episode với đèn cố định (không RL)."""
    env         = SumoEnv(SCENARIOS[scenario], use_gui=use_gui)
    s1, s2      = env.reset()
    total_r     = 0.0
    total_q     = 0
    step_cnt    = 0
    done        = False

    while not done:
        (_, _), reward, done, info = env.step(0, 0)
        total_r += reward
        total_q += (info['gl1']['total_queue'] + info['gl2']['total_queue'])
        step_cnt += 1

    env.close()
    return {
        'method'      : 'Fixed-Time',
        'scenario'    : scenario,
        'total_reward': total_r,
        'avg_queue'   : total_q / max(step_cnt, 1),
    }


# ---------------------------------------------------------------------------
# Q-Learning evaluation
# ---------------------------------------------------------------------------

def run_qlearning(scenario: str, use_gui: bool = False) -> dict | None:
    from agents.q_learning_agent import QLearningAgent

    path_gl1 = os.path.join(QTABLES_DIR, f'qtable_GL1_{scenario}.pkl')
    path_gl2 = os.path.join(QTABLES_DIR, f'qtable_GL2_{scenario}.pkl')

    if not os.path.exists(path_gl1) or not os.path.exists(path_gl2):
        print(f"  [Evaluate] Q-table chưa tồn tại cho '{scenario}'.")
        print(f"  Chạy trước: python -m training.train_q --scenario {scenario}")
        return None

    agent_gl1 = QLearningAgent(GL1_STATE_SIZE, name='GL1', epsilon=0.0)
    agent_gl2 = QLearningAgent(GL2_STATE_SIZE, name='GL2', epsilon=0.0)
    agent_gl1.load(path_gl1)
    agent_gl2.load(path_gl2)

    env      = SumoEnv(SCENARIOS[scenario], use_gui=use_gui)
    s1, s2   = env.reset()
    total_r  = 0.0
    total_q  = 0
    step_cnt = 0
    done     = False

    while not done:
        if step_cnt % DECISION_FREQ == 0:
            a1 = agent_gl1.get_action(s1)
            a2 = agent_gl2.get_action(s2)
        else:
            a1, a2 = 0, 0

        (s1, s2), reward, done, info = env.step(a1, a2)
        total_r += reward
        total_q += (info['gl1']['total_queue'] + info['gl2']['total_queue'])
        step_cnt += 1

    env.close()
    return {
        'method'      : 'Q-Learning',
        'scenario'    : scenario,
        'total_reward': total_r,
        'avg_queue'   : total_q / max(step_cnt, 1),
    }


# ---------------------------------------------------------------------------
# DQN evaluation
# ---------------------------------------------------------------------------

def run_dqn(scenario: str, use_gui: bool = False) -> dict | None:
    try:
        from agents.dqn_agent import DQNAgent
    except ModuleNotFoundError as exc:
        if exc.name == 'tensorflow':
            print("  [Evaluate] Bỏ qua DQN — chưa cài TensorFlow.")
            return None
        raise

    path_gl1 = os.path.join(DQN_WEIGHTS_DIR, f'dqn_GL1_{scenario}.keras')
    path_gl2 = os.path.join(DQN_WEIGHTS_DIR, f'dqn_GL2_{scenario}.keras')

    if not os.path.exists(path_gl1) or not os.path.exists(path_gl2):
        print(f"  [Evaluate] DQN model chưa tồn tại cho '{scenario}'.")
        print(f"  Chạy trước: python -m training.train_dqn --scenario {scenario}")
        return None

    agent_gl1 = DQNAgent(GL1_STATE_SIZE, name='GL1', epsilon=0.0)
    agent_gl2 = DQNAgent(GL2_STATE_SIZE, name='GL2', epsilon=0.0)
    agent_gl1.load(path_gl1)
    agent_gl2.load(path_gl2)

    env      = SumoEnv(SCENARIOS[scenario], use_gui=use_gui)
    s1, s2   = env.reset()
    total_r  = 0.0
    total_q  = 0
    step_cnt = 0
    done     = False

    while not done:
        if step_cnt % DECISION_FREQ == 0:
            a1 = agent_gl1.get_action(s1)
            a2 = agent_gl2.get_action(s2)
        else:
            a1, a2 = 0, 0

        (s1, s2), reward, done, info = env.step(a1, a2)
        total_r += reward
        total_q += (info['gl1']['total_queue'] + info['gl2']['total_queue'])
        step_cnt += 1

    env.close()
    return {
        'method'      : 'DQN',
        'scenario'    : scenario,
        'total_reward': total_r,
        'avg_queue'   : total_q / max(step_cnt, 1),
    }


# ---------------------------------------------------------------------------
# Tổng hợp & vẽ biểu đồ so sánh
# ---------------------------------------------------------------------------

def benchmark(scenarios: list, use_gui: bool = False):
    results = []

    for sc in scenarios:
        print(f"\n--- Scenario: {sc} ---")

        r_ft = run_fixed_time(sc, use_gui)
        if r_ft:
            results.append(r_ft)
            print(f"  Fixed-Time  | Reward: {r_ft['total_reward']:10.1f} | "
                  f"AvgQ: {r_ft['avg_queue']:.2f}")

        r_ql = run_qlearning(sc, use_gui)
        if r_ql:
            results.append(r_ql)
            print(f"  Q-Learning  | Reward: {r_ql['total_reward']:10.1f} | "
                  f"AvgQ: {r_ql['avg_queue']:.2f}")

        r_dq = run_dqn(sc, use_gui)
        if r_dq:
            results.append(r_dq)
            print(f"  DQN         | Reward: {r_dq['total_reward']:10.1f} | "
                  f"AvgQ: {r_dq['avg_queue']:.2f}")

    if not results:
        print("\nKhông có kết quả để vẽ biểu đồ.")
        return

    _plot_comparison(results, scenarios)
    _print_table(results, scenarios)


def _plot_comparison(results: list, scenarios: list):
    methods = ['Fixed-Time', 'Q-Learning', 'DQN']
    colors  = ['#4C72B0', '#55A868', '#C44E52']
    x       = np.arange(len(scenarios))
    width   = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('So sánh hiệu suất: Fixed-Time vs Q-Learning vs DQN',
                 fontsize=13)

    for i, method in enumerate(methods):
        rewards = []
        queues  = []
        for sc in scenarios:
            match = [r for r in results
                     if r['method'] == method and r['scenario'] == sc]
            rewards.append(match[0]['total_reward'] if match else 0)
            queues.append(match[0]['avg_queue']     if match else 0)

        axes[0].bar(x + i * width, rewards, width, label=method,
                    color=colors[i], alpha=0.85)
        axes[1].bar(x + i * width, queues,  width, label=method,
                    color=colors[i], alpha=0.85)

    for ax, title, ylabel in [
        (axes[0], 'Total Reward (cao hơn = tốt hơn)',     'Total Reward'),
        (axes[1], 'Avg Queue Length (thấp hơn = tốt hơn)', 'Avg Queue'),
    ]:
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios, rotation=15)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'benchmark_comparison.png')
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\n  Benchmark plot saved → {save_path}")


def _print_table(results: list, scenarios: list):
    methods = ['Fixed-Time', 'Q-Learning', 'DQN']
    print(f"\n{'─'*70}")
    print(f"  {'Scenario':<18} {'Method':<14} {'Total Reward':>14} "
          f"{'Avg Queue':>12}")
    print(f"{'─'*70}")
    for sc in scenarios:
        for m in methods:
            match = [r for r in results
                     if r['method'] == m and r['scenario'] == sc]
            if match:
                r = match[0]
                print(f"  {sc:<18} {m:<14} {r['total_reward']:>14.1f} "
                      f"{r['avg_queue']:>12.2f}")
        print(f"{'─'*70}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark N4TD RL')
    parser.add_argument('--scenarios', nargs='+',
                        default=['low_demand', 'medium_demand', 'peak_hour'],
                        help="Danh sách scenario hoặc 'all'")
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()

    sc_list = (list(SCENARIOS.keys())
               if args.scenarios == ['all'] else args.scenarios)
    benchmark(sc_list, use_gui=args.gui)
