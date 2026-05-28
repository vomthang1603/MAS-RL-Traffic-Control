# =============================================================================
# train_q.py — Vòng lặp training Q-Learning cho N4TD
#
# Sử dụng QLearningAgent (agents/q_learning_agent.py) và SumoEnv
# (environment/sumo_env.py).  Q-table được lưu vào models/q_tables/.
#
# Chạy:
#   python -m training.train_q --scenario low_demand --episodes 50
#   python -m training.train_q --scenario peak_hour  --gui
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
    SCENARIOS, EPSILON, TOTAL_EPISODES, EVAL_INTERVAL,
    DECISION_FREQ, PLOTS_DIR, QTABLES_DIR,
    GL1_STATE_SIZE, GL2_STATE_SIZE,
)
from environment.sumo_env import SumoEnv
from agents.q_learning_agent import QLearningAgent


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(scenario: str = 'low_demand',
          episodes: int = TOTAL_EPISODES,
          epsilon: float = EPSILON,
          use_gui: bool = False):
    """
    Train 2 QLearningAgent (GL1 + GL2) trên kịch bản cho trước.

    Returns
    -------
    ep_rewards : list[float]
    ep_queues  : list[float]
    """
    cfg_path  = SCENARIOS[scenario]
    env       = SumoEnv(cfg_path, use_gui=use_gui)
    agent_gl1 = QLearningAgent(state_size=GL1_STATE_SIZE,
                               name='GL1', epsilon=epsilon)
    agent_gl2 = QLearningAgent(state_size=GL2_STATE_SIZE,
                               name='GL2', epsilon=epsilon)

    ep_rewards = []
    ep_queues  = []

    print(f"\n{'='*60}")
    print(f"  Q-Learning | Scenario: {scenario} | Episodes: {episodes}")
    print(f"{'='*60}")

    for ep in range(episodes):
        s1, s2   = env.reset()
        k1       = agent_gl1.discretize_state(s1)
        k2       = agent_gl2.discretize_state(s2)

        total_reward = 0.0
        total_queue  = 0
        step_count   = 0
        done         = False

        while not done:
            if step_count % DECISION_FREQ == 0:
                a1 = agent_gl1.get_action_from_key(k1, agent_gl1.epsilon)
                a2 = agent_gl2.get_action_from_key(k2, agent_gl2.epsilon)
            else:
                a1, a2 = 0, 0

            (ns1, ns2), reward, done, info = env.step(a1, a2)

            nk1 = agent_gl1.discretize_state(ns1)
            nk2 = agent_gl2.discretize_state(ns2)

            if step_count % DECISION_FREQ == 0:
                agent_gl1.train(k1, a1, info['r1'], nk1)
                agent_gl2.train(k2, a2, info['r2'], nk2)

            k1, k2 = nk1, nk2
            total_reward += reward
            total_queue  += (info['gl1']['total_queue']
                             + info['gl2']['total_queue'])
            step_count   += 1

        ep_rewards.append(total_reward)
        ep_queues.append(total_queue / max(step_count, 1))

        # Epsilon decay
        agent_gl1.decay_epsilon()
        agent_gl2.decay_epsilon()

        if (ep % EVAL_INTERVAL == 0) or (ep == episodes - 1):
            print(f"  Ep {ep+1:4d}/{episodes} | "
                  f"Reward: {total_reward:10.1f} | "
                  f"Avg Queue: {ep_queues[-1]:.2f} | "
                  f"eps={agent_gl1.epsilon:.3f}")

    env.close()

    # Lưu Q-table
    agent_gl1.save(scenario)
    agent_gl2.save(scenario)

    # Vẽ biểu đồ
    _plot_results(ep_rewards, ep_queues, scenario, method='Q-Learning')

    return ep_rewards, ep_queues


# ---------------------------------------------------------------------------
# Evaluation (ε=0, không exploration)
# ---------------------------------------------------------------------------

def evaluate(scenario: str = 'low_demand',
             qtable_gl1_path: str = None,
             qtable_gl2_path: str = None,
             use_gui: bool = False):
    """Chạy 1 episode với policy đã train, không exploration."""

    cfg_path  = SCENARIOS[scenario]
    env       = SumoEnv(cfg_path, use_gui=use_gui)
    agent_gl1 = QLearningAgent(GL1_STATE_SIZE, name='GL1', epsilon=0.0)
    agent_gl2 = QLearningAgent(GL2_STATE_SIZE, name='GL2', epsilon=0.0)

    if qtable_gl1_path:
        agent_gl1.load(qtable_gl1_path)
    if qtable_gl2_path:
        agent_gl2.load(qtable_gl2_path)

    s1, s2   = env.reset()
    total_r  = 0.0
    total_q  = 0
    step_cnt = 0
    done     = False

    while not done:
        # Fix: chỉ ra quyết định mỗi DECISION_FREQ bước (giống vòng train)
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
    avg_q = total_q / max(step_cnt, 1)
    print(f"\n[Evaluate | Q-Learning | {scenario}] "
          f"Reward: {total_r:.1f} | Avg Queue: {avg_q:.2f}")
    return total_r, avg_q


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(rewards, queues, scenario, method='Q-Learning'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{method} — {scenario}', fontsize=14)

    axes[0].plot(rewards, linewidth=1.2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Reward mỗi Episode')
    axes[0].grid(True, alpha=0.4)

    win = min(5, len(queues))
    ma  = np.convolve(queues, np.ones(win) / win, mode='valid')
    axes[1].plot(queues, alpha=0.4, label='Raw')
    axes[1].plot(range(win - 1, len(queues)), ma,
                 linewidth=2, label=f'MA-{win}')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Avg Queue Length')
    axes[1].set_title('Queue Length trung bình mỗi Episode')
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    fname     = f"{method.lower().replace(' ', '_')}_{scenario}.png"
    save_path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Q-Learning cho N4TD')
    parser.add_argument('--scenario', default='low_demand',
                        choices=list(SCENARIOS.keys()))
    parser.add_argument('--episodes', type=int, default=TOTAL_EPISODES)
    parser.add_argument('--gui',      action='store_true')
    args = parser.parse_args()

    train(scenario=args.scenario, episodes=args.episodes, use_gui=args.gui)
