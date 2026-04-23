# =============================================================================
# train_dqn.py — Vòng lặp training DQN cho N4TD
#
# Sử dụng DQNAgent (agents/dqn_agent.py) và SumoEnv.
# Weights được lưu vào models/dqn_weights/, biểu đồ vào results/plots/.
#
# Chạy:
#   python -m training.train_dqn --scenario low_demand --episodes 50
#   python -m training.train_dqn --scenario peak_hour  --eval \
#          --model_gl1 models/dqn_weights/dqn_GL1_peak_hour.keras \
#          --model_gl2 models/dqn_weights/dqn_GL2_peak_hour.keras
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
    SCENARIOS, TOTAL_EPISODES, EVAL_INTERVAL,
    DECISION_FREQ, PLOTS_DIR,
    GL1_STATE_SIZE, GL2_STATE_SIZE, EPSILON,
)
from environment.sumo_env import SumoEnv
from agents.dqn_agent import DQNAgent


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(scenario: str = 'low_demand',
          episodes: int = TOTAL_EPISODES,
          use_gui: bool = False):

    cfg_path  = SCENARIOS[scenario]
    env       = SumoEnv(cfg_path, use_gui=use_gui)
    agent_gl1 = DQNAgent(state_size=GL1_STATE_SIZE, name='GL1', epsilon=EPSILON)
    agent_gl2 = DQNAgent(state_size=GL2_STATE_SIZE, name='GL2', epsilon=EPSILON)

    ep_rewards   = []
    ep_queues    = []
    ep_losses_g1 = []
    ep_losses_g2 = []

    print(f"\n{'='*60}")
    print(f"  DQN Training | Scenario: {scenario} | Episodes: {episodes}")
    print(f"  GL1 state_size={GL1_STATE_SIZE} | GL2 state_size={GL2_STATE_SIZE}")
    print(f"{'='*60}")

    for ep in range(episodes):
        s1, s2   = env.reset()
        done     = False
        step_cnt = 0

        total_reward = 0.0
        total_queue  = 0
        losses_g1    = []
        losses_g2    = []

        while not done:
            do_decision = (step_cnt % DECISION_FREQ == 0)

            if do_decision:
                a1 = agent_gl1.get_action(s1)
                a2 = agent_gl2.get_action(s2)
            else:
                a1, a2 = 0, 0

            (ns1, ns2), reward, done, info = env.step(a1, a2)

            if do_decision:
                agent_gl1.remember(s1, a1, info['r1'], ns1, float(done))
                agent_gl2.remember(s2, a2, info['r2'], ns2, float(done))

                loss1 = agent_gl1.train()
                loss2 = agent_gl2.train()
                if loss1 is not None:
                    losses_g1.append(loss1)
                if loss2 is not None:
                    losses_g2.append(loss2)

            s1, s2 = ns1, ns2
            total_reward += reward
            total_queue  += (info['gl1']['total_queue']
                             + info['gl2']['total_queue'])
            step_cnt += 1

        agent_gl1.decay_epsilon()
        agent_gl2.decay_epsilon()

        avg_queue = total_queue / max(step_cnt, 1)
        avg_l1    = np.mean(losses_g1) if losses_g1 else 0.0
        avg_l2    = np.mean(losses_g2) if losses_g2 else 0.0

        ep_rewards.append(total_reward)
        ep_queues.append(avg_queue)
        ep_losses_g1.append(avg_l1)
        ep_losses_g2.append(avg_l2)

        if (ep % EVAL_INTERVAL == 0) or (ep == episodes - 1):
            print(f"  Ep {ep+1:4d}/{episodes} | "
                  f"Reward: {total_reward:10.1f} | "
                  f"AvgQ: {avg_queue:6.2f} | "
                  f"Loss GL1: {avg_l1:.4f} | Loss GL2: {avg_l2:.4f} | "
                  f"eps: {agent_gl1.epsilon:.3f}")

    env.close()

    agent_gl1.save(scenario)
    agent_gl2.save(scenario)
    _plot_training(ep_rewards, ep_queues, ep_losses_g1, ep_losses_g2, scenario)

    return ep_rewards, ep_queues, agent_gl1, agent_gl2


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(scenario: str,
             model_gl1_path: str = None,
             model_gl2_path: str = None,
             use_gui: bool = False):
    """Chạy 1 episode với policy đã train (ε=0)."""

    cfg_path  = SCENARIOS[scenario]
    env       = SumoEnv(cfg_path, use_gui=use_gui)
    agent_gl1 = DQNAgent(state_size=GL1_STATE_SIZE, name='GL1', epsilon=0.0)
    agent_gl2 = DQNAgent(state_size=GL2_STATE_SIZE, name='GL2', epsilon=0.0)

    if model_gl1_path:
        agent_gl1.load(model_gl1_path)
    if model_gl2_path:
        agent_gl2.load(model_gl2_path)

    s1, s2   = env.reset()
    done     = False
    step_cnt = 0
    total_r  = 0.0
    total_q  = 0

    print(f"\n[Evaluate | DQN | {scenario}]")
    print(f"{'Step':>6} {'A_GL1':>6} {'A_GL2':>6} {'Reward':>9} "
          f"{'Q_GL1':>7} {'Q_GL2':>7}")

    while not done:
        a1 = agent_gl1.get_action(s1)
        a2 = agent_gl2.get_action(s2)
        (ns1, ns2), reward, done, info = env.step(a1, a2)

        q_gl1 = info['gl1']['total_queue']
        q_gl2 = info['gl2']['total_queue']
        total_r += reward
        total_q += q_gl1 + q_gl2

        if step_cnt % 100 == 0:
            print(f"  {step_cnt:6d} {a1:6d} {a2:6d} {reward:9.2f} "
                  f"{q_gl1:7d} {q_gl2:7d}")

        s1, s2 = ns1, ns2
        step_cnt += 1

    env.close()
    print(f"\n  Total Reward : {total_r:.1f}")
    print(f"  Avg Queue    : {total_q / max(step_cnt, 1):.2f}")
    return total_r, total_q / max(step_cnt, 1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training(rewards, queues, losses_g1, losses_g2, scenario):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'DQN Training — {scenario}', fontsize=14)

    axes[0, 0].plot(rewards, linewidth=1.2, color='steelblue')
    axes[0, 0].set_title('Total Reward / Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.4)

    win  = min(5, len(queues))
    ma_q = np.convolve(queues, np.ones(win) / win, mode='valid')
    axes[0, 1].plot(queues, alpha=0.35, color='coral', label='Raw')
    axes[0, 1].plot(range(win - 1, len(queues)), ma_q,
                    linewidth=2, color='red', label=f'MA-{win}')
    axes[0, 1].set_title('Avg Queue Length / Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Queue Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.4)

    axes[1, 0].plot(losses_g1, linewidth=1.0, color='green')
    axes[1, 0].set_title('Training Loss — GL1')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Huber Loss')
    axes[1, 0].grid(True, alpha=0.4)

    axes[1, 1].plot(losses_g2, linewidth=1.0, color='purple')
    axes[1, 1].set_title('Training Loss — GL2')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Huber Loss')
    axes[1, 1].grid(True, alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'dqn_training_{scenario}.png')
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DQN Training cho N4TD')
    parser.add_argument('--scenario',  default='low_demand',
                        choices=list(SCENARIOS.keys()))
    parser.add_argument('--episodes',  type=int, default=TOTAL_EPISODES)
    parser.add_argument('--eval',      action='store_true')
    parser.add_argument('--model_gl1', type=str, default=None)
    parser.add_argument('--model_gl2', type=str, default=None)
    parser.add_argument('--gui',       action='store_true')
    args = parser.parse_args()

    if args.eval:
        evaluate(args.scenario, args.model_gl1, args.model_gl2, args.gui)
    else:
        train(args.scenario, args.episodes, args.gui)
