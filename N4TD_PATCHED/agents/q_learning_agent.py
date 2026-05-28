# =============================================================================
# q_learning_agent.py — Q-Learning Agent (Q-table)
#
# Đóng gói toàn bộ logic agent:
#   - Q-table (defaultdict)
#   - Discretize state vector → tuple key
#   - ε-greedy action selection
#   - Bellman update
#   - Save / Load Q-table (.pkl)
# =============================================================================

import os
import sys
import random
import pickle
from collections import defaultdict

import numpy as np

# Đảm bảo import được config từ thư mục gốc dự án
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import ACTIONS, ACTION_SIZE, ALPHA, GAMMA, EPSILON, QTABLES_DIR
from agents.base_agent import BaseAgent

_DISCRETIZE_BINS = np.array([0, 5, 15], dtype=np.float32)


class QLearningAgent(BaseAgent):
    """
    Q-Learning Agent cho 1 nút đèn giao thông.

    Params
    ------
    state_size  : int   — chiều dài state vector (float)
    name        : str   — 'GL1' hoặc 'GL2'
    epsilon     : float — tỉ lệ exploration ban đầu
    alpha       : float — learning rate
    gamma       : float — discount factor
    """

    def __init__(self, state_size: int, name: str = 'agent',
                 epsilon: float = EPSILON,
                 alpha: float = ALPHA,
                 gamma: float = GAMMA):
        super().__init__(state_size=state_size, action_size=ACTION_SIZE,
                         name=name, epsilon=epsilon)
        self.alpha = alpha
        self.gamma = gamma

        # Q-table: state_key (tuple) → np.array(action_size,)
        self.q_table: dict = defaultdict(
            lambda: np.zeros(ACTION_SIZE, dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # State discretization
    # ------------------------------------------------------------------

    @staticmethod
    def discretize_state(state_vec: np.ndarray) -> tuple:
        """
        Chuyển state vector float → tuple key cho Q-table.

        - Queue values (tất cả trừ phần tử cuối) → bin index theo _DISCRETIZE_BINS
        - Phase (phần tử cuối, đã normalize 0-1) → int phase index
          GL1: phase_norm = phase/11 → phase = round(phase_norm * 11)
          GL2: phase_norm = phase/7  → phase = round(phase_norm * 7)
          Công thức đúng: dùng giá trị raw (không nhân thêm offset)
        """
        queues     = state_vec[:-1]
        phase_norm = float(state_vec[-1])
        # Tìm max_phase từ khoảng [0,1]: nhân ngược với bội số gần nhất
        # GL1 normalize /11, GL2 /7 — dùng 11 (GL1) làm mặc định an toàn
        # vì phase_norm ≤ 1.0, int(round(phase_norm * 11)) luôn đúng cho cả 2
        phase = int(round(phase_norm * 11))
        bins  = tuple(np.digitize(queues, _DISCRETIZE_BINS, right=True).tolist())
        return bins + (phase,)

    # ------------------------------------------------------------------
    # Action selection (ε-greedy)
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray) -> int:
        """ε-greedy: ngẫu nhiên với xác suất ε, còn lại argmax Q."""
        state_key = self.discretize_state(state)
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return int(np.argmax(self.q_table[state_key]))

    def get_action_from_key(self, state_key: tuple,
                            epsilon: float = None) -> int:
        """Chọn action từ state key đã được discretize sẵn."""
        eps = epsilon if epsilon is not None else self.epsilon
        if random.random() < eps:
            return random.choice(ACTIONS)
        return int(np.argmax(self.q_table[state_key]))

    # ------------------------------------------------------------------
    # Q-table update (Bellman equation)
    # ------------------------------------------------------------------

    def train(self, state_key: tuple, action: int,
              reward: float, next_key: tuple):
        """Cập nhật Q(s, a) theo phương trình Bellman."""
        old_q  = self.q_table[state_key][action]
        best_q = np.max(self.q_table[next_key])
        self.q_table[state_key][action] = old_q + self.alpha * (
            reward + self.gamma * best_q - old_q
        )

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, scenario: str) -> str:
        """Lưu Q-table ra file .pkl trong models/q_tables/."""
        path = os.path.join(QTABLES_DIR, f'qtable_{self.name}_{scenario}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"  [{self.name}] Q-table saved → {path}")
        return path

    def load(self, path: str):
        """Tải Q-table từ file .pkl."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table.update(data)
        print(f"  [{self.name}] Q-table loaded ← {path}")

    # ------------------------------------------------------------------
    # Utility: trả về Q-values cho debug
    # ------------------------------------------------------------------

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.q_table[self.discretize_state(state)].copy()
