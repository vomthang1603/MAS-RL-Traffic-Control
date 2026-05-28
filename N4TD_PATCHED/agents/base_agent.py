# =============================================================================
# base_agent.py — Interface chung cho tất cả RL Agent trong dự án N4TD
#
# Mọi agent (Q-Learning, DQN, ...) đều kế thừa BaseAgent và triển khai
# các phương thức trừu tượng dưới đây.
# =============================================================================

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Interface chung cho RL Agent điều khiển đèn giao thông.

    Mỗi nút đèn (GL1, GL2) sẽ có 1 agent riêng biệt kế thừa lớp này.
    """

    def __init__(self, state_size: int, action_size: int,
                 name: str = 'agent', epsilon: float = 0.1):
        self.state_size  = state_size
        self.action_size = action_size
        self.name        = name
        self.epsilon     = epsilon

    # ------------------------------------------------------------------
    # Chọn action — bắt buộc override
    # ------------------------------------------------------------------

    @abstractmethod
    def get_action(self, state: np.ndarray) -> int:
        """Trả về action index dựa trên state hiện tại."""
        ...

    # ------------------------------------------------------------------
    # Train / cập nhật policy — bắt buộc override
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, *args, **kwargs):
        """Cập nhật policy từ experience."""
        ...

    # ------------------------------------------------------------------
    # Lưu / Tải model — bắt buộc override
    # ------------------------------------------------------------------

    @abstractmethod
    def save(self, scenario: str) -> str:
        """Lưu model/q-table, trả về đường dẫn file."""
        ...

    @abstractmethod
    def load(self, path: str):
        """Tải model/q-table từ đường dẫn."""
        ...

    # ------------------------------------------------------------------
    # Epsilon decay — dùng chung, có thể override
    # ------------------------------------------------------------------

    def decay_epsilon(self, min_eps: float = 0.01, decay: float = 0.995):
        """Giảm epsilon sau mỗi episode (ε-greedy exploration)."""
        self.epsilon = max(min_eps, self.epsilon * decay)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"name={self.name!r}, "
                f"state_size={self.state_size}, "
                f"epsilon={self.epsilon:.3f})")
