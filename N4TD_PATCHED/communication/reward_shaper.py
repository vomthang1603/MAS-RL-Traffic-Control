"""
communication/reward_shaper.py
-------------------------------
Reward shaping cho hệ thống MAS:
    r_shaped = -local_queue - 0.3 × global_queue

Tham chiếu:
    GL1 (12 pha) + GL2 (8 pha) → tổng hợp global_queue từ cả 2 nút.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class NetworkState:
    """Snapshot trạng thái toàn mạng tại một bước thời gian."""
    gl1_queues: Dict[str, float]   # {lane_id: queue_length}
    gl2_queues: Dict[str, float]
    gl1_waiting: float             # avg waiting time GL1
    gl2_waiting: float
    sim_time: float


class RewardShaper:
    """
    Tính shaped reward cho từng agent theo công thức:
        r_shaped = -(α × local_queue) - (β × global_queue) - (γ × waiting_penalty)

    Mặc định:
        α = 1.0   (trọng số queue cục bộ)
        β = 0.3   (trọng số queue toàn mạng)
        γ = 0.1   (phạt thời gian chờ)

    Ý nghĩa:
        - Mỗi agent ưu tiên giảm queue tại nút mình quản lý.
        - Đồng thời có động lực giảm tắc nghẽn trên toàn mạng.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.1,
        queue_norm: float = 30.0,
        waiting_norm: float = 120.0,
        clip: float = 10.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.queue_norm = queue_norm        # chuẩn hoá queue (xe)
        self.waiting_norm = waiting_norm    # chuẩn hoá waiting time (giây)
        self.clip = clip                    # clip reward trong [-clip, 0]

        # Lưu lịch sử để tính delta reward (reward shaping Ng et al.)
        self._prev_state: Optional[NetworkState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        agent_id: str,
        current_state: NetworkState,
        use_delta: bool = False,
    ) -> float:
        """
        Tính reward cho agent_id ("GL1" hoặc "GL2").

        Args:
            agent_id:      "GL1" hoặc "GL2"
            current_state: trạng thái mạng tại bước hiện tại
            use_delta:     nếu True, dùng potential-based shaping (F = γ·Φ' - Φ)

        Returns:
            float: reward (âm, càng gần 0 càng tốt)
        """
        local_q, local_w = self._get_local(agent_id, current_state)
        global_q = self._get_global_queue(current_state)

        r = -(
            self.alpha * (local_q / self.queue_norm)
            + self.beta * (global_q / self.queue_norm)
            + self.gamma * (local_w / self.waiting_norm)
        )

        if use_delta and self._prev_state is not None:
            # Potential-based shaping: F = Φ(s') - Φ(s)
            prev_local_q, _ = self._get_local(agent_id, self._prev_state)
            prev_global_q = self._get_global_queue(self._prev_state)
            phi_prev = -(prev_local_q / self.queue_norm + self.beta * prev_global_q / self.queue_norm)
            phi_curr = -(local_q / self.queue_norm + self.beta * global_q / self.queue_norm)
            r += (phi_curr - phi_prev)

        self._prev_state = current_state
        return float(np.clip(r, -self.clip, 0.0))

    def compute_both(
        self,
        current_state: NetworkState,
        use_delta: bool = False,
    ) -> Dict[str, float]:
        """Tính reward cho cả GL1 và GL2 cùng lúc."""
        r_gl1 = self.compute("GL1", current_state, use_delta)
        # Không update _prev_state lần 2 — compute() đã cập nhật ở GL1
        local_q2, local_w2 = self._get_local("GL2", current_state)
        global_q = self._get_global_queue(current_state)
        r_gl2 = -(
            self.alpha * (local_q2 / self.queue_norm)
            + self.beta * (global_q / self.queue_norm)
            + self.gamma * (local_w2 / self.waiting_norm)
        )
        r_gl2 = float(np.clip(r_gl2, -self.clip, 0.0))
        return {"GL1": r_gl1, "GL2": r_gl2}

    def reset(self) -> None:
        """Gọi khi bắt đầu episode mới."""
        self._prev_state = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _get_local(agent_id: str, state: NetworkState):
        """Trả về (total_queue, avg_waiting) của agent."""
        if agent_id == "GL1":
            q = sum(state.gl1_queues.values())
            w = state.gl1_waiting
        else:
            q = sum(state.gl2_queues.values())
            w = state.gl2_waiting
        return q, w

    @staticmethod
    def _get_global_queue(state: NetworkState) -> float:
        """Tổng queue toàn mạng (GL1 + GL2)."""
        return sum(state.gl1_queues.values()) + sum(state.gl2_queues.values())

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def breakdown(self, agent_id: str, state: NetworkState) -> Dict[str, float]:
        """Trả về phân tích chi tiết các thành phần reward."""
        local_q, local_w = self._get_local(agent_id, state)
        global_q = self._get_global_queue(state)
        return {
            "local_queue_term":   -self.alpha * (local_q / self.queue_norm),
            "global_queue_term":  -self.beta  * (global_q / self.queue_norm),
            "waiting_term":       -self.gamma * (local_w / self.waiting_norm),
            "total":              -(
                self.alpha * local_q / self.queue_norm
                + self.beta * global_q / self.queue_norm
                + self.gamma * local_w / self.waiting_norm
            ),
        }
