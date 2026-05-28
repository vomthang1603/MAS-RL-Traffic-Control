"""
communication/message_bus.py
----------------------------
Message Bus cho hệ thống MAS: GL2 chia sẻ state/thống kê cho GL1.
GL1 (nút chính, 12 pha) nhận thêm thông tin từ GL2 (nút phụ, 8 pha)
để đưa ra quyết định phối hợp.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AgentMessage:
    """Bản tin trao đổi giữa các agent."""
    sender_id: str                        # "GL1" hoặc "GL2"
    timestamp: float                      # Simulation time (giây)
    queue_lengths: Dict[str, float]       # {lane_id: queue_length}
    avg_waiting_time: float               # Thời gian chờ trung bình (giây)
    current_phase: int                    # Pha đèn hiện tại
    occupancy: Dict[str, float]           # {lane_id: occupancy 0-1}
    extra: Dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """
    Bus trung tâm để các agent giao tiếp theo kiểu publish/subscribe.

    Cách dùng:
        bus = MessageBus(max_history=10)

        # GL2 publish
        bus.publish("GL2", message)

        # GL1 subscribe (lấy tin mới nhất)
        msg = bus.get_latest("GL2")

        # Lấy state vector của GL2 để ghép vào observation của GL1
        extra_state = bus.get_state_vector("GL2")
    """

    def __init__(self, max_history: int = 10):
        self._messages: Dict[str, deque] = {}
        self._max_history = max_history
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = {}   # đếm số tin đã publish

    # ------------------------------------------------------------------
    # Publish / Subscribe
    # ------------------------------------------------------------------

    def publish(self, sender_id: str, message: AgentMessage) -> None:
        """GL2 (hoặc bất kỳ agent nào) đẩy bản tin lên bus."""
        with self._lock:
            if sender_id not in self._messages:
                self._messages[sender_id] = deque(maxlen=self._max_history)
                self._stats[sender_id] = 0
            self._messages[sender_id].append(message)
            self._stats[sender_id] += 1

    def get_latest(self, sender_id: str) -> Optional[AgentMessage]:
        """Lấy bản tin mới nhất từ một agent cụ thể."""
        with self._lock:
            q = self._messages.get(sender_id)
            if q:
                return q[-1]
        return None

    def get_history(self, sender_id: str, n: int = 5):
        """Lấy n bản tin gần nhất (mới nhất ở cuối)."""
        with self._lock:
            q = self._messages.get(sender_id)
            if q:
                return list(q)[-n:]
        return []

    # ------------------------------------------------------------------
    # Helper: chuyển thành vector để ghép vào state của GL1
    # ------------------------------------------------------------------

    def get_state_vector(self, sender_id: str) -> list:
        """
        Trả về vector 7 chiều từ bản tin mới nhất của sender_id.
        Dùng để mở rộng state GL1: 18 → 25.

        Vector:
          [0]   avg_waiting_time (chuẩn hoá / 120)
          [1]   current_phase (chuẩn hoá / max_phase)
          [2-5] queue_lengths tổng hợp (4 hướng, chuẩn hoá / 30)
          [6]   mean occupancy
        """
        msg = self.get_latest(sender_id)
        if msg is None:
            return [0.0] * 7

        # Tổng queue theo 4 hướng (N/S/E/W) — lấy mean nếu nhiều làn
        direction_queues = self._aggregate_queues(msg.queue_lengths, n_dirs=4)

        # Occupancy trung bình
        mean_occ = (
            sum(msg.occupancy.values()) / len(msg.occupancy)
            if msg.occupancy else 0.0
        )

        vector = [
            min(msg.avg_waiting_time / 120.0, 1.0),          # [0]
            msg.current_phase / 8.0,                          # [1] GL2 có 8 pha
            *[min(q / 30.0, 1.0) for q in direction_queues], # [2-5]
            min(mean_occ, 1.0),                               # [6]
        ]
        return vector

    # ------------------------------------------------------------------
    # Stats / Debug
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def reset(self) -> None:
        """Xoá toàn bộ lịch sử (dùng khi reset episode)."""
        with self._lock:
            self._messages.clear()
            self._stats.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_queues(queue_dict: Dict[str, float], n_dirs: int = 4) -> list:
        """
        Gom queue_lengths (key = lane_id) thành n_dirs bucket.
        Giả sử lane_id chứa hướng: 'N', 'S', 'E', 'W'.
        Nếu không có thông tin hướng → chia đều.
        """
        dirs = ['N', 'S', 'E', 'W'][:n_dirs]
        buckets = {d: [] for d in dirs}

        for lane_id, q in queue_dict.items():
            matched = False
            for d in dirs:
                if d in lane_id.upper():
                    buckets[d].append(q)
                    matched = True
                    break
            if not matched:
                # fallback: chia đều vào tất cả bucket
                for d in dirs:
                    buckets[d].append(q / n_dirs)

        result = []
        for d in dirs:
            vals = buckets[d]
            result.append(sum(vals) / len(vals) if vals else 0.0)
        return result
