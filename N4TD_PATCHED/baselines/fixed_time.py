"""
baselines/fixed_time.py
------------------------
Baseline Fixed-Time Traffic Light Controller.

Không học, không thích nghi — dùng chu kỳ đèn cố định
để làm baseline so sánh với Independent RL và Cooperative MAS.

Cấu hình mặc định phù hợp với ngã 4 Thủ Đức (N4TD):
  - GL1 (nút chính): 12 pha, chu kỳ 90s
  - GL2 (nút phụ):   8 pha,  chu kỳ 60s
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PhaseConfig:
    """Cấu hình một pha đèn."""
    phase_id: int
    green_duration: float   # giây
    yellow_duration: float = 3.0
    min_green: float = 5.0
    max_green: float = 60.0


@dataclass
class FixedTimePlan:
    """Kế hoạch chu kỳ cố định cho một nút giao."""
    intersection_id: str
    phases: List[PhaseConfig]
    offset: float = 0.0     # độ lệch pha so với đồng hồ toàn mạng (giây)

    @property
    def cycle_length(self) -> float:
        return sum(p.green_duration + p.yellow_duration for p in self.phases)


class FixedTimeController:
    """
    Điều khiển đèn theo chu kỳ cố định cho một nút giao.

    Cách dùng trong vòng lặp simulation:
        ctrl = FixedTimeController(plan)
        for step in range(3600):
            action = ctrl.get_action(sim_time=step)
            env.set_phase(action)
    """

    def __init__(self, plan: FixedTimePlan):
        self.plan = plan
        self._phase_boundaries = self._compute_boundaries()

    def _compute_boundaries(self) -> List[float]:
        """Tính thời điểm chuyển pha trong một chu kỳ."""
        boundaries = []
        t = self.plan.offset
        for phase in self.plan.phases:
            t += phase.green_duration + phase.yellow_duration
            boundaries.append(t % self.plan.cycle_length)
        return boundaries

    def get_action(self, sim_time: float) -> int:
        """
        Trả về index pha hiện tại dựa vào thời gian simulation.

        Args:
            sim_time: thời gian tuyệt đối (giây) từ đầu simulation

        Returns:
            int: phase index (0-based)
        """
        cycle_time = (sim_time - self.plan.offset) % self.plan.cycle_length
        elapsed = 0.0
        for i, phase in enumerate(self.plan.phases):
            elapsed += phase.green_duration + phase.yellow_duration
            if cycle_time < elapsed:
                return i
        return len(self.plan.phases) - 1

    def get_phase_remaining(self, sim_time: float) -> float:
        """Thời gian còn lại của pha hiện tại (giây)."""
        phase_idx = self.get_action(sim_time)
        cycle_time = (sim_time - self.plan.offset) % self.plan.cycle_length
        elapsed = 0.0
        for i, phase in enumerate(self.plan.phases):
            duration = phase.green_duration + phase.yellow_duration
            if i == phase_idx:
                return elapsed + duration - cycle_time
            elapsed += duration
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình mặc định N4TD (Ngã 4 Thủ Đức)
# ──────────────────────────────────────────────────────────────────────────────

def make_gl1_fixed_plan() -> FixedTimePlan:
    """
    GL1 — Nút chính, 12 pha, chu kỳ ~90s.
    Green duration chia đều, có thể chỉnh theo volume thực tế.
    """
    phases = [
        PhaseConfig(phase_id=i, green_duration=6.0, yellow_duration=3.0)
        for i in range(12)
    ]
    # Điều chỉnh 4 pha trục chính dài hơn (pha 0,3,6,9)
    for i in [0, 3, 6, 9]:
        phases[i].green_duration = 9.0
    return FixedTimePlan(intersection_id="GL1", phases=phases, offset=0.0)


def make_gl2_fixed_plan() -> FixedTimePlan:
    """
    GL2 — Nút phụ, 8 pha, chu kỳ ~60s.
    Offset 15s để tránh xung đột với GL1.
    """
    phases = [
        PhaseConfig(phase_id=i, green_duration=6.0, yellow_duration=3.0)
        for i in range(8)
    ]
    # Pha trục chính dài hơn
    for i in [0, 4]:
        phases[i].green_duration = 9.0
    return FixedTimePlan(intersection_id="GL2", phases=phases, offset=15.0)


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper dùng như "agent" để tích hợp vào vòng lặp train/eval
# ──────────────────────────────────────────────────────────────────────────────

class FixedTimeAgent:
    """
    Wrapper để FixedTimeController có interface giống CooperativeAgent,
    dễ dàng so sánh trong vòng lặp evaluation.

    Dùng được với train_mas.py thay thế cho CooperativeAgent:
        agent = FixedTimeAgent("GL1")
        action = agent.act(state, sim_time=t)
    """

    def __init__(self, agent_id: str, plan: Optional[FixedTimePlan] = None):
        self.agent_id = agent_id
        if plan is None:
            plan = make_gl1_fixed_plan() if agent_id == "GL1" else make_gl2_fixed_plan()
        self.controller = FixedTimeController(plan)

    def act(self, state: Optional[object] = None, sim_time: float = 0.0) -> int:
        """Interface giống CooperativeAgent.act() — bỏ qua state."""
        return self.controller.get_action(sim_time)

    def remember(self, *args, **kwargs):
        """No-op — fixed-time không học."""
        pass

    def learn(self):
        """No-op."""
        return None

    @property
    def epsilon(self):
        return 0.0   # không có exploration

    def __repr__(self):
        return (
            f"FixedTimeAgent(id={self.agent_id}, "
            f"cycle={self.controller.plan.cycle_length:.1f}s)"
        )
