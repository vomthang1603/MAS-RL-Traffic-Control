# =============================================================================
# sumo_env.py — SUMO Environment: kết nối TraCI, đọc state, áp dụng action
#               cho cả 2 nút đèn GL1 và GL2
#
# Hàm reward được tách ra environment/reward.py để dễ thay thế/thử nghiệm.
# =============================================================================

import os
import sys
import numpy as np

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("[sumo_env] Lỗi: chưa khai báo biến môi trường SUMO_HOME")

import traci

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    TL_GL1, TL_GL2,
    GL1_GREEN_PHASES, GL2_GREEN_PHASES,
    GL1_DET_IDS, GL2_DET_IDS,
    GL1_STATE_SIZE, GL2_STATE_SIZE,
    STEP_LENGTH, MIN_GREEN_STEPS,
)
from environment.reward import compute_reward


class SumoEnv:
    """
    Môi trường SUMO bọc TraCI cho bài toán điều khiển đèn N4TD.

    Hỗ trợ đồng thời 2 nút đèn:
        GL1 — nút chính (12 pha, 17 detector, state_size=18)
        GL2 — nút phụ  ( 8 pha,  6 detector, state_size=7)

    Interface
    ---------
        env = SumoEnv(cfg_path, use_gui=False)
        s1, s2 = env.reset()
        (s1, s2), reward, done, info = env.step(action_gl1, action_gl2)
        env.close()
    """

    def __init__(self, scenario_cfg: str, use_gui: bool = False):
        self.scenario_cfg     = scenario_cfg
        self.use_gui          = use_gui
        self.sim_step         = 0
        self._last_switch_gl1 = -MIN_GREEN_STEPS
        self._last_switch_gl2 = -MIN_GREEN_STEPS

    # ------------------------------------------------------------------
    # reset / close
    # ------------------------------------------------------------------

    def reset(self):
        """Khởi động lại SUMO, trả về (state_gl1, state_gl2)."""
        if traci.isLoaded():
            traci.close()

        sumo_bin = 'sumo-gui' if self.use_gui else 'sumo'
        cmd = [
            sumo_bin,
            '-c', self.scenario_cfg,
            '--step-length', str(STEP_LENGTH),
            '--no-warnings',          'true',
            '--no-step-log',          'true',
            '--duration-log.disable', 'true',
            '--lateral-resolution',   '0',
        ]
        traci.start(cmd)
        self.sim_step         = 0
        self._last_switch_gl1 = -MIN_GREEN_STEPS
        self._last_switch_gl2 = -MIN_GREEN_STEPS

        return self._get_state_gl1(), self._get_state_gl2()

    def close(self):
        if traci.isLoaded():
            traci.close()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action_gl1: int, action_gl2: int):
        """
        Áp dụng action cho cả 2 nút, chạy 1 bước SUMO.

        Returns
        -------
        (state_gl1, state_gl2) : tuple[np.ndarray]
        reward                 : float  — tổng reward cả 2 nút
        done                   : bool
        info                   : dict   — chi tiết từng nút
        """
        self._apply_action(TL_GL1, action_gl1, GL1_GREEN_PHASES,
                           '_last_switch_gl1')
        self._apply_action(TL_GL2, action_gl2, GL2_GREEN_PHASES,
                           '_last_switch_gl2')

        traci.simulationStep()
        self.sim_step += 1

        s1 = self._get_state_gl1()
        s2 = self._get_state_gl2()

        r1, info1 = compute_reward(GL1_DET_IDS, TL_GL1)
        r2, info2 = compute_reward(GL2_DET_IDS, TL_GL2)
        reward    = r1 + r2

        done = (traci.simulation.getMinExpectedNumber() <= 0
                or self.sim_step >= int(3600 / STEP_LENGTH))

        info = {
            'gl1': info1, 'r1': r1,
            'gl2': info2, 'r2': r2,
            'sim_step': self.sim_step,
        }
        return (s1, s2), reward, done, info

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _get_state_gl1(self) -> np.ndarray:
        queues = self._read_queues(GL1_DET_IDS)
        phase  = traci.trafficlight.getPhase(TL_GL1) / 11.0
        return np.array(queues + [phase], dtype=np.float32)

    def _get_state_gl2(self) -> np.ndarray:
        queues = self._read_queues(GL2_DET_IDS)
        phase  = traci.trafficlight.getPhase(TL_GL2) / 7.0
        return np.array(queues + [phase], dtype=np.float32)

    @staticmethod
    def _read_queues(det_ids: list) -> list:
        return [
            float(traci.lanearea.getLastStepHaltingNumber(did))
            for did in det_ids
        ]

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _apply_action(self, tl_id: str, action: int,
                      green_phases: list, last_switch_attr: str):
        """
        action=0 → giữ nguyên pha
        action=1 → chuyển sang pha xanh kế tiếp (nếu đã qua MIN_GREEN_STEPS)
        """
        if action == 0:
            return
        last = getattr(self, last_switch_attr)
        if self.sim_step - last < MIN_GREEN_STEPS:
            return
        current    = traci.trafficlight.getPhase(tl_id)
        next_green = self._next_green_phase(current, green_phases)
        traci.trafficlight.setPhase(tl_id, next_green)
        setattr(self, last_switch_attr, self.sim_step)

    @staticmethod
    def _next_green_phase(current_phase: int, green_phases: list) -> int:
        for gp in green_phases:
            if gp > current_phase:
                return gp
        return green_phases[0]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_size_gl1(self):
        return GL1_STATE_SIZE

    @property
    def state_size_gl2(self):
        return GL2_STATE_SIZE
