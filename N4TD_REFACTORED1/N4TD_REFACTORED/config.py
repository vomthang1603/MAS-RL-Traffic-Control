# =============================================================================
# config.py — Cấu hình trung tâm cho toàn bộ dự án RL N4TD
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Đường dẫn gốc dự án
# ---------------------------------------------------------------------------
PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
CFG_DIR        = os.path.join(PROJECT_DIR, 'cfg')
NETWORK_DIR    = os.path.join(PROJECT_DIR, 'network')
SCENARIOS_DIR  = os.path.join(PROJECT_DIR, 'scenarios')

# Kết quả & model
PLOTS_DIR      = os.path.join(PROJECT_DIR, 'results', 'plots')
QTABLES_DIR    = os.path.join(PROJECT_DIR, 'models',  'q_tables')
DQN_WEIGHTS_DIR = os.path.join(PROJECT_DIR, 'models', 'dqn_weights')

# Tương thích ngược (một số module cũ vẫn dùng RESULTS_DIR)
RESULTS_DIR = PLOTS_DIR

for _d in (PLOTS_DIR, QTABLES_DIR, DQN_WEIGHTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Kịch bản SUMO
# ---------------------------------------------------------------------------
SCENARIOS = {
    'low_demand'    : os.path.join(CFG_DIR, 'low_demand.sumocfg'),
    'medium_demand' : os.path.join(CFG_DIR, 'medium_demand.sumocfg'),
    'mixed_vehicle' : os.path.join(CFG_DIR, 'mixed_vehicle.sumocfg'),
    'asymmetric'    : os.path.join(CFG_DIR, 'asymmetric.sumocfg'),
    'peak_hour'     : os.path.join(CFG_DIR, 'peak_hour.sumocfg'),
}

# ---------------------------------------------------------------------------
# ID đèn giao thông
# ---------------------------------------------------------------------------
TL_GL1 = 'clusterJ48_clusterJ44_clusterJ28_J41_clusterJ23_clusterJ25_clusterJ18_J24'
TL_GL2 = 'clusterJ49_clusterJ53_J54'

GL1_GREEN_PHASES = [0, 3, 6, 9]
GL2_GREEN_PHASES = [0, 2, 5]

# ---------------------------------------------------------------------------
# Detector e2
# ---------------------------------------------------------------------------
GL1_DETECTORS = {
    'North_E0'  : ['e2_1', 'e2_2', 'e2_3'],
    'West_E20'  : ['e4_1', 'e4_2', 'e4_3'],
    'East_E23'  : ['e2_4', 'e2_5', 'e2_6', 'e2_7'],
    'South_E25' : ['e5_1', 'e5_2', 'e5_3', 'e5_4'],
    'Loop_E31'  : ['e3_1', 'e3_2', 'e3_3'],
}
GL1_DET_IDS    = [d for dets in GL1_DETECTORS.values() for d in dets]
GL1_STATE_SIZE = len(GL1_DET_IDS) + 1   # +1 for current_phase → 18

GL2_DETECTORS = {
    'North_E27' : ['e2_9',  'e2_10', 'e2_11'],
    'South_E29' : ['e2_12', 'e2_13', 'e2_14'],
}
GL2_DET_IDS    = [d for dets in GL2_DETECTORS.values() for d in dets]
GL2_STATE_SIZE = len(GL2_DET_IDS) + 1   # → 7

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
ACTIONS     = [0, 1]   # 0 = giữ pha, 1 = chuyển pha xanh kế tiếp
ACTION_SIZE = len(ACTIONS)

# ---------------------------------------------------------------------------
# Siêu tham số RL chung
# ---------------------------------------------------------------------------
ALPHA   = 0.1
GAMMA   = 0.95
EPSILON = 0.1

# ---------------------------------------------------------------------------
# Siêu tham số DQN
# ---------------------------------------------------------------------------
DQN_HIDDEN_UNITS    = [64, 64]
DQN_LEARNING_RATE   = 0.001
REPLAY_BUFFER_SIZE  = 10_000
BATCH_SIZE          = 64
TARGET_UPDATE_FREQ  = 200

# ---------------------------------------------------------------------------
# Tham số vòng lặp training
# ---------------------------------------------------------------------------
STEP_LENGTH     = 1
DECISION_FREQ   = 10
MIN_GREEN_STEPS = 100
TOTAL_EPISODES  = 50
EVAL_INTERVAL   = 10

# ---------------------------------------------------------------------------
# Q-Learning discretization
# ---------------------------------------------------------------------------
Q_BINS = [0, 5, 15, float('inf')]

def discretize_queue(q: float) -> int:
    for i, threshold in enumerate(Q_BINS):
        if q <= threshold:
            return i
    return len(Q_BINS) - 1
