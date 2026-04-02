# run_demo.py — Chay mo phong voi model da huan luyen (co GUI)

import os, sys, torch, torch.nn as nn
import traci

# ---- Duong dan ----
SUMO_HOME = os.environ.get('SUMO_HOME', '')
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

CONFIG_PATH = r"C:\MAS-RL-Traffic-Control\Ngay3_31_T3\Learn_Reinforcement Learning Algorithms\config\RL1.sumocfg"

MODEL_PATH  = r"C:\MAS-RL-Traffic-Control\Ngay3_31_T3\Learn_Reinforcement Learning Algorithms\output\TongKet_50Episodes\dqn_model_final.pth"

# ---- Cau hinh SUMO co GUI ----
Sumo_config = [
    'sumo-gui',
    '-c', CONFIG_PATH,
    '--step-length', '0.10',
    '--delay', '50',              # Giam delay xuong de xem nhanh hon
    '--lateral-resolution', '0'
]

# ---- Load model ----
def build_model(state_size, action_size):
    return nn.Sequential(
        nn.Linear(state_size, 64), nn.ReLU(),
        nn.Linear(64, 64),         nn.ReLU(),
        nn.Linear(64, action_size)
    )

model = build_model(state_size=10, action_size=2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print(f"Da load model tu: {MODEL_PATH}")

# ---- Ham tien ich ----
def get_state():
    return (
        traci.lanearea.getLastStepVehicleNumber("Node1_2_EB_0"),
        traci.lanearea.getLastStepVehicleNumber("Node1_2_EB_1"),
        traci.lanearea.getLastStepVehicleNumber("Node1_2_EB_2"),
        traci.lanearea.getLastStepVehicleNumber("Node2_7_SB_0"),
        traci.lanearea.getLastStepVehicleNumber("Node2_7_SB_1"),
        traci.lanearea.getLastStepVehicleNumber("Node2_7_SB_2"),
        traci.lanearea.getLastStepVehicleNumber("Node2_3_WB_0"),
        traci.lanearea.getLastStepVehicleNumber("Node2_3_WB_1"),
        traci.lanearea.getLastStepVehicleNumber("Node2_3_WB_2"),
        traci.trafficlight.getPhase("Node2"),
    )

def get_best_action(state):
    """Luon chon hanh dong tot nhat (khong kham pha — epsilon=0)."""
    t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q = model(t)
    return int(torch.argmax(q).item())

MIN_GREEN_STEPS = 100
TOTAL_STEPS     = 10000

# ---- Chay mo phong ----
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

last_switch_step = -MIN_GREEN_STEPS
print("\n=== Bat dau mo phong Episode 51 (khong kham pha) ===")

for step in range(TOTAL_STEPS):
    state  = get_state()
    action = get_best_action(state)

    if action == 1 and (step - last_switch_step) >= MIN_GREEN_STEPS:
        prog       = traci.trafficlight.getAllProgramLogics("Node2")[0]
        next_phase = (traci.trafficlight.getPhase("Node2") + 1) % len(prog.phases)
        traci.trafficlight.setPhase("Node2", next_phase)
        last_switch_step = step

    traci.simulationStep()

    if step % 500 == 0:
        queue_total = sum(state[:-1])
        print(f"Buoc {step:5d} | Hang doi: {queue_total:3.0f} xe | Pha: {state[-1]}")

traci.close()
print("\nMo phong Episode 51 ket thuc.")