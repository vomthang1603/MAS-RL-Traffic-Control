# Step 1: Add modules to provide access to specific libraries and functions
import os  # Module provides functions to handle file paths, directories, environment variables
import sys  # Module provides access to Python-specific system parameters and functions
import random
import numpy as np
import matplotlib.pyplot as plt  # Visualization
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from datetime import datetime

from collections import deque
# Step 1.1: (Additional) Imports for Deep Q-Learning — using PyTorch instead of TensorFlow
import torch
import torch.nn as nn
import torch.optim as optim

# ---- Cấu hình font hỗ trợ tiếng Việt ----
# Ưu tiên các font có sẵn trên Windows hỗ trợ Unicode/tiếng Việt
_viet_fonts = ['Arial', 'Tahoma', 'Segoe UI', 'DejaVu Sans', 'Liberation Sans']
_chosen_font = next(
    (f for f in _viet_fonts
     if any(f.lower() in ff.name.lower() for ff in fm.fontManager.ttflist)),
    'DejaVu Sans'
)
plt.rcParams['font.family'] = _chosen_font
plt.rcParams['axes.unicode_minus'] = False

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci  # Static network information (such as reading and analyzing network files)
CONFIG_PATH = r"C:\MAS-RL-Traffic-Control\Ngay3_31_T3\Learn_Reinforcement Learning Algorithms\config\RL1.sumocfg"

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo-gui',
    '-c', CONFIG_PATH,
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# Step 6: Define Variables
# -------------------------

# Variables for RL State (queue lengths from detectors and current phase)
q_EB_0 = 0
q_EB_1 = 0
q_EB_2 = 0
q_SB_0 = 0
q_SB_1 = 0
q_SB_2 = 0
current_phase = 0

# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 10000    # The total number of simulation steps for continuous (online) training.

GAMMA = 0.9            # Discount factor (γ) between[0, 1]
EPSILON = 0.1          # Exploration rate (ε) between[0, 1]

ACTIONS = [0, 1]       # The discrete action space (0 = keep phase, 1 = switch phase)

# ---- DQN & Experience Replay Hyperparameters ----
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in memory to start training
BATCH_SIZE = 64
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# ---- Output Directory ----
BASE_OUTPUT_DIR = r"C:\MAS-RL-Traffic-Control\Ngay3_31_T3\Learn_Reinforcement Learning Algorithms\output"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Tu dong tim so episode tiep theo (Ep1, Ep2, Ep3, ...)
_existing_eps = [
    d for d in os.listdir(BASE_OUTPUT_DIR)
    if os.path.isdir(os.path.join(BASE_OUTPUT_DIR, d)) and d.startswith("Ep")
    and d[2:].isdigit()
]
_next_ep = max((int(d[2:]) for d in _existing_eps), default=0) + 1
EP_LABEL = f"Ep{_next_ep}"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, EP_LABEL)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[Thu muc ket qua] {OUTPUT_DIR}")

# -------------------------
# Step 7: Define Functions
# -------------------------

def build_model(state_size, action_size):
    """
    Build a simple feedforward neural network that approximates Q-values.
    """
    model = nn.Sequential(
        nn.Linear(state_size, 24),   # First hidden layer
        nn.ReLU(),
        nn.Linear(24, 24),           # Second hidden layer
        nn.ReLU(),
        nn.Linear(24, action_size)   # Output layer (linear activation)
    )
    return model

def to_tensor(state_tuple):
    """
    Convert the state tuple into a PyTorch tensor for neural network input.
    """
    return torch.tensor(state_tuple, dtype=torch.float32).unsqueeze(0)

# Create the DQN model
state_size = 10  # (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, q_WB_0, q_WB_1, q_WB_2, current_phase)
action_size = len(ACTIONS)
dqn_model = build_model(state_size, action_size)
optimizer = optim.Adam(dqn_model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

def get_reward(state): #2. Constraint 2
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    total_queue = sum(state[:-1])  # Exclude the current_phase element
    reward = -float(total_queue)
    return reward

def get_state():  #3&4. Constraint 3 & 4
    global q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase

    # Detector IDs for Node1-2-EB
    detector_Node1_2_EB_0 = "Node1_2_EB_0"
    detector_Node1_2_EB_1 = "Node1_2_EB_1"
    detector_Node1_2_EB_2 = "Node1_2_EB_2"

    # Detector IDs for Node2-7-SB
    detector_Node2_7_SB_0 = "Node2_7_SB_0"
    detector_Node2_7_SB_1 = "Node2_7_SB_1"
    detector_Node2_7_SB_2 = "Node2_7_SB_2"

    # Detector IDs for Node2-3-WB
    detector_Node2_3_WB_0 = "Node2_3_WB_0"
    detector_Node2_3_WB_1 = "Node2_3_WB_1"
    detector_Node2_3_WB_2 = "Node2_3_WB_2"

    # Traffic light ID
    traffic_light_id = "Node2"

    # Get queue lengths from each detector
    q_EB_0 = get_queue_length(detector_Node1_2_EB_0)
    q_EB_1 = get_queue_length(detector_Node1_2_EB_1)
    q_EB_2 = get_queue_length(detector_Node1_2_EB_2)

    q_SB_0 = get_queue_length(detector_Node2_7_SB_0)
    q_SB_1 = get_queue_length(detector_Node2_7_SB_1)
    q_SB_2 = get_queue_length(detector_Node2_7_SB_2)

    q_WB_0 = get_queue_length(detector_Node2_3_WB_0)
    q_WB_1 = get_queue_length(detector_Node2_3_WB_1)
    q_WB_2 = get_queue_length(detector_Node2_3_WB_2)

    # Get current phase index
    current_phase = get_current_phase(traffic_light_id)

    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, q_WB_0, q_WB_1, q_WB_2, current_phase)

def apply_action(action, tls_id="Node2"): #5. Constraint 5
    """
    Executes the chosen action on the traffic light, combining:
      - Min Green Time check
      - Switching to the next phase if allowed
    Constraint #5: Ensure at least MIN_GREEN_STEPS pass before switching again.
    """
    global last_switch_step

    if action == 0:
        # Do nothing (keep current phase)
        return
    elif action == 1:
        # Check if minimum green time has passed before switching
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            # Record when the switch happened
            last_switch_step = current_simulation_step

def update_replay_memory(transition):
    """
    Adds a transition (state, action, reward, new_state) to the replay memory.
    """
    replay_memory.append(transition)

def train_from_replay():
    """
    Trains the DQN agent by sampling from the replay memory.
    Returns the loss value for tracking, or None if not yet training.
    """
    if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
        return None

    minibatch = random.sample(replay_memory, BATCH_SIZE)

    # Get current states and future states from minibatch as tensors
    current_states = torch.tensor([t[0] for t in minibatch], dtype=torch.float32)
    actions        = [t[1] for t in minibatch]
    rewards        = [t[2] for t in minibatch]
    new_states     = torch.tensor([t[3] for t in minibatch], dtype=torch.float32)

    # Query NN for Q values of current and future states
    current_qs_list = dqn_model(current_states)
    with torch.no_grad():
        future_qs_list = dqn_model(new_states)

    # Clone current Q-values as targets, then update only the taken action
    targets = current_qs_list.clone()
    for index in range(BATCH_SIZE):
        # The target Q-value is the immediate reward + discounted max future Q-value
        max_future_q = torch.max(future_qs_list[index]).item()
        new_q = rewards[index] + GAMMA * max_future_q
        targets[index][actions[index]] = new_q

    # Fit the model on the entire batch of updated Q-values
    optimizer.zero_grad()
    loss = loss_fn(current_qs_list, targets.detach())
    loss.backward()
    optimizer.step()

    return loss.item()

def get_action_from_policy(state): #7. Constraint 7
    """
    Epsilon-greedy strategy using the DQN's predicted Q-values.
    """
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        with torch.no_grad():
            Q_values = dqn_model(to_tensor(state))
        return int(torch.argmax(Q_values).item())

def get_queue_length(detector_id): #8.Constraint 8
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id): #8.Constraint 8
    return traci.trafficlight.getPhase(tls_id)

# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

# Lists to record data for plotting
step_history   = []
reward_history = []
queue_history  = []
loss_history   = []
action_history = []  # track every action for distribution analysis

cumulative_reward = 0.0

print("\n=== Bat dau huan luyen truc tuyen lien tuc (DQN) ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step  # keep this variable for apply_action usage

    state = get_state()
    action = get_action_from_policy(state)
    apply_action(action)

    traci.simulationStep()  # Advance simulation by one step

    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward

    update_replay_memory((state, action, reward, new_state))
    loss_val = train_from_replay()

    action_history.append(action)

    # Record data every 100 steps
    if step % 100 == 0:
        with torch.no_grad():
            updated_q_vals = dqn_model(to_tensor(state)).numpy()[0]
        print(f"Buoc {step} | Trang thai: {state} | Hanh dong: {action} | "
              f"Phan thuong: {reward:.2f} | Tich luy: {cumulative_reward:.2f} | "
              f"Gia tri Q: {updated_q_vals}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))  # sum of queue lengths
        loss_history.append(loss_val if loss_val is not None else 0.0)

# -------------------------
# Step 9: Close connection between SUMO and Traci
# -------------------------
traci.close()

print("\nHoan thanh huan luyen.")
print("Cau truc mo hinh DQN:")
print(dqn_model)

# ===================================================
# Step 10: Tu danh gia & Luu bieu do ra thu muc output
# ===================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
action_array = np.array(action_history)

# ---- Tinh cac chi so danh gia ----
quarter = max(1, len(queue_history) // 4)
avg_queue_first   = np.mean(queue_history[:quarter])
avg_queue_last    = np.mean(queue_history[-quarter:])
queue_improvement = ((avg_queue_first - avg_queue_last) / (avg_queue_first + 1e-9)) * 100

reward_trend  = np.polyfit(range(len(reward_history)), reward_history, 1)[0]
switch_rate   = np.sum(action_array == 1) / len(action_array) * 100
avg_loss_last = np.mean(loss_history[-10:]) if len(loss_history) >= 10 else np.mean(loss_history)

# ---- Ket luan tu danh gia (tieng Viet) ----
verdicts = []
if queue_improvement > 5:
    verdicts.append(f"[DAT] Hang doi giam {queue_improvement:.1f}% (dau vs cuoi qua trinh huan luyen)")
else:
    verdicts.append(f"[CAN XEM XET] Hang doi thay doi {queue_improvement:.1f}% - can them buoc huan luyen")

if reward_trend > 0:
    verdicts.append(f"[DAT] Xu huong phan thuong TANG DAN (he so goc={reward_trend:.2f})")
else:
    verdicts.append(f"[CAN XEM XET] Phan thuong PHANG/GIAM (he so goc={reward_trend:.2f})")

if 5 <= switch_rate <= 60:
    verdicts.append(f"[DAT] Ti le chuyen pha {switch_rate:.1f}% - kham pha hop ly")
elif switch_rate < 5:
    verdicts.append(f"[CAN XEM XET] Ti le chuyen pha {switch_rate:.1f}% - agent qua thu dong")
else:
    verdicts.append(f"[CAN XEM XET] Ti le chuyen pha {switch_rate:.1f}% - agent chuyen pha qua nhieu")

if avg_loss_last < 1.0:
    verdicts.append(f"[DAT] Loss da hoi tu (trung binh cuoi={avg_loss_last:.4f})")
else:
    verdicts.append(f"[CAN XEM XET] Loss van cao (trung binh cuoi={avg_loss_last:.4f})")

print("\n=== Ket qua tu danh gia ===")
for v in verdicts:
    print(" ", v)

# -----------------------------------------------
# Bieu do 1: Phan thuong tich luy
# -----------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(step_history, reward_history, marker='o', linestyle='-',
         color='steelblue', label="Phan thuong tich luy")
z = np.polyfit(step_history, reward_history, 1)
ax1.plot(step_history, np.poly1d(z)(step_history), '--', color='red', alpha=0.7,
         label=f"Xu huong (he so goc={z[0]:.2f})")
ax1.set_xlabel("Buoc mo phong")
ax1.set_ylabel("Phan thuong tich luy")
ax1.set_title("Huan luyen DQN: Phan thuong tich luy theo buoc mo phong")
ax1.legend(); ax1.grid(True)
path1 = os.path.join(OUTPUT_DIR, f"phan_thuong_tich_luy_{timestamp}.png")
fig1.savefig(path1, dpi=150, bbox_inches='tight')
print(f"\n[Da luu] {path1}")
plt.show()

# -----------------------------------------------
# Bieu do 2: Do dai hang doi
# -----------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(step_history, queue_history, marker='o', linestyle='-',
         color='darkorange', label="Tong do dai hang doi")
if len(queue_history) >= 5:
    ma = np.convolve(queue_history, np.ones(5)/5, mode='valid')
    ax2.plot(step_history[4:], ma, '--', color='black', alpha=0.7,
             label="Trung binh truot (5 diem)")
ax2.set_xlabel("Buoc mo phong")
ax2.set_ylabel("Tong so xe trong hang doi (xe)")
ax2.set_title("Huan luyen DQN: Do dai hang doi theo buoc mo phong")
ax2.legend(); ax2.grid(True)
path2 = os.path.join(OUTPUT_DIR, f"do_dai_hang_doi_{timestamp}.png")
fig2.savefig(path2, dpi=150, bbox_inches='tight')
print(f"[Da luu] {path2}")
plt.show()

# -----------------------------------------------
# Bieu do 3: Mat mat huan luyen (Loss)
# -----------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(step_history, loss_history, marker='s', linestyle='-',
         color='mediumseagreen', label="Mat mat huan luyen (MSE)")
ax3.set_xlabel("Buoc mo phong")
ax3.set_ylabel("Gia tri mat mat (Loss)")
ax3.set_title("Huan luyen DQN: Mat mat theo buoc mo phong")
ax3.legend(); ax3.grid(True)
path3 = os.path.join(OUTPUT_DIR, f"mat_mat_huan_luyen_{timestamp}.png")
fig3.savefig(path3, dpi=150, bbox_inches='tight')
print(f"[Da luu] {path3}")
plt.show()

# -----------------------------------------------
# Bieu do 4: Phan phoi hanh dong
# -----------------------------------------------
fig4, ax4 = plt.subplots(figsize=(6, 5))
action_counts = [int(np.sum(action_array == 0)), int(np.sum(action_array == 1))]
bars = ax4.bar(
    ["Hanh dong 0\n(Giu nguyen pha)", "Hanh dong 1\n(Chuyen pha)"],
    action_counts,
    color=['royalblue', 'tomato'],
    edgecolor='black'
)
ax4.set_ylabel("So buoc thuc hien")
ax4.set_title("Phan phoi hanh dong trong qua trinh huan luyen")
for i, v in enumerate(action_counts):
    ax4.text(i, v + 10, f"{v}\n({v/TOTAL_STEPS*100:.1f}%)", ha='center', fontsize=10)
ax4.grid(axis='y')
path4 = os.path.join(OUTPUT_DIR, f"phan_phoi_hanh_dong_{timestamp}.png")
fig4.savefig(path4, dpi=150, bbox_inches='tight')
print(f"[Da luu] {path4}")
plt.show()

# -----------------------------------------------
# Bieu do 5: Tong hop (Dashboard)
# -----------------------------------------------
fig5 = plt.figure(figsize=(14, 10))
fig5.suptitle(f"Bao cao tong hop huan luyen DQN  |  {EP_LABEL}  |  {timestamp}", fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig5, hspace=0.45, wspace=0.35)

ax_r = fig5.add_subplot(gs[0, 0])
ax_r.plot(step_history, reward_history, color='steelblue')
ax_r.set_title("Phan thuong tich luy")
ax_r.set_xlabel("Buoc mo phong"); ax_r.grid(True)

ax_q = fig5.add_subplot(gs[0, 1])
ax_q.plot(step_history, queue_history, color='darkorange')
ax_q.set_title("Do dai hang doi")
ax_q.set_xlabel("Buoc mo phong"); ax_q.grid(True)

ax_l = fig5.add_subplot(gs[1, 0])
ax_l.plot(step_history, loss_history, color='mediumseagreen')
ax_l.set_title("Mat mat huan luyen (MSE)")
ax_l.set_xlabel("Buoc mo phong"); ax_l.grid(True)

ax_e = fig5.add_subplot(gs[1, 1])
ax_e.axis('off')
eval_text = (
    "=== KET QUA TU DANH GIA ===\n\n"
    + "\n".join(verdicts)
    + "\n\n--- Sieu tham so ---"
    + f"\n  GAMMA (he so chiet khau) = {GAMMA}"
    + f"\n  EPSILON (ti le kham pha) = {EPSILON}"
    + f"\n  BATCH_SIZE               = {BATCH_SIZE}"
    + f"\n  TONG SO BUOC             = {TOTAL_STEPS}"
    + f"\n  BUOC XANH TOI THIEU      = {MIN_GREEN_STEPS}"
    + f"\n  BO NHO REPLAY            = {REPLAY_MEMORY_SIZE}"
)
ax_e.text(0.02, 0.97, eval_text, transform=ax_e.transAxes,
          fontsize=8.5, verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

path5 = os.path.join(OUTPUT_DIR, f"bao_cao_tong_hop_{timestamp}.png")
fig5.savefig(path5, dpi=150, bbox_inches='tight')
print(f"[Da luu] {path5}")
plt.show()

print(f"\nTat ca bieu do da duoc luu vao: {OUTPUT_DIR}")