# =============================================================================
# DQN Traffic Control — 50 Episodes, Khong GUI, Luu bieu do theo tung Episode
# =============================================================================

# Step 1: Import cac thu vien can thiet
import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')                  # Khong hien thi cua so, chi luu file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# ---- Cau hinh font ho tro tieng Viet ----
_viet_fonts = ['Arial', 'Tahoma', 'Segoe UI', 'DejaVu Sans', 'Liberation Sans']
_chosen_font = next(
    (f for f in _viet_fonts
     if any(f.lower() in ff.name.lower() for ff in fm.fontManager.ttflist)),
    'DejaVu Sans'
)
plt.rcParams['font.family'] = _chosen_font
plt.rcParams['axes.unicode_minus'] = False

# Step 2: Khai bao duong dan SUMO
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Vui long khai bao bien moi truong 'SUMO_HOME'")

import traci

# =============================================================================
# Step 3: Cau hinh chung
# =============================================================================

CONFIG_PATH = r"C:\MAS-RL-Traffic-Control\Ngay3_31_T3\Learn_Reinforcement Learning Algorithms\config\RL1.sumocfg"

# ---- Sumo chay khong GUI (sumo thay vi sumo-gui) ----
Sumo_config = [
    'sumo',                        # <-- khong co "-gui"
    '-c', CONFIG_PATH,
    '--step-length', '0.10',
    '--no-warnings',               # Giam thong bao nhieu
    '--lateral-resolution', '0'
    # Khong co --delay vi khong co GUI
]

# ---- Sieu tham so RL ----
TOTAL_EPISODES  = 50       # Tong so episode can huan luyen
TOTAL_STEPS     = 10000    # So buoc mo phong moi episode

GAMMA           = 0.9      # He so chiet khau
EPSILON_START   = 1.0      # Ty le kham pha ban dau (cao de kham pha nhieu luc dau)
EPSILON_END     = 0.05     # Ty le kham pha cuoi cung (thap de khai thac khi da hoc)
EPSILON_DECAY   = 0.95     # He so giam EPSILON sau moi episode

ACTIONS         = [0, 1]   # 0 = giu nguyen pha, 1 = chuyen pha

# ---- DQN & Experience Replay ----
REPLAY_MEMORY_SIZE     = 10000
MIN_REPLAY_MEMORY_SIZE = 1000
BATCH_SIZE             = 64

# ---- Rang buoc pha xanh toi thieu ----
MIN_GREEN_STEPS = 100

# ---- Thu muc luu ket qua ----
BASE_OUTPUT_DIR = r"C:\MAS-RL-Traffic-Control\Ngay3_31_T3\Learn_Reinforcement Learning Algorithms\output"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Step 4: Dinh nghia mo hinh DQN (dung chung xuyen suot 50 episodes)
# =============================================================================

state_size  = 10   # (q_EB x3, q_SB x3, q_WB x3, current_phase)
action_size = len(ACTIONS)

def build_model(state_size, action_size):
    """Xay dung mang neural feedforward don gian de xap xi gia tri Q."""
    return nn.Sequential(
        nn.Linear(state_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_size)
    )

def to_tensor(state_tuple):
    """Chuyen tuple trang thai thanh tensor PyTorch."""
    return torch.tensor(state_tuple, dtype=torch.float32).unsqueeze(0)

# Khoi tao model va cac thanh phan toi uu mot lan, dung xuyen suot cac episode
dqn_model     = build_model(state_size, action_size)
optimizer     = optim.Adam(dqn_model.parameters(), lr=0.001)
loss_fn       = nn.MSELoss()
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

# =============================================================================
# Step 5: Dinh nghia cac ham tien ich
# =============================================================================

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_state():
    """Doc trang thai hien tai tu SUMO: do dai hang doi + pha hien tai."""
    q_EB_0 = get_queue_length("Node1_2_EB_0")
    q_EB_1 = get_queue_length("Node1_2_EB_1")
    q_EB_2 = get_queue_length("Node1_2_EB_2")
    q_SB_0 = get_queue_length("Node2_7_SB_0")
    q_SB_1 = get_queue_length("Node2_7_SB_1")
    q_SB_2 = get_queue_length("Node2_7_SB_2")
    q_WB_0 = get_queue_length("Node2_3_WB_0")
    q_WB_1 = get_queue_length("Node2_3_WB_1")
    q_WB_2 = get_queue_length("Node2_3_WB_2")
    current_phase = get_current_phase("Node2")
    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, q_WB_0, q_WB_1, q_WB_2, current_phase)

def get_reward(state):
    """Phan thuong = am tong hang doi (khuyen khich giam hang doi)."""
    return -float(sum(state[:-1]))

def apply_action(action, current_step, last_switch_step, tls_id="Node2"):
    """Thuc thi hanh dong len den tin hieu giao thong."""
    if action == 0:
        return last_switch_step  # Giu nguyen
    if action == 1 and (current_step - last_switch_step) >= MIN_GREEN_STEPS:
        program   = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        num_phases = len(program.phases)
        next_phase = (get_current_phase(tls_id) + 1) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        return current_step  # cap nhat buoc chuyen pha
    return last_switch_step

def get_action(state, epsilon):
    """Chon hanh dong theo chien luoc epsilon-greedy."""
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    with torch.no_grad():
        q_vals = dqn_model(to_tensor(state))
    return int(torch.argmax(q_vals).item())

def train_step():
    """Huan luyen model tu replay memory. Tra ve loss hoac None."""
    if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
        return None
    minibatch      = random.sample(replay_memory, BATCH_SIZE)
    cur_states     = torch.tensor([t[0] for t in minibatch], dtype=torch.float32)
    actions        = [t[1] for t in minibatch]
    rewards        = [t[2] for t in minibatch]
    new_states     = torch.tensor([t[3] for t in minibatch], dtype=torch.float32)

    cur_qs         = dqn_model(cur_states)
    with torch.no_grad():
        fut_qs     = dqn_model(new_states)

    targets = cur_qs.clone()
    for i in range(BATCH_SIZE):
        targets[i][actions[i]] = rewards[i] + GAMMA * torch.max(fut_qs[i]).item()

    optimizer.zero_grad()
    loss = loss_fn(cur_qs, targets.detach())
    loss.backward()
    optimizer.step()
    return loss.item()

# =============================================================================
# Step 6: Ham luu bieu do cho moi episode
# =============================================================================

def save_episode_charts(ep_label, output_dir, step_history, reward_history,
                        queue_history, loss_history, action_history,
                        epsilon, verdicts):
    """Ve va luu 5 bieu do phan tich cho mot episode."""
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    action_array = np.array(action_history)

    # --- Bieu do 1: Phan thuong tich luy ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(step_history, reward_history, marker='o', ms=3,
             linestyle='-', color='steelblue', label="Phan thuong tich luy")
    z = np.polyfit(step_history, reward_history, 1)
    ax1.plot(step_history, np.poly1d(z)(step_history), '--', color='red',
             alpha=0.7, label=f"Xu huong (he so goc={z[0]:.2f})")
    ax1.set_xlabel("Buoc mo phong")
    ax1.set_ylabel("Phan thuong tich luy")
    ax1.set_title(f"[{ep_label}] Phan thuong tich luy theo buoc mo phong")
    ax1.legend(); ax1.grid(True)
    fig1.savefig(os.path.join(output_dir, f"1_phan_thuong_tich_luy.png"),
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # --- Bieu do 2: Do dai hang doi ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(step_history, queue_history, marker='o', ms=3,
             linestyle='-', color='darkorange', label="Tong do dai hang doi")
    if len(queue_history) >= 5:
        ma = np.convolve(queue_history, np.ones(5)/5, mode='valid')
        ax2.plot(step_history[4:], ma, '--', color='black', alpha=0.7,
                 label="Trung binh truot (5 diem)")
    ax2.set_xlabel("Buoc mo phong")
    ax2.set_ylabel("Tong so xe trong hang doi (xe)")
    ax2.set_title(f"[{ep_label}] Do dai hang doi theo buoc mo phong")
    ax2.legend(); ax2.grid(True)
    fig2.savefig(os.path.join(output_dir, f"2_do_dai_hang_doi.png"),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # --- Bieu do 3: Mat mat huan luyen ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(step_history, loss_history, marker='s', ms=3,
             linestyle='-', color='mediumseagreen', label="Mat mat huan luyen (MSE)")
    ax3.set_xlabel("Buoc mo phong")
    ax3.set_ylabel("Gia tri mat mat (Loss)")
    ax3.set_title(f"[{ep_label}] Mat mat huan luyen theo buoc mo phong")
    ax3.legend(); ax3.grid(True)
    fig3.savefig(os.path.join(output_dir, f"3_mat_mat_huan_luyen.png"),
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # --- Bieu do 4: Phan phoi hanh dong ---
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    action_counts = [int(np.sum(action_array == 0)), int(np.sum(action_array == 1))]
    ax4.bar(["Hanh dong 0\n(Giu nguyen pha)", "Hanh dong 1\n(Chuyen pha)"],
            action_counts, color=['royalblue', 'tomato'], edgecolor='black')
    ax4.set_ylabel("So buoc thuc hien")
    ax4.set_title(f"[{ep_label}] Phan phoi hanh dong")
    for i, v in enumerate(action_counts):
        ax4.text(i, v + 5, f"{v}\n({v/TOTAL_STEPS*100:.1f}%)", ha='center', fontsize=10)
    ax4.grid(axis='y')
    fig4.savefig(os.path.join(output_dir, f"4_phan_phoi_hanh_dong.png"),
                 dpi=150, bbox_inches='tight')
    plt.close(fig4)

    # --- Bieu do 5: Dashboard tong hop ---
    fig5 = plt.figure(figsize=(14, 10))
    fig5.suptitle(f"Bao cao tong hop  |  {ep_label}  |  {timestamp}",
                  fontsize=13, fontweight='bold')
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
        f"=== KET QUA TU DANH GIA ({ep_label}) ===\n\n"
        + "\n".join(verdicts)
        + "\n\n--- Sieu tham so ---"
        + f"\n  GAMMA            = {GAMMA}"
        + f"\n  EPSILON hien tai = {epsilon:.4f}"
        + f"\n  BATCH_SIZE       = {BATCH_SIZE}"
        + f"\n  TONG SO BUOC     = {TOTAL_STEPS}"
        + f"\n  BUOC XANH TOI THIEU = {MIN_GREEN_STEPS}"
        + f"\n  BO NHO REPLAY    = {REPLAY_MEMORY_SIZE}"
    )
    ax_e.text(0.02, 0.97, eval_text, transform=ax_e.transAxes,
              fontsize=8.5, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    fig5.savefig(os.path.join(output_dir, f"5_bao_cao_tong_hop.png"),
                 dpi=150, bbox_inches='tight')
    plt.close(fig5)

def compute_verdicts(queue_history, reward_history, action_history, loss_history):
    """Tinh cac chi so va tra ve danh gia van ban."""
    quarter           = max(1, len(queue_history) // 4)
    avg_q_first       = np.mean(queue_history[:quarter])
    avg_q_last        = np.mean(queue_history[-quarter:])
    queue_improvement = ((avg_q_first - avg_q_last) / (avg_q_first + 1e-9)) * 100
    reward_trend      = np.polyfit(range(len(reward_history)), reward_history, 1)[0]
    action_array      = np.array(action_history)
    switch_rate       = np.sum(action_array == 1) / len(action_array) * 100
    avg_loss          = np.mean(loss_history[-10:]) if len(loss_history) >= 10 else np.mean(loss_history) if loss_history else 0.0

    verdicts = []
    verdicts.append(
        f"[DAT] Hang doi giam {queue_improvement:.1f}%"
        if queue_improvement > 5 else
        f"[XEM XET] Hang doi thay doi {queue_improvement:.1f}%"
    )
    verdicts.append(
        f"[DAT] Xu huong phan thuong TANG (goc={reward_trend:.2f})"
        if reward_trend > 0 else
        f"[XEM XET] Phan thuong PHANG/GIAM (goc={reward_trend:.2f})"
    )
    verdicts.append(
        f"[DAT] Ti le chuyen pha {switch_rate:.1f}% - hop ly"
        if 5 <= switch_rate <= 60 else
        (f"[XEM XET] Chuyen pha {switch_rate:.1f}% - qua thu dong" if switch_rate < 5
         else f"[XEM XET] Chuyen pha {switch_rate:.1f}% - qua nhieu")
    )
    verdicts.append(
        f"[DAT] Loss da hoi tu (avg={avg_loss:.4f})"
        if avg_loss < 1.0 else
        f"[XEM XET] Loss van cao (avg={avg_loss:.4f})"
    )
    return verdicts

# =============================================================================
# Step 7: Vong lap huan luyen 50 Episodes
# =============================================================================

epsilon          = EPSILON_START
all_ep_rewards   = []   # Theo doi tong phan thuong moi episode (ve bieu do toan cuc)
all_ep_queues    = []   # Theo doi trung binh hang doi moi episode

print("=" * 60)
print(f"  BAT DAU HUAN LUYEN DQN — {TOTAL_EPISODES} EPISODES")
print(f"  Moi episode: {TOTAL_STEPS} buoc | Khong GUI")
print("=" * 60)

for ep in range(1, TOTAL_EPISODES + 1):

    # ---- Tao thu muc Ep<x> ----
    ep_label   = f"Ep{ep}"
    output_dir = os.path.join(BASE_OUTPUT_DIR, ep_label)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Khoi dong SUMO (khong GUI) ----
    traci.start(Sumo_config)

    # ---- Reset bien theo doi trong episode ----
    step_history   = []
    reward_history = []
    queue_history  = []
    loss_history   = []
    action_history = []
    cumulative_reward  = 0.0
    last_switch_step   = -MIN_GREEN_STEPS

    print(f"\n[{ep_label}/{TOTAL_EPISODES}] Bat dau | Epsilon={epsilon:.4f}")

    for step in range(TOTAL_STEPS):
        state  = get_state()
        action = get_action(state, epsilon)
        last_switch_step = apply_action(action, step, last_switch_step)

        traci.simulationStep()

        new_state = get_state()
        reward    = get_reward(new_state)
        cumulative_reward += reward

        replay_memory.append((state, action, reward, new_state))
        loss_val = train_step()
        action_history.append(action)

        # Ghi lai moi 100 buoc
        if step % 100 == 0:
            step_history.append(step)
            reward_history.append(cumulative_reward)
            queue_history.append(sum(new_state[:-1]))
            loss_history.append(loss_val if loss_val is not None else 0.0)

    # ---- Dong SUMO sau moi episode ----
    traci.close()

    # ---- Giam Epsilon sau moi episode ----
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # ---- Ghi nhan ket qua toan cuc ----
    all_ep_rewards.append(cumulative_reward)
    all_ep_queues.append(np.mean(queue_history) if queue_history else 0.0)

    print(f"[{ep_label}] Xong | Tong phan thuong: {cumulative_reward:.1f} | "
          f"TB hang doi: {all_ep_queues[-1]:.2f} | Epsilon moi: {epsilon:.4f}")

    # ---- Luu bieu do cho episode nay ----
    verdicts = compute_verdicts(queue_history, reward_history, action_history, loss_history)
    save_episode_charts(
        ep_label, output_dir,
        step_history, reward_history, queue_history, loss_history, action_history,
        epsilon, verdicts
    )
    print(f"[{ep_label}] Bieu do da luu vao: {output_dir}")

# =============================================================================
# Step 8: Luu bieu do tong ket toan bo 50 episodes
# =============================================================================

summary_dir = os.path.join(BASE_OUTPUT_DIR, "TongKet_50Episodes")
os.makedirs(summary_dir, exist_ok=True)

ep_labels = [f"Ep{i}" for i in range(1, TOTAL_EPISODES + 1)]
ep_nums   = list(range(1, TOTAL_EPISODES + 1))

# --- Bieu do tong ket 1: Phan thuong tich luy theo episode ---
fig_s1, ax_s1 = plt.subplots(figsize=(12, 5))
ax_s1.plot(ep_nums, all_ep_rewards, marker='o', ms=4,
           color='steelblue', label="Tong phan thuong moi episode")
z = np.polyfit(ep_nums, all_ep_rewards, 1)
ax_s1.plot(ep_nums, np.poly1d(z)(ep_nums), '--', color='red', alpha=0.7,
           label=f"Xu huong (goc={z[0]:.2f})")
ax_s1.set_xlabel("Episode")
ax_s1.set_ylabel("Tong phan thuong tich luy")
ax_s1.set_title(f"Tong ket {TOTAL_EPISODES} Episodes: Phan thuong tich luy")
ax_s1.set_xticks(ep_nums[::5]); ax_s1.legend(); ax_s1.grid(True)
fig_s1.savefig(os.path.join(summary_dir, "tong_ket_phan_thuong.png"),
               dpi=150, bbox_inches='tight')
plt.close(fig_s1)

# --- Bieu do tong ket 2: Trung binh hang doi theo episode ---
fig_s2, ax_s2 = plt.subplots(figsize=(12, 5))
ax_s2.plot(ep_nums, all_ep_queues, marker='s', ms=4,
           color='darkorange', label="TB hang doi moi episode")
z2 = np.polyfit(ep_nums, all_ep_queues, 1)
ax_s2.plot(ep_nums, np.poly1d(z2)(ep_nums), '--', color='red', alpha=0.7,
           label=f"Xu huong (goc={z2[0]:.2f})")
ax_s2.set_xlabel("Episode")
ax_s2.set_ylabel("Trung binh tong hang doi (xe)")
ax_s2.set_title(f"Tong ket {TOTAL_EPISODES} Episodes: Trung binh hang doi")
ax_s2.set_xticks(ep_nums[::5]); ax_s2.legend(); ax_s2.grid(True)
fig_s2.savefig(os.path.join(summary_dir, "tong_ket_hang_doi.png"),
               dpi=150, bbox_inches='tight')
plt.close(fig_s2)

# --- Dashboard tong ket ---
fig_s3 = plt.figure(figsize=(14, 6))
fig_s3.suptitle(f"Dashboard tong ket huan luyen DQN — {TOTAL_EPISODES} Episodes",
                fontsize=13, fontweight='bold')
gs3 = gridspec.GridSpec(1, 2, figure=fig_s3, wspace=0.3)

ax3_l = fig_s3.add_subplot(gs3[0])
ax3_l.plot(ep_nums, all_ep_rewards, color='steelblue', marker='o', ms=3)
ax3_l.set_title("Phan thuong tich luy theo Episode")
ax3_l.set_xlabel("Episode"); ax3_l.grid(True)

ax3_r = fig_s3.add_subplot(gs3[1])
ax3_r.plot(ep_nums, all_ep_queues, color='darkorange', marker='s', ms=3)
ax3_r.set_title("Trung binh hang doi theo Episode")
ax3_r.set_xlabel("Episode"); ax3_r.grid(True)

fig_s3.savefig(os.path.join(summary_dir, "dashboard_tong_ket.png"),
               dpi=150, bbox_inches='tight')
plt.close(fig_s3)

# Luu model sau khi huan luyen xong
model_path = os.path.join(summary_dir, "dqn_model_final.pth")
torch.save(dqn_model.state_dict(), model_path)

print("\n" + "=" * 60)
print(f"  HUAN LUYEN HOAN THANH — {TOTAL_EPISODES} EPISODES")
print(f"  Bieu do tong ket: {summary_dir}")
print(f"  Model da luu:     {model_path}")
print("=" * 60)