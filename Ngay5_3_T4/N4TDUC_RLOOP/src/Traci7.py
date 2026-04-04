# =============================================================================
# Traci7.py  -  DQN Traffic Control - Nga tu Thu Duc
# Giao lo: clusterJ48_clusterJ44_clusterJ28_J41_clusterJ23_clusterJ25_clusterJ18_J24
# Du lieu xe: 4.209 xe/gio  |  Detector: e2_0..e2_24 (N4TD.add.xml)
# =============================================================================
# Vong lap:
#   1) simulationStep()
#   2) spawn_vehicles()   <- dung findRoute() de SUMO tu tim duong
#   3) get_state()        <- doc detector
#   4) get_action() / apply_action()
#   5) ghi log / train DQN
# =============================================================================
# DANH SACH SUA:
#   [FIX 1] Walrus operator loi tren dong OUTPUT DIR -> viet lai don gian
#   [FIX 2] make_schedule(): pick_vtype() goi 2 lan -> id/type khong khop
#   [FIX 3] try/except traci.start() bi mat except -> SyntaxError
#   [FIX 4] debug_detectors(): kiem tra ID detector thuc te
#   [FIX 5] spawn(): dung findRoute() thay vi route.add([in,out]) de tranh
#           SUMO crash khi 2 canh khong noi truc tiep nhau -> simulation quit
# =============================================================================

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from datetime import datetime
from collections import deque, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

# -- Font tieng Viet
_viet_fonts = ['Arial', 'Tahoma', 'Segoe UI', 'DejaVu Sans', 'Liberation Sans']
_chosen_font = next(
    (f for f in _viet_fonts
     if any(f.lower() in ff.name.lower() for ff in fm.fontManager.ttflist)),
    'DejaVu Sans')
plt.rcParams['font.family'] = _chosen_font
plt.rcParams['axes.unicode_minus'] = False

# -- SUMO tools
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.path.append(r"C:\Program Files (x86)\Eclipse\Sumo\tools")
import traci
import traci.exceptions

# =============================================================================
# DUONG DAN
# =============================================================================
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
CFG_FILE = os.path.join(ROOT_DIR, "data", "network", "N4TD.sumocfg")
LOG_FILE = os.path.join(SRC_DIR, "sumo_output.log")

_SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
_GUI = os.path.join(_SUMO_HOME, "bin", "sumo-gui.exe")
_CMD = os.path.join(_SUMO_HOME, "bin", "sumo.exe")
SUMO_BIN = _GUI if os.path.isfile(_GUI) else (_CMD if os.path.isfile(_CMD) else "sumo-gui")

SUMO_CMD = [
    SUMO_BIN, "-c", CFG_FILE,
    "--step-length",      "1",
    "--no-step-log",      "false",
    "--time-to-teleport", "300",
    "--collision.action", "warn",
    "--seed",             "42",
    "--start",
    # KHONG dung --quit-on-end: tranh SUMO tu quit khi chua co xe
    "--error-log",        LOG_FILE,
]

# =============================================================================
# MANG LUOI
# =============================================================================
TLS_ID     = "clusterJ48_clusterJ44_clusterJ28_J41_clusterJ23_clusterJ25_clusterJ18_J24"
NUM_PHASES = 12

DET_BAC  = ["e2_18", "e2_17", "e2_16"]
DET_DONG = ["e2_3",  "e2_2",  "e2_1",  "e2_0",  "e2_15", "e2_14"]
DET_TAY  = ["e2_10", "e2_9",  "e2_8",  "e2_7",  "e2_6",  "e2_5",  "e2_4"]
DET_NAM  = ["e2_13", "e2_12", "e2_11"]
ALL_DETS = DET_BAC + DET_DONG + DET_TAY + DET_NAM

# =============================================================================
# GIAO THONG
# =============================================================================
TOTAL_VEH_H = 4209
SIM_STEPS   = 3600

IN_EDGES  = ["E31", "E26", "E14", "E15", "E20", "E23"]
OUT_EDGES = ["E12", "E16", "E17", "E24", "E30"]
EDGE_W    = {
    "E31": 0.20, "E26": 0.18, "E14": 0.15,
    "E15": 0.20, "E20": 0.15, "E23": 0.12,
}

VEH_CFG = {
    "motorcycle": {"count": 4000, "ratio": 0.900},
    "car":        {"count":  160, "ratio": 0.036},
    "taxi":       {"count":   15, "ratio": 0.003},
    "bus":        {"count":   14, "ratio": 0.003},
    "coach":      {"count":    9, "ratio": 0.002},
    "truck":      {"count":   11, "ratio": 0.002},
}
VEH_COLOR = {
    "motorcycle": (0,   0,   255, 255),
    "car":        (0,   200,   0, 255),
    "taxi":       (255, 255,   0, 255),
    "bus":        (255, 140,   0, 255),
    "coach":      (128,   0, 128, 255),
    "truck":      (139,  69,  19, 255),
}
VEH_PARAMS = {
    "motorcycle": dict(length=2.0,  width=0.8, maxSpeed=13.89, accel=2.5, decel=4.5, sigma=0.5, minGap=1.0, tau=1.0, vClass="motorcycle"),
    "car":        dict(length=4.5,  width=1.8, maxSpeed=13.89, accel=2.0, decel=4.0, sigma=0.5, minGap=2.5, tau=1.2, vClass="passenger"),
    "taxi":       dict(length=4.5,  width=1.8, maxSpeed=13.89, accel=2.0, decel=4.0, sigma=0.4, minGap=2.5, tau=1.2, vClass="taxi"),
    "bus":        dict(length=12.0, width=2.5, maxSpeed=11.11, accel=1.2, decel=3.0, sigma=0.3, minGap=3.0, tau=1.5, vClass="bus"),
    "coach":      dict(length=14.0, width=2.6, maxSpeed=11.11, accel=1.0, decel=3.0, sigma=0.3, minGap=3.0, tau=1.5, vClass="coach"),
    "truck":      dict(length=8.0,  width=2.4, maxSpeed=11.11, accel=1.0, decel=3.5, sigma=0.4, minGap=3.0, tau=1.5, vClass="truck"),
}
_CUM = []
_c = 0.0
for _vt, _cf in VEH_CFG.items():
    _c += _cf["ratio"]
    _CUM.append((_vt, _c))

# =============================================================================
# RL HYPERPARAMETERS
# =============================================================================
GAMMA        = 0.90
EPSILON      = 0.10
LR           = 0.001
ACTIONS      = [0, 1]
REPLAY_SIZE  = 5000
MIN_REPLAY   = 200
BATCH_SIZE   = 64
MIN_GREEN    = 10
STATE_SIZE   = 5
ACTION_SIZE  = 2
LOG_INTERVAL = 50

# =============================================================================
# [FIX 1] OUTPUT DIR - don gian, khong walrus operator
# =============================================================================
BASE_OUT = os.path.join(ROOT_DIR, "output")
os.makedirs(BASE_OUT, exist_ok=True)

_eps = [
    d for d in os.listdir(BASE_OUT)
    if os.path.isdir(os.path.join(BASE_OUT, d))
    and d.startswith("Ep")
    and d[2:].isdigit()
]
_nxt     = max((int(d[2:]) for d in _eps), default=0) + 1
EP_LABEL = "Ep" + str(_nxt)
OUT_DIR  = os.path.join(BASE_OUT, EP_LABEL)
os.makedirs(OUT_DIR, exist_ok=True)
print("[Thu muc ket qua] " + OUT_DIR)

# =============================================================================
# DQN MODEL
# =============================================================================
def build_model(s, a):
    return nn.Sequential(
        nn.Linear(s, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, a)
    )

def to_tensor(state):
    return torch.tensor(list(state), dtype=torch.float32).unsqueeze(0)

dqn    = build_model(STATE_SIZE, ACTION_SIZE)
opt    = optim.Adam(dqn.parameters(), lr=LR)
lossfn = nn.MSELoss()
mem    = deque(maxlen=REPLAY_SIZE)

# =============================================================================
# HAM PHAT SINH XE
# =============================================================================
def define_vtypes():
    existing = traci.vehicletype.getIDList()
    for vt, p in VEH_PARAMS.items():
        if vt not in existing:
            traci.vehicletype.copy("DEFAULT_VEHTYPE", vt)
            traci.vehicletype.setLength(vt,       p["length"])
            traci.vehicletype.setWidth(vt,        p["width"])
            traci.vehicletype.setMaxSpeed(vt,     p["maxSpeed"])
            traci.vehicletype.setAccel(vt,        p["accel"])
            traci.vehicletype.setDecel(vt,        p["decel"])
            traci.vehicletype.setImperfection(vt, p["sigma"])
            traci.vehicletype.setMinGap(vt,       p["minGap"])
            traci.vehicletype.setTau(vt,          p["tau"])
            traci.vehicletype.setColor(vt,        VEH_COLOR[vt])
    print("[INFO] Vehicle types da dinh nghia.")


def debug_detectors():
    """[FIX 4] In detector thuc te - phat hien ID sai ngay lap tuc."""
    real_dets = set(traci.lanearea.getIDList())
    print("\n[DEBUG] Detectors thuc te trong SUMO (" + str(len(real_dets)) + "):")
    print("  " + str(sorted(real_dets)))
    missing = [d for d in ALL_DETS if d not in real_dets]
    if missing:
        print("[CANH BAO] " + str(len(missing)) + "/" + str(len(ALL_DETS)) +
              " detector KHONG TON TAI -> se doc duoc 0 xe:")
        for d in missing:
            print("  [MISSING] " + d)
        print("  --> Vui long cap nhat DET_BAC/DONG/TAY/NAM cho dung voi N4TD.add.xml")
    else:
        print("[OK] Tat ca " + str(len(ALL_DETS)) + " detector hop le.")
    print()


def debug_edges():
    """In kiem tra canh VAO/RA co ton tai trong network khong."""
    real_edges = set(traci.edge.getIDList())
    ok   = [e for e in IN_EDGES + OUT_EDGES if     e in real_edges]
    bad  = [e for e in IN_EDGES + OUT_EDGES if     e not in real_edges]
    print("[DEBUG] Kiem tra canh VAO/RA:")
    for e in IN_EDGES + OUT_EDGES:
        print("  " + ("OK  " if e in real_edges else "BAD ") + ": " + e)
    if bad:
        print("[CANH BAO] " + str(len(bad)) + " canh KHONG TON TAI: " + str(bad))
        print("  --> Kiem tra ten canh trong N4TD_net.xml!")
    print()


def pick_vtype():
    r = random.random()
    for vt, cum in _CUM:
        if r <= cum:
            return vt
    return "motorcycle"


def pick_in():
    r = random.random()
    c = 0.0
    for e, w in EDGE_W.items():
        c += w
        if r <= c:
            return e
    return IN_EDGES[-1]


def make_schedule():
    """[FIX 2] pick_vtype() goi 1 lan, dung chung cho id va type."""
    sch  = []
    rate = TOTAL_VEH_H / SIM_STEPS
    t    = 0.0
    vid  = 0
    while True:
        t += random.expovariate(rate)
        if t >= SIM_STEPS:
            break
        vtype = pick_vtype()
        sch.append({
            "id":     "v" + str(vid) + "_" + vtype,
            "type":   vtype,
            "depart": int(t),
            "in":     pick_in(),
            "out":    random.choice(OUT_EDGES),
        })
        vid += 1
    print("[INFO] Lich xe: " + str(len(sch)) + " phuong tien.")
    return sch


def build_route_cache():
    """
    [FIX 5] Dung findRoute() de SUMO tu tinh duong day du qua cac canh
    trung gian. Route chi co 2 canh [in, out] se crash neu chung khong
    noi truc tiep - findRoute() giai quyet van de nay.
    Cache tat ca cap (in_edge, out_edge) truoc khi mo phong.
    """
    print("[INFO] Dang xay dung route cache voi findRoute()...")
    rcache  = {}
    n_ok    = 0
    n_fail  = 0

    for in_e in IN_EDGES:
        for out_e in OUT_EDGES:
            if in_e == out_e:
                continue
            key = (in_e, out_e)
            rid = "r_" + in_e + "_" + out_e
            try:
                result = traci.simulation.findRoute(in_e, out_e)
                if result.edges and len(result.edges) >= 1:
                    traci.route.add(rid, list(result.edges))
                    rcache[key] = rid
                    n_ok += 1
                else:
                    n_fail += 1
                    print("  [WARN] Khong tim duoc duong: " + in_e + " -> " + out_e)
            except traci.exceptions.TraCIException as ex:
                n_fail += 1
                print("  [WARN] findRoute loi " + in_e + "->" + out_e + ": " + str(ex))

    total = len(IN_EDGES) * len(OUT_EDGES)
    print("  Route thanh cong: " + str(n_ok) + "/" + str(total))
    if n_fail > 0:
        print("  Route that bai  : " + str(n_fail))
    print()
    return rcache


def spawn(step, dmap, rcache, stats, failed_counts):
    """Them xe vao SUMO theo lich, dung route da build san."""
    for e in dmap.get(step, []):
        key = (e["in"], e["out"])
        rid = rcache.get(key)
        if rid is None:
            failed_counts["no_route"] += 1
            continue
        try:
            traci.vehicle.add(
                vehID       = e["id"],
                routeID     = rid,
                typeID      = e["type"],
                depart      = "now",
                departLane  = "best",
                departSpeed = "desired",
            )
            stats[e["type"]] = stats.get(e["type"], 0) + 1
        except traci.exceptions.TraCIException as ex:
            failed_counts["add_fail"] += 1
            if failed_counts["add_fail"] <= 5:
                print("[WARN spawn] " + e["id"] + ": " + str(ex))

# =============================================================================
# HAM RL
# =============================================================================
def read_det_group(dets):
    total = 0
    for d in dets:
        try:
            total += traci.lanearea.getLastStepVehicleNumber(d)
        except traci.exceptions.TraCIException:
            pass
    return total


def get_state():
    qB = read_det_group(DET_BAC)
    qD = read_det_group(DET_DONG)
    qT = read_det_group(DET_TAY)
    qN = read_det_group(DET_NAM)
    try:
        ph = traci.trafficlight.getPhase(TLS_ID)
    except traci.exceptions.TraCIException:
        ph = 0
    state = (qB / 50.0, qD / 50.0, qT / 50.0, qN / 50.0, ph / float(NUM_PHASES))
    raw   = (qB, qD, qT, qN, ph)
    return state, raw


def get_reward(raw):
    total = raw[0] + raw[1] + raw[2] + raw[3]
    return -float(total)


def choose_action(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    with torch.no_grad():
        return int(torch.argmax(dqn(to_tensor(state))).item())


def do_action(action, step, last_sw):
    if action == 0:
        return last_sw
    if step - last_sw < MIN_GREEN:
        return last_sw
    try:
        prog = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
        nph  = len(prog.phases)
        cur  = traci.trafficlight.getPhase(TLS_ID)
        traci.trafficlight.setPhase(TLS_ID, (cur + 1) % nph)
    except traci.exceptions.TraCIException:
        pass
    return step


def train():
    if len(mem) < MIN_REPLAY:
        return None
    batch = random.sample(mem, BATCH_SIZE)
    cs = torch.tensor([list(t[0]) for t in batch], dtype=torch.float32)
    ac = [t[1] for t in batch]
    rw = [t[2] for t in batch]
    ns = torch.tensor([list(t[3]) for t in batch], dtype=torch.float32)

    cq = dqn(cs)
    with torch.no_grad():
        nq = dqn(ns)

    tgt = cq.clone()
    for i in range(BATCH_SIZE):
        tgt[i][ac[i]] = rw[i] + GAMMA * torch.max(nq[i]).item()

    opt.zero_grad()
    loss = lossfn(cq, tgt.detach())
    loss.backward()
    opt.step()
    return loss.item()

# =============================================================================
# VONG LAP CHINH
# =============================================================================
def run():
    random.seed(42)

    print("=" * 65)
    print("  Traci7.py  -  DQN Dieu Khien Den Giao Thong  |  " + EP_LABEL)
    print("  4209 xe/gio  |  " + str(SIM_STEPS) + " buoc  |  TLS: cluster...J18_J24")
    print("=" * 65)

    if not os.path.isfile(CFG_FILE):
        print("[ERROR] Khong tim thay: " + CFG_FILE)
        sys.exit(1)

    # [FIX 3] try/except day du
    try:
        traci.start(SUMO_CMD)
    except Exception as e:
        print("[FATAL] Khong the ket noi TraCI: " + str(e))
        if os.path.isfile(LOG_FILE):
            print("--- SUMO log ---")
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                print(f.read())
        sys.exit(1)

    print("[OK] Ket noi TraCI thanh cong.\n")

    define_vtypes()
    debug_detectors()  # [FIX 4]
    debug_edges()

    # [FIX 5] Build route cache truoc khi mo phong
    rcache = build_route_cache()

    if not rcache:
        print("[FATAL] Khong co route nao hop le!")
        print("  Kiem tra lai ten canh IN_EDGES / OUT_EDGES trong N4TD_net.xml.")
        traci.close()
        sys.exit(1)

    sch  = make_schedule()
    dmap = defaultdict(list)
    for e in sch:
        dmap[e["depart"]].append(e)

    failed_counts = {"no_route": 0, "add_fail": 0}
    stats   = {}
    last_sw = -MIN_GREEN
    cum_rew = 0.0
    prev_state  = None
    prev_action = None
    prev_reward = None

    sh = []; rh = []; qh = []; lh = []; ah = []

    print("[INFO] Bat dau mo phong " + str(SIM_STEPS) + " buoc...\n")

    for step in range(SIM_STEPS):
        # 1) Tien SUMO 1 buoc
        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError as e:
            print("[FATAL step=" + str(step) + "] SUMO ngat ket noi: " + str(e))
            if os.path.isfile(LOG_FILE):
                print("--- SUMO log ---")
                with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                    print(f.read())
            break

        # 2) Them xe
        spawn(step, dmap, rcache, stats, failed_counts)

        # 3) Doc trang thai
        state, raw = get_state()
        reward     = get_reward(raw)
        cum_rew   += reward

        # 4) Cap nhat replay
        if prev_state is not None:
            mem.append((prev_state, prev_action, prev_reward, state))

        # 5) Train
        loss_val = train()

        # 6) Hanh dong
        action  = choose_action(state)
        last_sw = do_action(action, step, last_sw)
        ah.append(action)

        prev_state  = state
        prev_action = action
        prev_reward = reward

        # 7) Ghi log
        if step % LOG_INTERVAL == 0:
            on_road = traci.vehicle.getIDCount()
            q_total = raw[0] + raw[1] + raw[2] + raw[3]
            with torch.no_grad():
                qv = dqn(to_tensor(state)).numpy()[0]
            print(
                "B{:4d} | Xe:{:4d} | Q B={:2d} D={:2d} T={:2d} N={:2d} Tot={:3d}"
                " | Mem={:4d} | A={} | R={:+.2f} | Cum={:+.1f} | Loss={} | Qv={}".format(
                    step, on_road,
                    raw[0], raw[1], raw[2], raw[3], q_total,
                    len(mem), action, reward, cum_rew,
                    "{:.4f}".format(loss_val) if loss_val is not None else "N/A",
                    np.round(qv, 2)
                )
            )
            sh.append(step)
            rh.append(cum_rew)
            qh.append(q_total)
            lh.append(loss_val if loss_val is not None else 0.0)

    traci.close()
    print("\n[INFO] Da dong ket noi TraCI.")

    print("\n--- Thong ke xe ---")
    for vt, cnt in sorted(stats.items()):
        print("  {:12s}: {:5d}".format(vt, cnt))
    print("  {:12s}: {:5d}".format("TONG", sum(stats.values())))
    print("  No route    : {:5d}".format(failed_counts["no_route"]))
    print("  Add fail    : {:5d}".format(failed_counts["add_fail"]))

    return sh, rh, qh, lh, ah

# =============================================================================
# CHAY
# =============================================================================
sh, rh, qh, lh, ah = run()

mp = os.path.join(OUT_DIR, "dqn_" + EP_LABEL + ".pth")
torch.save(dqn.state_dict(), mp)
print("\n[Model da luu] " + mp)

ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
arr = np.array(ah)

# =============================================================================
# CHI SO DANH GIA
# =============================================================================
q4 = max(1, len(qh) // 4)
qi = ((np.mean(qh[:q4]) - np.mean(qh[-q4:])) / (np.mean(qh[:q4]) + 1e-9) * 100) if qh else 0
rt = np.polyfit(range(len(rh)), rh, 1)[0] if len(rh) > 1 else 0
sr = np.sum(arr == 1) / max(len(arr), 1) * 100
al = np.mean(lh[-10:]) if len(lh) >= 10 else (np.mean(lh) if lh else 0)

verdicts = [
    ("[DAT] "     if qi > 5        else "[XEM XET] ") + "Hang doi: {:+.1f}% (dau vs cuoi)".format(qi),
    ("[DAT] "     if rt > 0        else "[XEM XET] ") + "Xu huong reward: {:.4f}".format(rt),
    ("[DAT] "     if 5 <= sr <= 60 else "[XEM XET] ") + "Ti le chuyen pha: {:.1f}%".format(sr),
    ("[DAT] "     if al < 1.0      else "[XEM XET] ") + "Loss cuoi: {:.4f}".format(al),
]

print("\n=== TU DANH GIA ===")
for v in verdicts:
    print("  " + v)

# =============================================================================
# VE BIEU DO
# =============================================================================
def savefig(fig, name):
    p = os.path.join(OUT_DIR, name + "_" + ts + ".png")
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print("[Da luu] " + p)


if sh:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sh, rh, color='steelblue', linewidth=1.5, label="Phan thuong tich luy")
    if len(sh) > 1:
        z = np.polyfit(sh, rh, 1)
        ax.plot(sh, np.poly1d(z)(sh), '--r', alpha=0.7,
                label="Xu huong ({:+.2f}/buoc)".format(z[0]))
    ax.set_xlabel("Buoc"); ax.set_ylabel("Phan thuong tich luy")
    ax.set_title("DQN Nga tu Thu Duc - Phan thuong | " + EP_LABEL)
    ax.legend(); ax.grid(True)
    savefig(fig, "01_reward"); plt.show()

if sh:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sh, qh, color='darkorange', linewidth=1.5, label="Tong hang doi (4 huong)")
    if len(qh) >= 5:
        ma = np.convolve(qh, np.ones(5) / 5, mode='valid')
        ax.plot(sh[4:], ma, '--k', alpha=0.6, label="TB truot (5)")
    ax.set_xlabel("Buoc"); ax.set_ylabel("So xe trong hang doi")
    ax.set_title("DQN Nga tu Thu Duc - Hang doi | " + EP_LABEL)
    ax.legend(); ax.grid(True)
    savefig(fig, "02_hangdoi"); plt.show()

if sh:
    fig, ax = plt.subplots(figsize=(10, 5))
    sh_arr = np.array(sh); lh_arr = np.array(lh)
    ax.plot(sh, lh, color='lightgray', linewidth=0.8, label="Loss (0=chua train)")
    mask = lh_arr > 0
    if mask.any():
        ax.plot(sh_arr[mask], lh_arr[mask], color='mediumseagreen',
                linewidth=1.5, label="Loss thuc")
    ax.set_xlabel("Buoc"); ax.set_ylabel("MSE Loss")
    ax.set_title("DQN Nga tu Thu Duc - Loss | " + EP_LABEL)
    ax.legend(); ax.grid(True)
    savefig(fig, "03_loss"); plt.show()

if len(ah) > 0:
    fig, ax = plt.subplots(figsize=(6, 5))
    c0 = int(np.sum(arr == 0)); c1 = int(np.sum(arr == 1))
    ax.bar(["Giu pha (0)", "Chuyen pha (1)"], [c0, c1],
           color=['royalblue', 'tomato'], edgecolor='black')
    ax.text(0, c0 + 2, "{}\n({:.1f}%)".format(c0, c0 / max(len(ah), 1) * 100), ha='center')
    ax.text(1, c1 + 2, "{}\n({:.1f}%)".format(c1, c1 / max(len(ah), 1) * 100), ha='center')
    ax.set_ylabel("So buoc"); ax.set_title("Phan phoi hanh dong | " + EP_LABEL)
    ax.grid(axis='y')
    savefig(fig, "04_action"); plt.show()

if sh:
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Bao cao DQN - Nga tu Thu Duc  |  " + EP_LABEL + "  |  " + ts,
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(sh, rh, color='steelblue')
    ax.set_title("Phan thuong tich luy"); ax.set_xlabel("Buoc"); ax.grid(True)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(sh, qh, color='darkorange')
    ax.set_title("Tong hang doi (xe)"); ax.set_xlabel("Buoc"); ax.grid(True)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(sh, lh, color='mediumseagreen')
    ax.set_title("Loss (MSE)"); ax.set_xlabel("Buoc"); ax.grid(True)

    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    info = (
        "=== KET QUA ===\n\n" + "\n".join(verdicts) +
        "\n\n--- Sieu tham so ---" +
        "\n  GAMMA        = " + str(GAMMA) +
        "\n  EPSILON      = " + str(EPSILON) +
        "\n  LR           = " + str(LR) +
        "\n  BATCH_SIZE   = " + str(BATCH_SIZE) +
        "\n  MIN_GREEN    = " + str(MIN_GREEN) + "s" +
        "\n  MIN_REPLAY   = " + str(MIN_REPLAY) +
        "\n  REPLAY_SIZE  = " + str(REPLAY_SIZE) +
        "\n  LOG_INTERVAL = " + str(LOG_INTERVAL) +
        "\n\n--- Giao thong ---" +
        "\n  Tong xe/gio  = " + str(TOTAL_VEH_H) +
        "\n  Buoc         = " + str(SIM_STEPS)
    )
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=8,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    savefig(fig, "05_dashboard"); plt.show()

print("\n[XONG] Tat ca da luu vao: " + OUT_DIR)