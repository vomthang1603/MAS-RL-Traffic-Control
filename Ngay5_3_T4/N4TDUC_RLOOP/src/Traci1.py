"""
Traci1.py - Nap du lieu xe vao Nga tu Thu Duc
================================================================
Giao lo: clusterJ48_clusterJ44_clusterJ28_J41_clusterJ23_clusterJ25_clusterJ18_J24

Thanh phan xe (4.209 xe/gio):
  Xe may  : 4.000 (~90.0%)
  O to    :   160 (~ 3.6%)
  Taxi    :    15 (~ 0.3%)
  Xe buyt :    14 (~ 0.3%)
  Xe khach:     9 (~ 0.2%)
  Xe tai  :    11 (~ 0.2%)

Canh VAO: E31(J55) E26(J47) E14(J22) E15(J16) E20(J32) E23(J40)
Canh RA : E12 E16 E17 E24 E30
================================================================
"""

import os
import sys
import random
from collections import defaultdict

# -- Tim SUMO tools
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.path.append(r"C:\Program Files (x86)\Eclipse\Sumo\tools")

import traci
import traci.exceptions

# ============================================================================
# HANG SO
# ============================================================================

JUNCTION_ID = "clusterJ48_clusterJ44_clusterJ28_J41_clusterJ23_clusterJ25_clusterJ18_J24"

INCOMING_EDGES = ["E31", "E26", "E14", "E15", "E20", "E23"]
OUTGOING_EDGES = ["E12", "E16", "E17", "E24", "E30"]

EDGE_WEIGHTS = {
    "E31": 0.20, "E26": 0.18, "E14": 0.15,
    "E15": 0.20, "E20": 0.15, "E23": 0.12,
}

SIM_DURATION       = 3600
TOTAL_VEH_PER_HOUR = 4209

VEHICLE_CONFIG = {
    "motorcycle": {"count": 4000, "ratio": 0.900},
    "car":        {"count":  160, "ratio": 0.036},
    "taxi":       {"count":   15, "ratio": 0.003},
    "bus":        {"count":   14, "ratio": 0.003},
    "coach":      {"count":    9, "ratio": 0.002},
    "truck":      {"count":   11, "ratio": 0.002},
}

VEHICLE_COLORS = {
    "motorcycle": (0,   0,   255, 255),
    "car":        (0,   200,   0, 255),
    "taxi":       (255, 255,   0, 255),
    "bus":        (255, 140,   0, 255),
    "coach":      (128,   0, 128, 255),
    "truck":      (139,  69,  19, 255),
}

VEHICLE_PARAMS = {
    "motorcycle": dict(length=2.0,  width=0.8, maxSpeed=13.89, accel=2.5, decel=4.5, sigma=0.5, minGap=1.0, tau=1.0, vClass="motorcycle"),
    "car":        dict(length=4.5,  width=1.8, maxSpeed=13.89, accel=2.0, decel=4.0, sigma=0.5, minGap=2.5, tau=1.2, vClass="passenger"),
    "taxi":       dict(length=4.5,  width=1.8, maxSpeed=13.89, accel=2.0, decel=4.0, sigma=0.4, minGap=2.5, tau=1.2, vClass="taxi"),
    "bus":        dict(length=12.0, width=2.5, maxSpeed=11.11, accel=1.2, decel=3.0, sigma=0.3, minGap=3.0, tau=1.5, vClass="bus"),
    "coach":      dict(length=14.0, width=2.6, maxSpeed=11.11, accel=1.0, decel=3.0, sigma=0.3, minGap=3.0, tau=1.5, vClass="coach"),
    "truck":      dict(length=8.0,  width=2.4, maxSpeed=11.11, accel=1.0, decel=3.5, sigma=0.4, minGap=3.0, tau=1.5, vClass="truck"),
}

# Bang xac suat tich luy
_CUM_PROBS = []
_c = 0.0
for _vt, _cfg in VEHICLE_CONFIG.items():
    _c += _cfg["ratio"]
    _CUM_PROBS.append((_vt, _c))


# ============================================================================
# HAM TIEN ICH
# ============================================================================

def define_vehicle_types():
    existing = traci.vehicletype.getIDList()
    for vtype, params in VEHICLE_PARAMS.items():
        if vtype not in existing:
            traci.vehicletype.copy("DEFAULT_VEHTYPE", vtype)
            traci.vehicletype.setLength(vtype, params["length"])
            traci.vehicletype.setWidth(vtype, params["width"])
            traci.vehicletype.setMaxSpeed(vtype, params["maxSpeed"])
            traci.vehicletype.setAccel(vtype, params["accel"])
            traci.vehicletype.setDecel(vtype, params["decel"])
            traci.vehicletype.setImperfection(vtype, params["sigma"])
            traci.vehicletype.setMinGap(vtype, params["minGap"])
            traci.vehicletype.setTau(vtype, params["tau"])
            traci.vehicletype.setColor(vtype, VEHICLE_COLORS[vtype])
    print("[INFO] Da dinh nghia xong cac loai phuong tien.")


def pick_vtype():
    r = random.random()
    for vtype, cum in _CUM_PROBS:
        if r <= cum:
            return vtype
    return "motorcycle"


def pick_in_edge():
    r = random.random()
    c = 0.0
    for edge, w in EDGE_WEIGHTS.items():
        c += w
        if r <= c:
            return edge
    return INCOMING_EDGES[-1]


def compute_schedule():
    schedule = []
    rate = TOTAL_VEH_PER_HOUR / SIM_DURATION
    t    = 0.0
    vid  = 0
    while True:
        t += random.expovariate(rate)
        if t >= SIM_DURATION:
            break
        vtype = pick_vtype()
        in_e  = pick_in_edge()
        out_e = random.choice(OUTGOING_EDGES)
        schedule.append({
            "id":       "veh_" + str(vid) + "_" + vtype,
            "type":     vtype,
            "depart":   int(t),
            "in_edge":  in_e,
            "out_edge": out_e,
        })
        vid += 1
    print("[INFO] Da len lich " + str(len(schedule)) + " phuong tien trong " + str(SIM_DURATION) + "s")
    return schedule


# ============================================================================
# HAM CHINH
# ============================================================================

def run(sumo_cmd):
    random.seed(42)

    src_dir  = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(src_dir, "sumo_output.log")
    full_cmd = sumo_cmd + ["--error-log", log_file]

    print("[INFO] Khoi dong SUMO...")
    print("[CMD]  " + " ".join(full_cmd))

    try:
        traci.start(full_cmd)
    except Exception as e:
        print("[FATAL] Khong the ket noi TraCI: " + str(e))
        if os.path.isfile(log_file):
            print("--- SUMO log ---")
            with open(log_file, "r", encoding="utf-8", errors="replace") as lf:
                print(lf.read())
        sys.exit(1)

    print("[INFO] Ket noi TraCI thanh cong.")

    define_vehicle_types()
    schedule = compute_schedule()

    depart_map = defaultdict(list)
    for entry in schedule:
        depart_map[entry["depart"]].append(entry)

    stats   = {vt: 0 for vt in VEHICLE_CONFIG}
    routes  = {}
    failed  = 0
    step    = 0

    print("[INFO] Bat dau mo phong " + str(SIM_DURATION) + " buoc...")

    while step < SIM_DURATION:
        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError as e:
            print("[FATAL] SUMO dong ket noi tai buoc " + str(step) + "s: " + str(e))
            if os.path.isfile(log_file):
                print("--- SUMO log ---")
                with open(log_file, "r", encoding="utf-8", errors="replace") as lf:
                    print(lf.read())
            break

        for entry in depart_map.get(step, []):
            veh_id = entry["id"]
            vtype  = entry["type"]
            in_e   = entry["in_edge"]
            out_e  = entry["out_edge"]
            rkey   = (in_e, out_e)
            rid    = "route_" + in_e + "_" + out_e

            if rkey not in routes:
                try:
                    traci.route.add(rid, [in_e, out_e])
                except traci.exceptions.TraCIException:
                    pass
                routes[rkey] = rid

            try:
                traci.vehicle.add(
                    vehID       = veh_id,
                    routeID     = routes[rkey],
                    typeID      = vtype,
                    depart      = "now",
                    departLane  = "best",
                    departSpeed = "desired",
                )
                stats[vtype] += 1
            except traci.exceptions.TraCIException as e:
                failed += 1
                if failed <= 5:
                    print("[WARN] Khong them duoc xe " + veh_id + ": " + str(e))

        if step > 0 and step % 300 == 0:
            print("[STEP " + str(step) + "s] Phat sinh: " + str(sum(stats.values())) +
                  " xe | Tren duong: " + str(traci.vehicle.getIDCount()))

        step += 1

    # Bao cao
    sep = "=" * 60
    print("")
    print(sep)
    print("  KET QUA MO PHONG - NGA TU THU DUC")
    print(sep)
    print("  Giao lo   : " + JUNCTION_ID)
    print("  Thoi gian : " + str(SIM_DURATION) + "s (1 gio)")
    print("  Tong xe   : " + str(sum(stats.values())))
    print("  Xe loi    : " + str(failed))
    print("-" * 60)
    total_a = max(sum(stats.values()), 1)
    for vtype, cfg in VEHICLE_CONFIG.items():
        pct = stats[vtype] / total_a * 100
        print("  {:<12}  ke hoach={:>5}  thuc te={:>5}  {:>6.1f}%".format(
            vtype, cfg["count"], stats[vtype], pct))
    print(sep)

    traci.close()
    print("[INFO] Da dong ket noi TraCI.")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    src_dir  = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    cfg_file = os.path.join(root_dir, "data", "network", "N4TD.sumocfg")

    print("=" * 60)
    print("  Traci1.py - Nap xe vao Nga tu Thu Duc")
    print("=" * 60)
    print("  root dir    : " + root_dir)
    print("  config file : " + cfg_file)

    if not os.path.isfile(cfg_file):
        print("[ERROR] Khong tim thay: " + cfg_file)
        print("[SCAN]  Tim .sumocfg trong project...")
        found = False
        for dp, _, fnames in os.walk(root_dir):
            for fn in fnames:
                if fn.endswith(".sumocfg"):
                    print("  [FOUND] " + os.path.join(dp, fn))
                    found = True
        if not found:
            print("  Khong tim thay file .sumocfg nao.")
        sys.exit(1)

    print("  Config OK")
    print("  Tong xe/gio : " + str(TOTAL_VEH_PER_HOUR))
    print("=" * 60)

    sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
    gui_bin   = os.path.join(sumo_home, "bin", "sumo-gui.exe")
    cmd_bin   = os.path.join(sumo_home, "bin", "sumo.exe")

    if os.path.isfile(gui_bin):
        binary = gui_bin
    elif os.path.isfile(cmd_bin):
        binary = cmd_bin
    else:
        binary = "sumo-gui"

    print("[INFO] SUMO binary: " + binary)

    sumo_cmd = [
        binary,
        "-c", cfg_file,
        "--step-length", "1",
        "--no-step-log", "false",
        "--time-to-teleport", "300",
        "--collision.action", "warn",
        "--seed", "42",
        "--start",
        "--quit-on-end",
    ]

    run(sumo_cmd)


if __name__ == "__main__":
    main()