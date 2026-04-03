"""
traci1.py – Mô phỏng giao thông N4TD_RL với xe ô tô và xe máy
Routes được trích xuất trực tiếp từ N4TD_RL_net.xml (hợp lệ 100%)
"""

import os
import sys
import random

# ──────────────────────────────────────────────
# SUMO PATH
# ──────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("[ERROR] Chưa khai báo biến môi trường SUMO_HOME")

import traci  # noqa: E402

# ──────────────────────────────────────────────
# ĐƯỜNG DẪN FILE
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(BASE_DIR, "network", "N4TD_RL.net.xml")

if not os.path.isfile(NET_FILE):
    sys.exit(
        f"[ERROR] Không tìm thấy: {NET_FILE}\n"
        "Đặt traci1.py trong thư mục N4TD_RL/ (cùng cấp với network/)"
    )

# ──────────────────────────────────────────────
# THAM SỐ MÔ PHỎNG
# ──────────────────────────────────────────────
SIM_STEPS       = 3600   # số bước (giây)
STEP_LENGTH     = 1.0    # độ dài mỗi bước
GUI             = True   # True = sumo-gui, False = sumo (nền)
CAR_SPAWN_PROB  = 0.20   # xác suất sinh ô tô mỗi bước
MOTO_SPAWN_PROB = 0.35   # xác suất sinh xe máy mỗi bước

# ──────────────────────────────────────────────
# ROUTES HỢP LỆ – trích từ N4TD_RL_net.xml
# ──────────────────────────────────────────────
CAR_ROUTES = {
    "car_r1":  ["E85",   "E86",   "-E235", "-E358"],
    "car_r2":  ["E142",  "E143",  "E144",  "E135"],
    "car_r3":  ["E94",   "E95",   "E96",   "E97"],
    "car_r4":  ["E111",  "E112",  "E113",  "E114"],
    "car_r5":  ["E290",  "E303",  "-E92",  "-E14"],
    "car_r6":  ["-E140", "-E139", "-E138", "-E137"],
    "car_r7":  ["-E211", "-E210", "-E209", "-E208"],
    "car_r8":  ["E210",  "E211",  "E222",  "E84"],
    "car_r9":  ["E281",  "E193",  "E194",  "E195"],
    "car_r10": ["-E38",  "-E37",  "-E36"],
}

MOTO_ROUTES = {
    "moto_r1":  ["-E218", "-E217", "E233"],
    "moto_r2":  ["E38",   "E39",   "-E19",  "-E18"],
    "moto_r3":  ["-E38",  "-E37",  "-E36"],
    "moto_r4":  ["E362",  "-E213", "-E212", "E106"],
    "moto_r5":  ["-E201", "-E200", "-E199", "-E198"],
    "moto_r6":  ["-E217", "-E232", "E208",  "E277"],
    "moto_r7":  ["E41",   "E331",  "E191",  "E192"],
    "moto_r8":  ["E203",  "E204",  "E193",  "-E159"],
    "moto_r9":  ["E299",  "E76",   "E243"],
    "moto_r10": ["E212",  "E213",  "-E362"],
}

# ──────────────────────────────────────────────
# LOẠI XE
# ──────────────────────────────────────────────
VEHICLE_TYPES = {
    # ---------- Ô TÔ ----------
    "car_sedan": dict(
        vClass="passenger", length=4.5, width=1.8, height=1.5,
        minGap=2.5, maxSpeed=13.89, accel=2.6, decel=4.5, sigma=0.5,
        color=(0, 80, 200, 255),
    ),
    "car_van": dict(
        vClass="passenger", length=5.5, width=2.0, height=2.2,
        minGap=3.0, maxSpeed=11.11, accel=2.0, decel=4.0, sigma=0.5,
        color=(255, 140, 0, 255),
    ),
    # ---------- XE MÁY ----------
    "moto_normal": dict(
        vClass="motorcycle", length=2.2, width=0.8, height=1.5,
        minGap=1.5, maxSpeed=13.89, accel=3.5, decel=5.0, sigma=0.6,
        color=(220, 30, 30, 255),
    ),
    "moto_electric": dict(
        vClass="motorcycle", length=1.9, width=0.7, height=1.4,
        minGap=1.2, maxSpeed=11.11, accel=2.8, decel=4.5, sigma=0.5,
        color=(30, 180, 30, 255),
    ),
}

CAR_WEIGHTS  = [("car_sedan",     0.75), ("car_van",       0.25)]
MOTO_WEIGHTS = [("moto_normal",   0.70), ("moto_electric", 0.30)]

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def weighted_choice(pairs):
    types, weights = zip(*pairs)
    return random.choices(types, weights=weights, k=1)[0]


def setup_vehicle_types():
    for tid, p in VEHICLE_TYPES.items():
        traci.vehicletype.copy("DEFAULT_VEHTYPE", tid)
        traci.vehicletype.setVehicleClass(tid, p["vClass"])
        traci.vehicletype.setLength(tid, p["length"])
        traci.vehicletype.setWidth(tid, p["width"])
        traci.vehicletype.setHeight(tid, p["height"])
        traci.vehicletype.setMinGap(tid, p["minGap"])
        traci.vehicletype.setMaxSpeed(tid, p["maxSpeed"])
        traci.vehicletype.setAccel(tid, p["accel"])
        traci.vehicletype.setDecel(tid, p["decel"])
        traci.vehicletype.setImperfection(tid, p["sigma"])
        traci.vehicletype.setColor(tid, p["color"])
    print(f"[SETUP] Đã đăng ký {len(VEHICLE_TYPES)} loại xe")


def setup_routes():
    ok = 0
    all_routes = {**CAR_ROUTES, **MOTO_ROUTES}
    for rid, edges in all_routes.items():
        try:
            traci.route.add(rid, edges)
            ok += 1
        except traci.TraCIException as e:
            print(f"[WARN] Route '{rid}': {e}")
    print(f"[SETUP] Đã đăng ký {ok}/{len(all_routes)} routes")


# ──────────────────────────────────────────────
# SINH XE
# ──────────────────────────────────────────────
_cnt = {"car": 0, "moto": 0}

def spawn_vehicle(step, vtype_pairs, route_pool, prefix):
    route_id = random.choice(list(route_pool.keys()))
    type_id  = weighted_choice(vtype_pairs)
    veh_id   = f"{prefix}_{_cnt[prefix]:05d}"
    try:
        traci.vehicle.add(
            vehID=veh_id,
            routeID=route_id,
            typeID=type_id,
            depart=str(step),
            departLane="best",
            departSpeed="desired",
        )
        _cnt[prefix] += 1
    except traci.TraCIException:
        pass  # bỏ qua lỗi lane bận


# ──────────────────────────────────────────────
# THỐNG KÊ
# ──────────────────────────────────────────────
def print_stats(step):
    if step % 60 != 0:
        return
    vehs  = traci.vehicle.getIDList()
    cars  = [v for v in vehs if v.startswith("car_")]
    motos = [v for v in vehs if v.startswith("moto_")]
    avg_kmh = 0.0
    if vehs:
        avg_kmh = sum(traci.vehicle.getSpeed(v) for v in vehs) / len(vehs) * 3.6
    print(
        f"[T={step:4d}s] "
        f"Ô tô: {len(cars):3d}  Xe máy: {len(motos):3d}  "
        f"Tổng: {len(vehs):3d}  Tốc độ TB: {avg_kmh:5.1f} km/h  | "
        f"Đã tạo – Ô tô: {_cnt['car']}  Xe máy: {_cnt['moto']}"
    )


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def run():
    random.seed(42)
    binary = "sumo-gui" if GUI else "sumo"
    cmd = [
        binary,
        "-n", NET_FILE,
        "--step-length",      str(STEP_LENGTH),
        "--no-warnings",      "true",
        "--no-step-log",      "true",
        "--collision.action", "warn",
        "--time-to-teleport", "300",
    ]

    print("[INFO] Khởi động SUMO...")
    traci.start(cmd)
    setup_vehicle_types()
    setup_routes()

    print(f"[INFO] Bắt đầu mô phỏng {SIM_STEPS} bước...\n")
    for step in range(SIM_STEPS):
        traci.simulationStep()

        if random.random() < CAR_SPAWN_PROB:
            spawn_vehicle(step, CAR_WEIGHTS, CAR_ROUTES, "car")

        if random.random() < MOTO_SPAWN_PROB:
            spawn_vehicle(step, MOTO_WEIGHTS, MOTO_ROUTES, "moto")

        print_stats(step)

    print("\n" + "=" * 60)
    print("KẾT QUẢ MÔ PHỎNG")
    print(f"  Tổng ô tô đã tạo   : {_cnt['car']}")
    print(f"  Tổng xe máy đã tạo : {_cnt['moto']}")
    print("=" * 60)
    traci.close()


if __name__ == "__main__":
    run()