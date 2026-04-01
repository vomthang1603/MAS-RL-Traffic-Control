"""
sumo_simulation.py
==================
Mô phỏng giao thông ngã tư Thủ Đức.
- Ưu tiên dùng route file sẵn có
- Fallback: inject xe trực tiếp qua TraCI nếu route file lỗi
- Fallback cuối: mock data nếu không có SUMO
"""

import os, sys, subprocess, logging, random
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ── Đường dẫn ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "sumo_config"
DATA_DIR   = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SUMO_CFG  = CONFIG_DIR / "thuduc.sumocfg"
NET_FILE  = CONFIG_DIR / "thuduc.net.xml"
ROU_FILE  = CONFIG_DIR / "thuduc.rou.xml"
TRIP_FILE = CONFIG_DIR / "trips.trips.xml"

# ── SUMO_HOME ─────────────────────────────────────────────────────────────────
SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
tools_path = os.path.join(SUMO_HOME, "tools")
if tools_path not in sys.path:
    sys.path.insert(0, tools_path)

try:
    import traci
    import sumolib
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DATA_DIR / "simulation.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Tham số ───────────────────────────────────────────────────────────────────
SIMULATION_STEPS  = 3600
STEP_INTERVAL     = 10
PORT              = 8813
TARGET_VEHICLES   = 50    # số xe inject mỗi lần spawn (TraCI fallback)
SPAWN_INTERVAL    = 12    # spawn batch xe mỗi 30 giây


# ══════════════════════════════════════════════════════════════════════════════
# Tiện ích network
# ══════════════════════════════════════════════════════════════════════════════

def get_valid_edges(net) -> list:
    """Trả về danh sách edge có thể đặt xe (không phải internal, có lane)."""
    edges = []
    for e in net.getEdges():
        eid = e.getID()
        if eid.startswith(":"):
            continue
        if e.getSpeed() < 1.0:
            continue
        edges.append(eid)
    return edges


def _count_vehicles_in_route(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        root = ET.parse(path).getroot()
        return (len(root.findall("vehicle"))
                + len(root.findall("flow"))
                + len(root.findall("trip")))
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# Tạo route file
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_routes(force: bool = False):
    """Tạo lại route từ network. Bỏ qua nếu đã có và force=False."""
    n = _count_vehicles_in_route(ROU_FILE)
    if n > 0 and not force:
        log.info("Route file OK (%d entries). Dùng lại.", n)
        return True

    log.info("Tạo lại route file...")
    rtrips = Path(SUMO_HOME) / "tools" / "randomTrips.py"
    dua    = Path(SUMO_HOME) / "bin" / "duarouter.exe"

    if not rtrips.exists() or not dua.exists():
        log.error("Không tìm thấy randomTrips.py hoặc duarouter. Kiểm tra SUMO_HOME.")
        return False

    # Xóa file cũ
    ROU_FILE.unlink(missing_ok=True)
    TRIP_FILE.unlink(missing_ok=True)

    # Sinh trips
    r1 = subprocess.run([
        sys.executable, str(rtrips),
        "-n", str(NET_FILE),
        "-o", str(TRIP_FILE),
        "--begin", "0", "--end", str(SIMULATION_STEPS),
        "--period", "2",
        "--fringe-factor", "10",
        "--random-seed", "42",
        "--validate",
    ], capture_output=True, text=True)

    if r1.returncode != 0:
        log.error("randomTrips lỗi:\n%s", r1.stderr[-600:])
        return False
    log.info("✅ Trips OK → %s", TRIP_FILE)

    # Convert trips → routes
    r2 = subprocess.run([
        str(dua),
        "-n", str(NET_FILE),
        "-r", str(TRIP_FILE),
        "-o", str(ROU_FILE),
        "--ignore-errors", "--no-warnings",
        "--begin", "0", "--end", str(SIMULATION_STEPS),
    ], capture_output=True, text=True)

    n_after = _count_vehicles_in_route(ROU_FILE)
    if n_after == 0:
        log.error("duarouter không tạo được route!\n%s", r2.stderr[-600:])
        return False

    log.info("✅ Route file: %d entries", n_after)
    return True


def write_sumocfg():
    """Ghi sumocfg chuẩn trỏ đúng net + route."""
    cfg = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="thuduc.net.xml"/>
        <route-files value="thuduc.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{SIMULATION_STEPS}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
        <error-log value="sumo_errors.txt"/>
    </report>
</configuration>
"""
    SUMO_CFG.write_text(cfg, encoding="utf-8")
    log.info("✅ sumocfg → %s", SUMO_CFG)


# ══════════════════════════════════════════════════════════════════════════════
# Inject xe qua TraCI (fallback khi route không hoạt động)
# ══════════════════════════════════════════════════════════════════════════════

class VehicleInjector:
    """Thêm xe vào mô phỏng trực tiếp qua TraCI."""

    def __init__(self, net):
        self.valid_edges = get_valid_edges(net)
        self._vtype_added = False
        self._counter = 0
        random.seed(42)
        log.info("VehicleInjector: %d edges hợp lệ", len(self.valid_edges))

    def _ensure_vtype(self):
        if self._vtype_added:
            return
        try:
            traci.vehicletype.copy("DEFAULT_VEHTYPE", "car")
            traci.vehicletype.setMaxSpeed("car", 13.9)   # 50 km/h
            traci.vehicletype.setLength("car", 4.5)
        except Exception:
            pass
        self._vtype_added = True

    def _find_valid_route(self, max_try: int = 20) -> tuple[str, str] | None:
        """Tìm cặp (origin, destination) có route hợp lệ."""
        for _ in range(max_try):
            src, dst = random.sample(self.valid_edges, 2)
            try:
                route = traci.simulation.findRoute(src, dst)
                if route.edges:
                    return src, dst, list(route.edges)
            except Exception:
                continue
        return None

    def spawn_batch(self, n: int = 10):
        """Spawn n xe ngẫu nhiên."""
        self._ensure_vtype()
        spawned = 0
        for _ in range(n * 3):   # thử nhiều lần để đủ n xe
            if spawned >= n:
                break
            result = self._find_valid_route()
            if result is None:
                continue
            src, dst, route_edges = result
            vid = f"injected_{self._counter}"
            rid = f"route_{self._counter}"
            self._counter += 1
            try:
                traci.route.add(rid, route_edges)
                traci.vehicle.add(vid, rid, typeID="car",
                                  depart="now", departLane="best",
                                  departSpeed="random")
                spawned += 1
            except Exception:
                continue
        return spawned


# ══════════════════════════════════════════════════════════════════════════════
# Lớp mô phỏng
# ══════════════════════════════════════════════════════════════════════════════

class SumoSimulation:

    def __init__(self, cfg_path: Path = SUMO_CFG, gui: bool = False):
        self.cfg_path  = cfg_path
        self.gui       = gui
        self.records: list[dict] = []
        self._net      = None
        self._injector = None
        self._use_injection = False   # bật nếu route file không spawn được xe

    # ── Khởi động ────────────────────────────────────────────────────────────
    def start(self):
        if not TRACI_AVAILABLE:
            log.warning("TraCI không có → mock mode.")
            return

        write_sumocfg()
        route_ok = rebuild_routes(force=False)

        if not route_ok:
            log.warning("Không tạo được route → sẽ dùng TraCI injection.")
            self._use_injection = True
            # Ghi route rỗng để SUMO không báo lỗi thiếu file
            if not ROU_FILE.exists():
                ROU_FILE.write_text(
                    '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"></routes>',
                    encoding="utf-8",
                )

        binary = str(Path(SUMO_HOME) / "bin" / ("sumo-gui" if self.gui else "sumo"))
        cmd = [
            binary, "-c", str(self.cfg_path),
            "--no-step-log",
            "--time-to-teleport", "-1",
            "--collision.action", "warn",
            "--quit-on-end",
        ]
        traci.start(cmd, port=PORT)
        log.info("SUMO started ✅")

        # Load network
        if NET_FILE.exists():
            self._net = sumolib.net.readNet(str(NET_FILE), withInternal=False)
            n_edges = len(get_valid_edges(self._net))
            log.info("Network: %d valid edges", n_edges)
            self._injector = VehicleInjector(self._net)

        # Kiểm tra nếu xe chưa load được thì bật injection
        traci.simulationStep()
        if traci.vehicle.getIDCount() == 0:
            log.warning("Không có xe sau bước đầu → bật TraCI injection mode.")
            self._use_injection = True

    # ── Vòng lặp ─────────────────────────────────────────────────────────────
    def run(self):
        if not TRACI_AVAILABLE:
            self._generate_mock_data()
            return

        log.info("Mô phỏng bắt đầu... (injection=%s)", self._use_injection)
        no_vehicle_count = 0

        for step in range(1, SIMULATION_STEPS):
            # Inject xe định kỳ nếu cần
            if self._use_injection and self._injector:
                if step % SPAWN_INTERVAL == 0:
                    current = traci.vehicle.getIDCount()
                    need    = max(0, TARGET_VEHICLES - current)
                    if need > 0:
                        spawned = self._injector.spawn_batch(min(need, 20))
                        if step % 300 == 0:
                            log.info("Injected %d xe (hiện tại: %d)", spawned, current + spawned)

            traci.simulationStep()

            n_veh = traci.vehicle.getIDCount()
            if n_veh == 0:
                no_vehicle_count += 1
                if no_vehicle_count == 200 and not self._use_injection:
                    log.warning("200 bước 0 xe → bật injection mode.")
                    self._use_injection = True
            else:
                no_vehicle_count = 0

            if step % STEP_INTERVAL == 0:
                self._collect(step)

        log.info("Mô phỏng xong. %d mẫu thu thập.", len(self.records))

    # ── Thu thập dữ liệu ─────────────────────────────────────────────────────
    def _collect(self, step: int):
        vids = traci.vehicle.getIDList()
        n    = len(vids)

        speeds = [traci.vehicle.getSpeed(v)       for v in vids] or [0.0]
        waits  = [traci.vehicle.getWaitingTime(v) for v in vids] or [0.0]
        avg_spd  = float(np.mean(speeds))
        avg_wait = float(np.mean(waits))
        max_wait = float(np.max(waits))

        road_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
        edge_data  = self._edge_metrics(road_edges[:8])

        self.records.append({
            "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sim_time_s":     step,
            "total_vehicles": n,
            "avg_speed_ms":   round(avg_spd, 3),
            "avg_speed_kmh":  round(avg_spd * 3.6, 2),
            "avg_wait_s":     round(avg_wait, 2),
            "max_wait_s":     round(max_wait, 2),
            "throughput":     traci.simulation.getArrivedNumber(),
            "departed":       traci.simulation.getDepartedNumber(),
            "teleported":     traci.simulation.getStartingTeleportNumber(),
            **edge_data,
        })

        if step % 300 == 0:
            log.info("⏱ t=%4ds | 🚗 %3d xe | 🏎 %.1f km/h | ⏳ %.1fs",
                     step, n, avg_spd * 3.6, avg_wait)

    def _edge_metrics(self, edges: list) -> dict:
        out = {}
        for i, eid in enumerate(edges):
            k = f"e{i+1}"
            try:
                v   = traci.edge.getLastStepVehicleNumber(eid)
                spd = traci.edge.getLastStepMeanSpeed(eid)
                occ = traci.edge.getLastStepOccupancy(eid)
            except Exception:
                v = spd = occ = 0.0
            out[f"{k}_id"]      = eid
            out[f"{k}_count"]   = v
            out[f"{k}_spd_kmh"] = round(spd * 3.6, 2)
            out[f"{k}_occ"]     = round(occ, 3)
        return out

    # ── Mock data ─────────────────────────────────────────────────────────────
    def _generate_mock_data(self):
        log.info("Mock mode: sinh %d mẫu", SIMULATION_STEPS // STEP_INTERVAL)
        rng = np.random.default_rng(42)
        for step in range(0, SIMULATION_STEPS, STEP_INTERVAL):
            h    = (step / 3600) % 24
            peak = (0.9 * np.exp(-0.5*((h-8)/1)**2)
                   + 0.8 * np.exp(-0.5*((h-18)/1)**2))
            n   = max(5, int(50 + 150*peak + rng.normal(0, 10)))
            spd = max(3.0, 14.0 - 10.0*peak + rng.normal(0, 1))
            wt  = max(0.0, 30*peak + rng.normal(0, 5))
            self.records.append({
                "timestamp":      f"2024-01-01 {int(h):02d}:{int((h%1)*60):02d}:00",
                "sim_time_s":     step,
                "total_vehicles": n,
                "avg_speed_ms":   round(spd, 3),
                "avg_speed_kmh":  round(spd * 3.6, 2),
                "avg_wait_s":     round(wt, 2),
                "max_wait_s":     round(wt*2.5 + rng.exponential(10), 2),
                "throughput":     int(rng.poisson(5 + 10*peak)),
                "departed":       int(rng.poisson(6 + 12*peak)),
                "teleported":     int(rng.poisson(0.2)),
            })

    def stop(self):
        if TRACI_AVAILABLE:
            try:
                traci.close()
            except Exception:
                pass

    def export(self) -> Path:
        if not self.records:
            log.error("Không có dữ liệu!")
            return None

        df = pd.DataFrame(self.records)
        df["traffic_state"]    = df["avg_speed_kmh"].apply(_classify)
        df["congestion_index"] = _cong_idx(df)

        out = DATA_DIR / "simulation_data.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        df.describe().round(3).to_csv(DATA_DIR / "simulation_summary.csv",
                                      encoding="utf-8-sig")

        log.info("✅ %d bản ghi → %s", len(df), out)
        log.info("  Xe TB: %.1f | Tốc độ TB: %.1f km/h | Chờ TB: %.1fs",
                 df["total_vehicles"].mean(),
                 df["avg_speed_kmh"].mean(),
                 df["avg_wait_s"].mean())
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _classify(kmh: float) -> str:
    if kmh >= 40: return "Thông thoáng"
    if kmh >= 20: return "Bình thường"
    if kmh >= 10: return "Chậm"
    return "Tắc nghẽn"

def _cong_idx(df: pd.DataFrame) -> pd.Series:
    ns = 1 - (df["avg_speed_kmh"].clip(0, 50) / 50)
    nw = df["avg_wait_s"].clip(0, 120) / 120
    return (ns*0.6 + nw*0.4).round(3)


# ══════════════════════════════════════════════════════════════════════════════
# Entry points
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(gui: bool = False) -> Path:
    sim = SumoSimulation(gui=gui)
    try:
        sim.start()
        sim.run()
    finally:
        sim.stop()
    return sim.export()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gui",          action="store_true")
    p.add_argument("--regen-routes", action="store_true",
                   help="Xóa và tạo lại route file")
    args = p.parse_args()

    if args.regen_routes:
        rebuild_routes(force=True)
    else:
        run_simulation(gui=args.gui)