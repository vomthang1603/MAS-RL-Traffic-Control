import sys
import os
from collections import defaultdict

# Thêm đường dẫn đến thư viện tools của SUMO
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Lỗi: Vui lòng khai báo biến môi trường 'SUMO_HOME'.")

import traci

# --- Cấu hình ---
SUMOCFG_FILE = "thuduc.sumocfg"
USE_GUI = True

# Ngưỡng cơ bản
JAM_THRESHOLD_SPEED = 2.0
JAM_HALTING_VEHICLES = 5

# Điều khiển
GREEN_PHASE_EXTENSION = 5
CONTROL_INTERVAL = 10


# =========================
# 🚦 PHÂN LOẠI GIAO THÔNG
# =========================
def calculate_congestion_score(speed, halting, density):
    """Tính điểm tắc nghẽn (0 → 1)."""

    speed_score = max(0, 1 - speed / 15)
    halt_score = min(1, halting / 10)
    density_score = min(1, density / 0.2)

    total_score = (0.4 * speed_score +
                   0.3 * halt_score +
                   0.3 * density_score)

    return total_score


def diagnose_traffic(score):
    """Phân loại tình trạng giao thông."""
    if score < 0.3:
        return "FREE_FLOW"
    elif score < 0.5:
        return "MODERATE"
    elif score < 0.7:
        return "HEAVY"
    else:
        return "JAM"


# =========================
# 🚀 SUMO INIT
# =========================
def init_sumo_simulation(sumocfg_file, use_gui=False):
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--quit-on-end", "--step-length", "1"]

    try:
        traci.start(sumo_cmd)
        print(f">> Đã kết nối SUMO ({sumo_binary})")
        return True
    except Exception as e:
        print(f"Lỗi SUMO: {e}")
        return False


def close_sumo_simulation():
    traci.close()
    print(">> Đã đóng SUMO")


# =========================
# 🔍 PHÂN TÍCH LÀN
# =========================
def get_green_lanes_for_phase(traci, tls_id, phase_state_string):
    green_lanes = set()
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    link_index = 0
    for group in controlled_links:
        for _ in group:
            if link_index < len(phase_state_string):
                if phase_state_string[link_index].lower() == 'g':
                    incoming_lane = controlled_links[link_index][0][0]
                    green_lanes.add(incoming_lane)
            link_index += 1

    return green_lanes


# =========================
# 📊 UPDATE STATE
# =========================
def update_lane_states(traci, all_lanes, lane_states, lane_scores):
    for lane_id in all_lanes:
        try:
            speed = traci.lane.getLastStepMeanSpeed(lane_id)
            halting = traci.lane.getLastStepHaltingNumber(lane_id)
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            lane_length = traci.lane.getLength(lane_id)

            density = vehicle_count / lane_length if lane_length > 0 else 0

            score = calculate_congestion_score(speed, halting, density)
            state = diagnose_traffic(score)

            lane_states[lane_id] = state
            lane_scores[lane_id] = score

            print(f"[Lane {lane_id}] {state} | Score={score:.2f}")

        except traci.TraCIException:
            lane_states[lane_id] = "UNKNOWN"
            lane_scores[lane_id] = 0


# =========================
# 🚥 ĐIỀU KHIỂN ĐÈN
# =========================
def adaptive_control_logic(traci, tls_id, tls_phase_lanes, lane_scores):
    phase_pressures = {}

    for phase_index, lanes in tls_phase_lanes.items():
        pressure = sum(lane_scores.get(l, 0) for l in lanes)
        phase_pressures[phase_index] = pressure

    if not phase_pressures:
        return

    best_phase_index = max(phase_pressures, key=phase_pressures.get)
    current_phase = traci.trafficlight.getPhase(tls_id)

    current_state = traci.trafficlight.getPhaseState(tls_id)
    if 'y' in current_state.lower():
        return

    if current_phase == best_phase_index:
        remaining = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
        traci.trafficlight.setPhaseDuration(tls_id, remaining + GREEN_PHASE_EXTENSION)
        print(f"→ Kéo dài pha {current_phase}")
    else:
        traci.trafficlight.setPhase(tls_id, best_phase_index)
        print(f"→ Chuyển sang pha {best_phase_index}")


# =========================
# 🔁 MAIN
# =========================
def main_control():
    print("=== BẮT ĐẦU MÔ PHỎNG ===")

    if not init_sumo_simulation(SUMOCFG_FILE, USE_GUI):
        return

    step = 0
    tls_ids = traci.trafficlight.getIDList()

    all_controlled_lanes = set()
    tls_phase_lanes_map = defaultdict(dict)

    for tls_id in tls_ids:
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        traci.trafficlight.setProgramLogic(tls_id, logic)

        for i, phase in enumerate(logic.phases):
            if 'y' not in phase.state.lower() and 'g' in phase.state.lower():
                lanes = get_green_lanes_for_phase(traci, tls_id, phase.state)
                if lanes:
                    tls_phase_lanes_map[tls_id][i] = lanes
                    all_controlled_lanes.update(lanes)

    lane_states = {}
    lane_scores = {}

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1

            # Cập nhật trạng thái
            update_lane_states(traci, all_controlled_lanes, lane_states, lane_scores)

            # Điều khiển định kỳ
            if step % CONTROL_INTERVAL == 0:
                print(f"\n=== STEP {step} ===")
                for tls_id in tls_ids:
                    if tls_id in tls_phase_lanes_map:
                        adaptive_control_logic(
                            traci,
                            tls_id,
                            tls_phase_lanes_map[tls_id],
                            lane_scores
                        )

    except traci.exceptions.FatalTraCIError:
        print("Mô phỏng kết thúc")

    finally:
        close_sumo_simulation()


# =========================
# ▶ RUN
# =========================
if __name__ == "__main__":
    if not os.path.exists("thuduc_custom.rou.xml"):
        print("Lỗi: thiếu file route")
    else:
        main_control()