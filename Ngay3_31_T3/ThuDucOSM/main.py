"""
main.py
=======
Điều phối toàn bộ pipeline giao thông ngã tư Thủ Đức:
  1. Mô phỏng SUMO  → data/simulation_data.csv
  2. Chẩn đoán      → data/diagnosis_results.csv + report.png
  3. Huấn luyện     → model/ + data/training_results.csv

Cách chạy:
    python main.py              # chạy toàn bộ pipeline
    python main.py --step sim   # chỉ chạy mô phỏng
    python main.py --step diag  # chỉ chạy chẩn đoán
    python main.py --step train # chỉ huấn luyện
    python main.py --gui        # bật SUMO-GUI khi mô phỏng
    python main.py --predict 80 15 30  # dự đoán nhanh (xe, tốc độ, chờ)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║     HỆ THỐNG PHÂN TÍCH GIAO THÔNG NGÃ TƯ THỦ ĐỨC       ║
║         SUMO Simulation + ML Diagnosis Pipeline          ║
╚══════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ══════════════════════════════════════════════════════════════════════════════
def step_simulate(gui: bool = False):
    print("\n🚗 [BƯỚC 1/3] Mô phỏng giao thông SUMO...")
    t0 = time.time()
    from simulation.sumo_simulation import run_simulation
    path = run_simulation(gui=gui)
    elapsed = time.time() - t0
    print(f"   ✅ Hoàn tất trong {elapsed:.1f}s → {path}")
    return path


def step_diagnose():
    print("\n🔍 [BƯỚC 2/3] Chẩn đoán tình trạng giao thông...")
    t0 = time.time()
    from ai.diagnosis import run_diagnosis
    paths = run_diagnosis()
    elapsed = time.time() - t0
    print(f"   ✅ Hoàn tất trong {elapsed:.1f}s")
    for k, v in paths.items():
        print(f"      {k}: {v}")
    return paths


def step_train():
    print("\n🤖 [BƯỚC 3/3] Huấn luyện mô hình dự đoán...")
    t0 = time.time()
    from ai.train_model import run_training
    paths = run_training()
    elapsed = time.time() - t0
    print(f"   ✅ Hoàn tất trong {elapsed:.1f}s")
    for k, v in paths.items():
        print(f"      {k}: {v}")
    return paths


def step_predict(vehicles: int, speed_kmh: float, wait_s: float):
    print(f"\n🔮 Dự đoán: {vehicles} xe | {speed_kmh} km/h | {wait_s}s chờ")
    from ai.train_model import predict_traffic
    result = predict_traffic(vehicles, speed_kmh, wait_s)
    print(f"   Trạng thái    : {result['state']}")
    print(f"   Tắc nghẽn idx : {result['congestion_index']}")
    print("   Xác suất      :")
    for state, prob in sorted(result["confidence"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"     {state:<15} {prob:.3f}  {bar}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="Pipeline phân tích giao thông ngã tư Thủ Đức"
    )
    parser.add_argument(
        "--step", choices=["sim", "diag", "train", "all"],
        default="all", help="Bước cần chạy (mặc định: all)",
    )
    parser.add_argument("--gui",     action="store_true", help="Bật SUMO-GUI")
    parser.add_argument(
        "--predict", nargs=3, metavar=("VEHICLES", "SPEED_KMH", "WAIT_S"),
        type=float, help="Dự đoán nhanh không cần chạy pipeline",
    )
    args = parser.parse_args()

    # ── Chế độ dự đoán nhanh ─────────────────────────────────────────────────
    if args.predict:
        step_predict(int(args.predict[0]), args.predict[1], args.predict[2])
        return

    # ── Pipeline ──────────────────────────────────────────────────────────────
    t_total = time.time()

    if args.step in ("sim", "all"):
        step_simulate(gui=args.gui)

    if args.step in ("diag", "all"):
        step_diagnose()

    if args.step in ("train", "all"):
        step_train()

    total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"🏁 Pipeline hoàn tất trong {total:.1f} giây")
    print(f"   Dữ liệu   : {BASE_DIR / 'data'}")
    print(f"   Mô hình   : {BASE_DIR / 'model'}")
    print(f"   Báo cáo   : {BASE_DIR / 'data' / 'traffic_diagnostic_report.png'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()