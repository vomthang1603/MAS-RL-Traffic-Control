# =============================================================================
# run_all.py — Pipeline RL N4TD (5 bước)
#
# Bước 1: Kiểm tra kết nối SUMO/TraCI
# Bước 2: Q-Learning trên low_demand
# Bước 3: Validate so với Fixed-Time
# Bước 4: DQN trên medium_demand và peak_hour
# Bước 5: Benchmark toàn bộ
#
# Chạy:
#   python run_all.py                  # toàn bộ pipeline
#   python run_all.py --step 4         # bắt đầu từ Bước 4
#   python run_all.py --step 2 --gui
# =============================================================================

import os
import sys
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SCENARIOS, GL1_STATE_SIZE, GL2_STATE_SIZE


# ---------------------------------------------------------------------------
# Bước 1: Kiểm tra kết nối SUMO/TraCI
# ---------------------------------------------------------------------------

def step1_check_connection() -> bool:
    print("\n" + "=" * 60)
    print("  BƯỚC 1: Kiểm tra kết nối SUMO ↔ TraCI")
    print("=" * 60)

    if 'SUMO_HOME' not in os.environ:
        print("  ❌ SUMO_HOME chưa được khai báo!")
        print("     Windows: set SUMO_HOME=C:\\sumo")
        print("     Linux  : export SUMO_HOME=/usr/share/sumo")
        return False

    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    try:
        import traci
        print(f"  ✅ TraCI import OK")
        print(f"  ✅ SUMO_HOME = {os.environ['SUMO_HOME']}")
    except ImportError:
        print("  ❌ Không import được traci. Kiểm tra lại SUMO_HOME.")
        return False

    from environment.sumo_env import SumoEnv
    try:
        env     = SumoEnv(SCENARIOS['low_demand'], use_gui=False)
        s1, s2  = env.reset()
        print(f"  ✅ SumoEnv reset OK")
        print(f"     GL1 state: {s1.shape}  (expected: ({GL1_STATE_SIZE},))")
        print(f"     GL2 state: {s2.shape}  (expected: ({GL2_STATE_SIZE},))")
        (_, _), reward, _, _ = env.step(0, 0)
        print(f"  ✅ env.step(0,0) OK  reward={reward:.2f}")
        env.close()
        print(f"  ✅ Kết nối TraCI đóng thành công\n")
        return True
    except Exception as e:
        print(f"  ❌ Lỗi khi chạy SumoEnv: {e}")
        return False


# ---------------------------------------------------------------------------
# Bước 2: Q-Learning
# ---------------------------------------------------------------------------

def step2_qlearning(episodes: int = 30, use_gui: bool = False):
    print("\n" + "=" * 60)
    print("  BƯỚC 2: Q-Learning — Scenario: low_demand")
    print("=" * 60)
    from training.train_q import train
    rewards, queues = train(scenario='low_demand',
                            episodes=episodes, use_gui=use_gui)
    print(f"\n  ✅ Q-Learning hoàn tất.")
    print(f"     Best reward : {max(rewards):.1f}")
    print(f"     Final avg Q : {queues[-1]:.2f}")


# ---------------------------------------------------------------------------
# Bước 3: Validate
# ---------------------------------------------------------------------------

def step3_validate(use_gui: bool = False):
    print("\n" + "=" * 60)
    print("  BƯỚC 3: Validate — Fixed-Time vs Q-Learning (low_demand)")
    print("=" * 60)
    from training.evaluate import benchmark
    benchmark(scenarios=['low_demand'], use_gui=use_gui)
    print("\n  ✅ Validation hoàn tất. Xem biểu đồ tại results/plots/")


# ---------------------------------------------------------------------------
# Bước 4: DQN
# ---------------------------------------------------------------------------

def step4_dqn(episodes: int = 50, use_gui: bool = False):
    print("\n" + "=" * 60)
    print("  BƯỚC 4: DQN Training — medium_demand & peak_hour")
    print("=" * 60)
    try:
        from training.train_dqn import train
    except ModuleNotFoundError as exc:
        if exc.name == 'tensorflow':
            print("  ⚠️  Bỏ qua BƯỚC 4 — chưa cài TensorFlow.")
            print("     Cài thêm: pip install tensorflow")
            return
        raise

    for scenario in ['medium_demand', 'peak_hour']:
        print(f"\n  --- Scenario: {scenario} ---")
        train(scenario=scenario, episodes=episodes, use_gui=use_gui)
        print(f"  ✅ DQN [{scenario}] hoàn tất.")


# ---------------------------------------------------------------------------
# Bước 5: Benchmark
# ---------------------------------------------------------------------------

def step5_benchmark(use_gui: bool = False):
    print("\n" + "=" * 60)
    print("  BƯỚC 5: Benchmark toàn bộ scenario")
    print("=" * 60)
    from training.evaluate import benchmark
    benchmark(
        scenarios=['low_demand', 'medium_demand', 'peak_hour'],
        use_gui=use_gui,
    )
    print("\n  ✅ Benchmark hoàn tất → results/plots/benchmark_comparison.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pipeline RL N4TD — chạy từng bước hoặc toàn bộ'
    )
    parser.add_argument('--step', type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help='Bắt đầu từ bước nào (default=1)')
    parser.add_argument('--episodes_ql',  type=int, default=30)
    parser.add_argument('--episodes_dqn', type=int, default=50)
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()

    steps = range(args.step, 6)

    if 1 in steps:
        if not step1_check_connection():
            sys.exit(1)

    if 2 in steps:
        step2_qlearning(episodes=args.episodes_ql, use_gui=args.gui)

    if 3 in steps:
        step3_validate(use_gui=args.gui)

    if 4 in steps:
        step4_dqn(episodes=args.episodes_dqn, use_gui=args.gui)

    if 5 in steps:
        step5_benchmark(use_gui=args.gui)

    print("\n" + "=" * 60)
    print("  Pipeline hoàn tất! Xem kết quả tại results/plots/")
    print("=" * 60)
