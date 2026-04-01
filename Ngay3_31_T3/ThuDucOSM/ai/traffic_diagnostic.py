import sys
import os
from collections import Counter
import numpy as np

# Kiểm tra và import thư viện đồ họa, nếu không có thì bỏ qua phần vẽ
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# --- Cấu hình ---
SUMOCFG_FILE = "thuduc.sumocfg"
PEAK_START = 900  # Thời điểm bắt đầu giờ cao điểm (giây)
PEAK_END = 2700    # Thời điểm kết thúc giờ cao điểm (giây)
TIME_BUCKET = 300  # Độ rộng của mỗi bucket thời gian cho biểu đồ (giây)

# Ngưỡng phân loại Mức độ Dịch vụ (Level of Service - LOS)
LOS_THRESHOLDS = [
    (0.35, "A", "Free Flow"), (0.55, "B", "Stable Flow"),
    (0.75, "C", "Stable Flow (approaching limit)"), (0.90, "D", "Approaching Unstable Flow"),
    (1.00, "E", "Unstable Flow"), (float('inf'), "F", "Forced/Breakdown Flow"),
]

def bpr_delay_factor(vc_ratio, alpha=0.15, beta=4.0):
    """Tính hệ số trễ BPR (Bureau of Public Roads)."""
    return 1 + alpha * (vc_ratio ** beta)

def init_sumo_simulation(sumocfg_file, use_gui=False, verbose=False):
    """Khởi tạo và kết nối với SUMO."""
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        if tools not in sys.path:
            sys.path.append(tools)
    else:
        sys.exit("Lỗi: Vui lòng khai báo biến môi trường 'SUMO_HOME' trỏ đến thư mục cài đặt SUMO.")

    try:
        import traci
    except ImportError:
        sys.exit("Lỗi: Thư viện TraCI không tìm thấy. Hãy chắc chắn 'SUMO_HOME' đã được thiết lập đúng.")

    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--quit-on-end"]
    
    try:
        traci.start(sumo_cmd)
        if verbose:
            print(f"      >> Đã kết nối thành công với SUMO ({sumo_binary}).")
        return traci, True
    except traci.TraCIException as e:
        print(f"      >> Lỗi khi khởi tạo SUMO: {e}")
        return None, False
    except FileNotFoundError:
        print(f"      >> Lỗi: Không tìm thấy '{sumo_binary}'. Hãy chắc chắn SUMO đã được cài đặt và có trong PATH của hệ thống.")
        return None, False

def close_sumo_simulation(traci):
    """Đóng kết nối TraCI."""
    if traci:
        traci.close()
        print("      >> Đã đóng kết nối TraCI.")

# ══════════════════════════════════════════════════════════════════════════════
#  CHẠY CHƯƠNG TRÌNH CHÍNH
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 70)
    print("  KHỞI ĐỘNG – Mô phỏng giao thông SUMO")
    print("  Ngã tư Thủ Đức, TP.HCM")
    print("═" * 70)

    # ── Bước 1: Khởi tạo và CHẠY mô phỏng trực quan ──────────────────────
    print("\n[1/2] Khởi tạo mô phỏng SUMO...")
    # Đặt thời gian mô phỏng tối đa (giây), 3600s = 1 giờ
    SIM_DURATION = 3600
    traci, sumo_running = init_sumo_simulation(SUMOCFG_FILE, verbose=True)
    
    if sumo_running:
        print("      >> Đang chạy mô phỏng thực tế trên giao diện GUI...")
        step = 0
        # Vòng lặp: Chạy cho đến khi hết xe hoặc đạt giới hạn thời gian
        while traci.simulation.getMinExpectedNumber() > 0 and step < SIM_DURATION:
            traci.simulationStep() # Lệnh này làm SUMO nhích lên 1 giây
            
            # Hiển thị dữ liệu thực tế mỗi 100 giây mô phỏng
            if step % 100 == 0:
                veh_count = traci.vehicle.getIDCount()
                print(f"      [TraCI] Giây thứ {step}: Đang có {veh_count} xe di chuyển trên bản đồ.")
            
            step += 1
            
        print("\n[2/2] Hoàn tất chạy mô phỏng trực quan!")
        close_sumo_simulation(traci) # Đóng GUI sau khi chạy xong

    # ── Ghi chú: Phần phân tích và vẽ biểu đồ đã được tách ra ───────────
    # Để chạy phần phân tích, bạn cần tạo dữ liệu đầu vào (metrics, flow_series, etc.)
    # và sau đó gọi hàm plot_diagnostics(). Hiện tại, script này chỉ tập trung
    # vào việc chạy mô phỏng.

# ══════════════════════════════════════════════════════════════════════════════
#  HÀM VẼ BIỂU ĐỒ CHUẨN ĐOÁN (TÁCH RIÊNG)
# ══════════════════════════════════════════════════════════════════════════════
def plot_diagnostics(metrics: dict,
                    flow_series: list,
                    interventions: list,
                    edge_freq: Counter):
    if not HAS_PLOT:
        print("[PLOT] Bỏ qua việc vẽ biểu đồ vì không tìm thấy Matplotlib.")
        return

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Chuẩn đoán Giao thông – Ngã tư Thủ Đức", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Màu theo LOS
    LOS_COLORS = {"A": "#1D9E75", "B": "#5DCAA5", "C": "#EF9F27",
                    "D": "#BA7517", "E": "#E24B4A", "F": "#8B0000"}

    # ── Biểu đồ 1: Lưu lượng theo thời gian ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x    = [i * TIME_BUCKET for i in range(len(flow_series))]
    cols = ["#EF9F27" if PEAK_START <= t < PEAK_END else "#1D9E75" for t in x]
    ax1.bar(x, flow_series, width=TIME_BUCKET * 0.85, color=cols, edgecolor="none")
    ax1.set_title("Lưu lượng khởi hành theo thời gian", fontsize=11)
    ax1.set_xlabel("Thời gian (s)")
    ax1.set_ylabel("Số xe / bucket")
    ax1.legend(handles=[
        plt.Rectangle((0,0),1,1, color="#EF9F27", label="Cao điểm"),
        plt.Rectangle((0,0),1,1, color="#1D9E75", label="Thấp điểm"),
    ], fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ── Biểu đồ 2: Phân bố LOS toàn mạng ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    los_cnt = Counter(m["los_grade"] for m in metrics.values() if m["freq"] > 0)
    grades  = [g for g in "ABCDEF" if g in los_cnt]
    counts  = [los_cnt[g] for g in grades]
    colors  = [LOS_COLORS[g] for g in grades]
    bars    = ax2.bar(grades, counts, color=colors, edgecolor="none")
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(cnt), ha="center", va="bottom", fontsize=9)
    ax2.set_title("Phân bố Mức Dịch Vụ (LOS) toàn mạng", fontsize=11)
    ax2.set_xlabel("LOS Grade")
    ax2.set_ylabel("Số cạnh")
    ax2.grid(axis="y", alpha=0.3)

    # ── Biểu đồ 3: Risk Score top 10 cạnh ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    top10_ids   = [r["edge_id"][:16] for r in interventions]
    top10_risk  = [r["risk_score"] for r in interventions]
    top10_cols  = [LOS_COLORS.get(r["los_grade"], "#888") for r in interventions]
    y_pos       = range(len(top10_ids))
    ax3.barh(list(y_pos), top10_risk, color=top10_cols, edgecolor="none")
    ax3.set_yticks(list(y_pos))
    ax3.set_yticklabels(top10_ids, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_title("Top 10 Điểm nguy cơ (Risk Score)", fontsize=11)
    ax3.set_xlabel("Risk Score")
    ax3.set_xlim(0, 1.0)
    ax3.axvline(0.6, color="#E24B4A", linestyle="--", linewidth=0.8, label="Ngưỡng cao")
    ax3.axvline(0.35, color="#EF9F27", linestyle="--", linewidth=0.8, label="Ngưỡng vừa")
    ax3.legend(fontsize=8)
    ax3.grid(axis="x", alpha=0.3)

    # ── Biểu đồ 4: V/C ratio phân phối (histogram) ───────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    vc_vals = [m["vc_ratio"] for m in metrics.values() if m["freq"] > 0]
    ax4.hist(vc_vals, bins=20, color="#1D9E75", edgecolor="white", linewidth=0.4)
    for thr, grade, _ in LOS_THRESHOLDS[:-1]:
        ax4.axvline(thr, linestyle="--", linewidth=0.7, color="#E24B4A", alpha=0.5)
        ax4.text(thr + 0.01, ax4.get_ylim()[1] * 0.9, grade, fontsize=7, color="#E24B4A")
    ax4.set_title("Phân phối V/C Ratio toàn mạng", fontsize=11)
    ax4.set_xlabel("V/C Ratio")
    ax4.set_ylabel("Số cạnh")
    ax4.grid(alpha=0.3)

    # ── Biểu đồ 5: Scatter – V/C vs Tần suất ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    active = [(eid, m) for eid, m in metrics.items() if m["freq"] > 0]
    sc_x   = [m["vc_ratio"]  for _, m in active]
    sc_y   = [m["freq"]      for _, m in active]
    sc_c   = [LOS_COLORS.get(m["los_grade"], "#888") for _, m in active]
    sc_s   = [m["risk_score"] * 200 for _, m in active]
    ax5.scatter(sc_x, sc_y, c=sc_c, s=sc_s, alpha=0.7, edgecolors="none")
    ax5.set_title("V/C Ratio vs Tần suất xuất hiện", fontsize=11)
    ax5.set_xlabel("V/C Ratio")
    ax5.set_ylabel("Tổng lượt đi qua")
    ax5.axvline(1.0, color="#E24B4A", linestyle="--", linewidth=0.8, label="V/C = 1.0")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    # Chú thích top 3 cạnh nguy hiểm nhất
    top3 = sorted(active, key=lambda x: x[1]["risk_score"], reverse=True)[:3]
    for eid, m in top3:
        ax5.annotate(eid[:12], (m["vc_ratio"], m["freq"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)

    # ── Biểu đồ 6: BPR Delay curve + điểm thực tế ────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    vc_curve = np.linspace(0, 1.5, 300)
    bpr_curve = [bpr_delay_factor(v) for v in vc_curve]
    ax6.plot(vc_curve, bpr_curve, color="#1D9E75", linewidth=2, label="BPR curve")
    # Điểm các cạnh thực tế
    pts_vc  = [m["vc_ratio"]   for _, m in active[:50]]
    pts_bpr = [m["bpr_factor"] for _, m in active[:50]]
    pts_c   = [LOS_COLORS.get(m["los_grade"], "#888") for _, m in active[:50]]
    ax6.scatter(pts_vc, pts_bpr, c=pts_c, s=20, alpha=0.6, edgecolors="none", zorder=3)
    ax6.axvline(1.0, color="#E24B4A", linestyle="--", linewidth=0.8, alpha=0.6)
    ax6.set_title("BPR Delay Factor theo V/C Ratio", fontsize=11)
    ax6.set_xlabel("V/C Ratio")
    ax6.set_ylabel("Hệ số trễ BPR (t_actual / t_free)")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    plt.savefig("traffic_diagnostic_report.png", dpi=150, bbox_inches="tight")
    print("[PLOT] Đồ thị đã lưu: traffic_diagnostic_report.png")
    plt.show()

if __name__ == "__main__":
    # Trước khi chạy, hãy đảm bảo bạn đã tạo file route bằng generate_scenario.py
    if not os.path.exists("thuduc_custom.rou.xml"):
        print("Lỗi: File 'thuduc_custom.rou.xml' không tồn tại.")
        print("Vui lòng chạy 'python generate_scenario.py' trước để tạo file này.")
    else:
        main()