"""
diagnosis.py
============
Chẩn đoán tình trạng giao thông từ dữ liệu mô phỏng SUMO.
Đầu vào : data/simulation_data.csv
Đầu ra  : data/diagnosis_results.csv
           data/traffic_diagnostic_report.png
           data/diagnosis_summary.txt

Yêu cầu:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Đường dẫn ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DATA_DIR / "diagnosis.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Màu sắc trạng thái ────────────────────────────────────────────────────────
STATE_COLORS = {
    "Thông thoáng": "#2ecc71",
    "Bình thường":  "#f39c12",
    "Chậm":         "#e67e22",
    "Tắc nghẽn":    "#e74c3c",
}
STATE_ORDER = ["Thông thoáng", "Bình thường", "Chậm", "Tắc nghẽn"]


# ══════════════════════════════════════════════════════════════════════════════
# Lớp chẩn đoán
# ══════════════════════════════════════════════════════════════════════════════
class TrafficDiagnosis:
    """Phân tích và chẩn đoán tình trạng giao thông ngã tư Thủ Đức."""

    def __init__(self, data_path: Path = DATA_DIR / "simulation_data.csv"):
        self.data_path = data_path
        self.df: pd.DataFrame = None
        self.results: pd.DataFrame = None

    # ── Tải dữ liệu ──────────────────────────────────────────────────────────
    def load(self) -> "TrafficDiagnosis":
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {self.data_path}. Hãy chạy sumo_simulation.py trước."
            )
        self.df = pd.read_csv(self.data_path, encoding="utf-8-sig")
        log.info("Tải %d bản ghi từ %s", len(self.df), self.data_path)
        return self

    # ── Tính đặc trưng ────────────────────────────────────────────────────────
    def _engineer_features(self) -> pd.DataFrame:
        df = self.df.copy()

        # Rolling windows
        df["speed_ma5"]   = df["avg_speed_kmh"].rolling(5,  min_periods=1).mean()
        df["speed_ma15"]  = df["avg_speed_kmh"].rolling(15, min_periods=1).mean()
        df["wait_ma5"]    = df["avg_wait_s"].rolling(5,  min_periods=1).mean()
        df["vehicle_ma5"] = df["total_vehicles"].rolling(5, min_periods=1).mean()

        # Tỉ lệ thay đổi tốc độ
        df["speed_change_pct"] = df["avg_speed_kmh"].pct_change().fillna(0) * 100

        # Hiệu suất giao thông (throughput / vehicles)
        df["efficiency"] = np.where(
            df["total_vehicles"] > 0,
            df["throughput"] / df["total_vehicles"],
            0,
        )

        # Thời điểm trong ngày từ sim_time_s
        df["hour_of_day"] = (df["sim_time_s"] / 3600).astype(int) % 24

        # Áp lực giao thông
        df["traffic_pressure"] = (
            df["total_vehicles"] * df["avg_wait_s"] / (df["avg_speed_kmh"] + 1)
        ).round(3)

        return df

    # ── Chẩn đoán đa tầng ────────────────────────────────────────────────────
    def diagnose(self) -> "TrafficDiagnosis":
        df = self._engineer_features()

        # ─ 1. Phân loại trạng thái tổng thể ─
        df["state_by_speed"]  = df["avg_speed_kmh"].apply(self._classify_by_speed)
        df["state_by_wait"]   = df["avg_wait_s"].apply(self._classify_by_wait)
        df["state_by_density"]= df["congestion_index"].apply(self._classify_by_index) \
            if "congestion_index" in df.columns else "Bình thường"

        # Kết hợp bằng voting (lấy trạng thái nặng nhất)
        df["final_state"] = df.apply(self._consensus_state, axis=1)
        df["severity"]    = df["final_state"].map(
            {"Thông thoáng": 0, "Bình thường": 1, "Chậm": 2, "Tắc nghẽn": 3}
        )

        # ─ 2. Phát hiện sự kiện bất thường ─
        df["is_anomaly"] = self._detect_anomalies(df)

        # ─ 3. Dự đoán trạng thái tiếp theo (look-ahead đơn giản) ─
        df["predicted_next_state"] = df["final_state"].shift(-1).fillna(df["final_state"])

        # ─ 4. Khuyến nghị hành động ─
        df["recommendation"] = df["final_state"].apply(self._recommend)

        self.results = df
        log.info(
            "Chẩn đoán hoàn tất. Phân bố: %s",
            df["final_state"].value_counts().to_dict(),
        )
        return self

    # ── Hàm phân loại ────────────────────────────────────────────────────────
    @staticmethod
    def _classify_by_speed(speed: float) -> str:
        if speed >= 40: return "Thông thoáng"
        if speed >= 25: return "Bình thường"
        if speed >= 10: return "Chậm"
        return "Tắc nghẽn"

    @staticmethod
    def _classify_by_wait(wait: float) -> str:
        if wait < 15:  return "Thông thoáng"
        if wait < 40:  return "Bình thường"
        if wait < 90:  return "Chậm"
        return "Tắc nghẽn"

    @staticmethod
    def _classify_by_index(idx: float) -> str:
        if idx < 0.2:  return "Thông thoáng"
        if idx < 0.45: return "Bình thường"
        if idx < 0.7:  return "Chậm"
        return "Tắc nghẽn"

    @staticmethod
    def _consensus_state(row) -> str:
        votes = [
            row.get("state_by_speed",  "Bình thường"),
            row.get("state_by_wait",   "Bình thường"),
            row.get("state_by_density","Bình thường"),
        ]
        order = {"Thông thoáng": 0, "Bình thường": 1, "Chậm": 2, "Tắc nghẽn": 3}
        return max(votes, key=lambda s: order.get(s, 1))

    @staticmethod
    def _detect_anomalies(df: pd.DataFrame) -> pd.Series:
        """Z-score trên tốc độ + thời gian chờ."""
        z_speed = (df["avg_speed_kmh"] - df["avg_speed_kmh"].mean()) / (df["avg_speed_kmh"].std() + 1e-9)
        z_wait  = (df["avg_wait_s"]    - df["avg_wait_s"].mean())    / (df["avg_wait_s"].std()    + 1e-9)
        return ((z_speed.abs() > 2.5) | (z_wait.abs() > 2.5)).astype(int)

    @staticmethod
    def _recommend(state: str) -> str:
        return {
            "Thông thoáng": "Duy trì chu kỳ đèn hiện tại",
            "Bình thường":  "Theo dõi, tối ưu pha đèn nếu cần",
            "Chậm":         "Kéo dài pha xanh hướng đông-tây, cảnh báo tài xế",
            "Tắc nghẽn":    "Điều tiết khẩn cấp, ưu tiên hướng thoát, gửi cảnh báo",
        }.get(state, "Không xác định")

    # ── Xuất kết quả ─────────────────────────────────────────────────────────
    def export_csv(self) -> Path:
        out = DATA_DIR / "diagnosis_results.csv"
        self.results.to_csv(out, index=False, encoding="utf-8-sig")
        log.info("✅ Kết quả chẩn đoán → %s", out)
        return out

    def export_summary(self) -> Path:
        df = self.results
        lines = [
            "=" * 60,
            "   BÁO CÁO CHẨN ĐOÁN GIAO THÔNG NGÃ TƯ THỦ ĐỨC",
            "=" * 60,
            f"Tổng số mẫu phân tích : {len(df)}",
            f"Thời gian mô phỏng    : {df['sim_time_s'].max()} giây",
            "",
            "── Thống kê tốc độ (km/h) ──",
            f"  Trung bình : {df['avg_speed_kmh'].mean():.2f}",
            f"  Thấp nhất  : {df['avg_speed_kmh'].min():.2f}",
            f"  Cao nhất   : {df['avg_speed_kmh'].max():.2f}",
            "",
            "── Thống kê thời gian chờ (giây) ──",
            f"  Trung bình : {df['avg_wait_s'].mean():.2f}",
            f"  Thấp nhất  : {df['avg_wait_s'].min():.2f}",
            f"  Cao nhất   : {df['avg_wait_s'].max():.2f}",
            "",
            "── Phân bố trạng thái giao thông ──",
        ]
        for state in STATE_ORDER:
            cnt = (df["final_state"] == state).sum()
            pct = cnt / len(df) * 100
            lines.append(f"  {state:<15}: {cnt:4d} mẫu ({pct:5.1f}%)")
        lines += [
            "",
            f"── Bất thường phát hiện : {df['is_anomaly'].sum()} sự kiện",
            "",
            "── Khuyến nghị ──",
            f"  {df['recommendation'].value_counts().idxmax()}",
            "=" * 60,
        ]
        out = DATA_DIR / "diagnosis_summary.txt"
        out.write_text("\n".join(lines), encoding="utf-8")
        log.info("✅ Tóm tắt → %s", out)
        return out

    # ── Vẽ biểu đồ báo cáo ───────────────────────────────────────────────────
    def plot_report(self) -> Path:
        df = self.results
        sns.set_theme(style="whitegrid", palette="muted")

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            "Báo cáo chẩn đoán giao thông ngã tư Thủ Đức",
            fontsize=16, fontweight="bold", y=0.98,
        )
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        t = df["sim_time_s"] / 60  # phút

        # ── 1. Tốc độ trung bình theo thời gian ──
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(t, df["avg_speed_kmh"], color="#3498db", alpha=0.4, linewidth=0.8, label="Thực tế")
        ax1.plot(t, df["speed_ma15"],    color="#2980b9", linewidth=1.8, label="MA-15")
        ax1.axhline(40, color="#2ecc71", linestyle="--", linewidth=1, label="Ngưỡng thoáng (40)")
        ax1.axhline(10, color="#e74c3c", linestyle="--", linewidth=1, label="Ngưỡng tắc (10)")
        ax1.fill_between(t, df["avg_speed_kmh"], alpha=0.1, color="#3498db")
        ax1.set(title="Tốc độ trung bình theo thời gian", xlabel="Thời gian (phút)", ylabel="km/h")
        ax1.legend(fontsize=8)

        # ── 2. Phân bố trạng thái (pie) ──
        ax2 = fig.add_subplot(gs[0, 2])
        state_counts = df["final_state"].value_counts().reindex(STATE_ORDER, fill_value=0)
        colors = [STATE_COLORS[s] for s in state_counts.index]
        wedges, texts, autotexts = ax2.pie(
            state_counts, labels=state_counts.index,
            colors=colors, autopct="%1.1f%%", startangle=140,
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax2.set_title("Phân bố trạng thái")

        # ── 3. Số lượng xe + tắc nghẽn ──
        ax3 = fig.add_subplot(gs[1, :2])
        color_map = df["final_state"].map(STATE_COLORS).fillna("#95a5a6")
        ax3.bar(t, df["total_vehicles"], color=color_map, width=t.diff().median() * 0.8, alpha=0.85)
        ax3.set(title="Số xe và trạng thái giao thông", xlabel="Thời gian (phút)", ylabel="Số xe")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLORS.values()]
        ax3.legend(handles, STATE_COLORS.keys(), fontsize=8, loc="upper right")

        # ── 4. Thời gian chờ ──
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.fill_between(t, df["avg_wait_s"], alpha=0.3, color="#e67e22")
        ax4.plot(t, df["avg_wait_s"], color="#e67e22", linewidth=1.5)
        ax4.plot(t, df["max_wait_s"], color="#e74c3c", linewidth=1, linestyle="--", alpha=0.7, label="Max")
        ax4.set(title="Thời gian chờ", xlabel="Thời gian (phút)", ylabel="Giây")
        ax4.legend(fontsize=8)

        # ── 5. Chỉ số tắc nghẽn ──
        ax5 = fig.add_subplot(gs[2, :2])
        if "congestion_index" in df.columns:
            ci = df["congestion_index"]
            ax5.fill_between(t, ci, alpha=0.3, color="#9b59b6")
            ax5.plot(t, ci, color="#8e44ad", linewidth=1.5)
            ax5.axhline(0.7, color="#e74c3c", linestyle=":", linewidth=1, label="Ngưỡng nghiêm trọng")
            ax5.axhline(0.45, color="#e67e22", linestyle=":", linewidth=1, label="Ngưỡng chậm")
        ax5.set(title="Chỉ số tắc nghẽn [0–1]", xlabel="Thời gian (phút)", ylabel="Congestion Index")
        ax5.legend(fontsize=8)

        # ── 6. Heatmap bất thường ──
        ax6 = fig.add_subplot(gs[2, 2])
        feature_cols = ["avg_speed_kmh", "avg_wait_s", "total_vehicles", "throughput"]
        feat_df = df[feature_cols].copy()
        feat_df.columns = ["Tốc độ", "Chờ", "Xe", "Throughput"]
        corr = feat_df.corr()
        sns.heatmap(corr, ax=ax6, annot=True, fmt=".2f", cmap="coolwarm",
                    square=True, linewidths=0.5, annot_kws={"size": 8})
        ax6.set_title("Tương quan đặc trưng")

        out = DATA_DIR / "traffic_diagnostic_report.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("✅ Biểu đồ báo cáo → %s", out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def run_diagnosis() -> dict:
    diag = TrafficDiagnosis()
    diag.load().diagnose()
    csv_path  = diag.export_csv()
    summary   = diag.export_summary()
    plot_path = diag.plot_report()
    return {"csv": csv_path, "summary": summary, "plot": plot_path}


if __name__ == "__main__":
    paths = run_diagnosis()
    print("\n📊 Chẩn đoán hoàn tất:")
    for k, v in paths.items():
        print(f"   {k}: {v}")