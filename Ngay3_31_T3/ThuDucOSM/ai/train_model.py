"""
train_model.py
==============
Huấn luyện mô hình dự đoán tình trạng giao thông ngã tư Thủ Đức.
Đầu vào : data/diagnosis_results.csv  (hoặc simulation_data.csv)
Đầu ra  : model/traffic_model.pkl
           model/scaler.pkl
           model/label_encoder.pkl
           data/training_results.csv
           data/model_evaluation.png

Yêu cầu:
    pip install scikit-learn pandas numpy matplotlib seaborn joblib
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Đường dẫn ─────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DATA_DIR / "training.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Tên file model ─────────────────────────────────────────────────────────────
MODEL_PATH   = MODEL_DIR / "traffic_model.pkl"
SCALER_PATH  = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# ── Tên cột ───────────────────────────────────────────────────────────────────
TARGET_COL = "final_state"
FEATURE_COLS = [
    "total_vehicles",
    "avg_speed_kmh",
    "avg_wait_s",
    "max_wait_s",
    "throughput",
    "departed",
    "speed_ma5",
    "wait_ma5",
    "vehicle_ma5",
    "efficiency",
    "traffic_pressure",
    "congestion_index",
]


# ══════════════════════════════════════════════════════════════════════════════
# Lớp huấn luyện
# ══════════════════════════════════════════════════════════════════════════════
class TrafficModelTrainer:
    """Huấn luyện và đánh giá mô hình phân loại trạng thái giao thông."""

    def __init__(self):
        self.df_raw:   pd.DataFrame = None
        self.df_feat:  pd.DataFrame = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.scaler   = StandardScaler()
        self.encoder  = LabelEncoder()
        self.model    = None
        self.report:  dict = {}

    # ── Tải dữ liệu ──────────────────────────────────────────────────────────
    def load(self) -> "TrafficModelTrainer":
        # Ưu tiên dùng file đã chẩn đoán
        for fname in ["diagnosis_results.csv", "simulation_data.csv"]:
            path = DATA_DIR / fname
            if path.exists():
                self.df_raw = pd.read_csv(path, encoding="utf-8-sig")
                log.info("Tải %d bản ghi từ %s", len(self.df_raw), path)
                return self
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu. Chạy sumo_simulation.py và diagnosis.py trước."
        )

    # ── Chuẩn bị đặc trưng ───────────────────────────────────────────────────
    def prepare_features(self) -> "TrafficModelTrainer":
        df = self.df_raw.copy()

        # Tính lại các đặc trưng nếu chưa có
        if "speed_ma5" not in df.columns:
            df["speed_ma5"]   = df["avg_speed_kmh"].rolling(5,  min_periods=1).mean()
        if "wait_ma5" not in df.columns:
            df["wait_ma5"]    = df["avg_wait_s"].rolling(5,  min_periods=1).mean()
        if "vehicle_ma5" not in df.columns:
            df["vehicle_ma5"] = df["total_vehicles"].rolling(5, min_periods=1).mean()
        if "efficiency" not in df.columns:
            df["efficiency"]  = np.where(
                df["total_vehicles"] > 0,
                df["throughput"] / df["total_vehicles"], 0,
            )
        if "traffic_pressure" not in df.columns:
            df["traffic_pressure"] = (
                df["total_vehicles"] * df["avg_wait_s"] / (df["avg_speed_kmh"] + 1)
            )
        if "congestion_index" not in df.columns:
            norm_speed = 1 - (df["avg_speed_kmh"].clip(0, 50) / 50)
            norm_wait  = df["avg_wait_s"].clip(0, 120) / 120
            df["congestion_index"] = norm_speed * 0.6 + norm_wait * 0.4

        # Đặt nhãn mục tiêu
        if TARGET_COL not in df.columns:
            # Fallback từ traffic_state
            if "traffic_state" in df.columns:
                df[TARGET_COL] = df["traffic_state"]
            else:
                df[TARGET_COL] = df["avg_speed_kmh"].apply(self._classify_speed)

        # Loại bỏ hàng thiếu
        avail_features = [c for c in FEATURE_COLS if c in df.columns]
        df = df[avail_features + [TARGET_COL]].dropna()
        log.info("Đặc trưng sử dụng (%d): %s", len(avail_features), avail_features)

        self.df_feat = df
        self.avail_features = avail_features
        return self

    @staticmethod
    def _classify_speed(speed: float) -> str:
        if speed >= 40: return "Thông thoáng"
        if speed >= 25: return "Bình thường"
        if speed >= 10: return "Chậm"
        return "Tắc nghẽn"

    # ── Chia train/test ───────────────────────────────────────────────────────
    def split(self, test_size: float = 0.2) -> "TrafficModelTrainer":
        df = self.df_feat
        X  = df[self.avail_features].values
        y  = self.encoder.fit_transform(df[TARGET_COL])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        self.X_train = self.scaler.fit_transform(X_tr)
        self.X_test  = self.scaler.transform(X_te)
        self.y_train = y_tr
        self.y_test  = y_te

        log.info(
            "Phân chia: train=%d, test=%d | Nhãn: %s",
            len(y_tr), len(y_te), list(self.encoder.classes_),
        )
        return self

    # ── Xây dựng mô hình tổng hợp ────────────────────────────────────────────
    def build_model(self) -> "TrafficModelTrainer":
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=5,
            subsample=0.8, random_state=42,
        )
        lr = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, C=1.0,
        )
        self.model = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
            voting="soft",
        )
        log.info("Mô hình: VotingClassifier (RF + GradientBoosting + LogReg)")
        return self

    # ── Huấn luyện ───────────────────────────────────────────────────────────
    def train(self) -> "TrafficModelTrainer":
        log.info("Bắt đầu huấn luyện...")
        self.model.fit(self.X_train, self.y_train)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        self.report["cv_mean"] = cv_scores.mean()
        self.report["cv_std"]  = cv_scores.std()
        log.info(
            "CV accuracy: %.4f ± %.4f",
            cv_scores.mean(), cv_scores.std(),
        )
        return self

    # ── Đánh giá ─────────────────────────────────────────────────────────────
    def evaluate(self) -> "TrafficModelTrainer":
        y_pred = self.model.predict(self.X_test)
        labels = list(self.encoder.classes_)

        report_str = classification_report(self.y_test, y_pred, target_names=labels)
        log.info("\n%s", report_str)

        self.report["test_accuracy"] = (y_pred == self.y_test).mean()
        self.report["classification_report"] = report_str
        self.report["y_pred"]  = y_pred
        self.report["labels"]  = labels

        log.info("Test accuracy: %.4f", self.report["test_accuracy"])
        return self

    # ── Lưu model ─────────────────────────────────────────────────────────────
    def save(self) -> "TrafficModelTrainer":
        joblib.dump(self.model,   MODEL_PATH)
        joblib.dump(self.scaler,  SCALER_PATH)
        joblib.dump(self.encoder, ENCODER_PATH)
        log.info("✅ Model → %s", MODEL_PATH)
        log.info("✅ Scaler → %s", SCALER_PATH)
        log.info("✅ Encoder → %s", ENCODER_PATH)
        return self

    # ── Xuất kết quả CSV ─────────────────────────────────────────────────────
    def export_results(self) -> Path:
        df  = self.df_feat.copy().reset_index(drop=True)
        X   = self.scaler.transform(df[self.avail_features].values)
        preds = self.encoder.inverse_transform(self.model.predict(X))
        proba = self.model.predict_proba(X)

        df["predicted_state"] = preds
        df["prediction_correct"] = (df["predicted_state"] == df[TARGET_COL]).astype(int)
        for i, cls in enumerate(self.encoder.classes_):
            df[f"prob_{cls}"] = proba[:, i].round(4)

        out = DATA_DIR / "training_results.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        log.info("✅ Kết quả huấn luyện → %s", out)
        return out

    # ── Vẽ biểu đồ đánh giá ──────────────────────────────────────────────────
    def plot_evaluation(self) -> Path:
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            "Đánh giá mô hình dự đoán giao thông ngã tư Thủ Đức",
            fontsize=15, fontweight="bold",
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        labels  = self.report["labels"]
        y_pred  = self.report["y_pred"]
        y_test  = self.y_test

        # ── 1. Confusion Matrix ──
        ax1 = fig.add_subplot(gs[0, :2])
        cm  = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=ax1, colorbar=True, cmap="Blues")
        ax1.set_title(
            f"Ma trận nhầm lẫn  |  Test Acc: {self.report['test_accuracy']:.3f}  |  "
            f"CV: {self.report['cv_mean']:.3f}±{self.report['cv_std']:.3f}"
        )

        # ── 2. Phân bố nhãn thực tế ──
        ax2 = fig.add_subplot(gs[0, 2])
        real_counts = pd.Series(self.encoder.inverse_transform(y_test)).value_counts()
        colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
        real_counts.plot(kind="barh", ax=ax2, color=colors[:len(real_counts)])
        ax2.set_title("Phân bố nhãn (test set)")
        ax2.set_xlabel("Số mẫu")

        # ── 3. Feature Importance (từ RF) ──
        ax3 = fig.add_subplot(gs[1, :2])
        try:
            rf_model = self.model.named_estimators_["rf"]
            importances = pd.Series(
                rf_model.feature_importances_, index=self.avail_features
            ).sort_values(ascending=True)
            colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importances)))
            importances.plot(kind="barh", ax=ax3, color=colors_fi)
            ax3.set_title("Tầm quan trọng đặc trưng (Random Forest)")
            ax3.set_xlabel("Importance")
        except Exception:
            ax3.text(0.5, 0.5, "Feature importance\nkhông khả dụng",
                     ha="center", va="center", transform=ax3.transAxes)

        # ── 4. Predicted vs Actual ──
        ax4 = fig.add_subplot(gs[1, 2])
        pred_counts = pd.Series(self.encoder.inverse_transform(y_pred)).value_counts()
        x = range(len(labels))
        width = 0.35
        actual_vals = [real_counts.get(l, 0) for l in labels]
        pred_vals   = [pred_counts.get(l, 0) for l in labels]
        ax4.bar([i - width/2 for i in x], actual_vals, width, label="Thực tế", color="#3498db", alpha=0.8)
        ax4.bar([i + width/2 for i in x], pred_vals,   width, label="Dự đoán", color="#e74c3c", alpha=0.8)
        ax4.set_xticks(list(x))
        ax4.set_xticklabels(labels, rotation=15, fontsize=8)
        ax4.set_title("Thực tế vs Dự đoán")
        ax4.legend(fontsize=8)

        out = DATA_DIR / "model_evaluation.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("✅ Biểu đồ đánh giá → %s", out)
        return out

    # ── Xuất thống kê model ───────────────────────────────────────────────────
    def export_model_stats(self) -> Path:
        stats = {
            "model_type":      "VotingClassifier (RF + GradientBoosting + LogReg)",
            "n_features":      len(self.avail_features),
            "feature_names":   ", ".join(self.avail_features),
            "n_classes":       len(self.encoder.classes_),
            "class_names":     ", ".join(self.encoder.classes_),
            "train_samples":   len(self.y_train),
            "test_samples":    len(self.y_test),
            "test_accuracy":   round(self.report["test_accuracy"], 4),
            "cv_accuracy_mean":round(self.report["cv_mean"], 4),
            "cv_accuracy_std": round(self.report["cv_std"], 4),
        }
        out = DATA_DIR / "model_stats.csv"
        pd.DataFrame([stats]).to_csv(out, index=False, encoding="utf-8-sig")
        log.info("✅ Thống kê model → %s", out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Hàm inference tiện dụng
# ══════════════════════════════════════════════════════════════════════════════
def predict_traffic(vehicles: int, speed_kmh: float, wait_s: float) -> dict:
    """Dự đoán trạng thái giao thông từ 3 thông số cơ bản."""
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # Tính đặc trưng phái sinh đơn giản
    congestion_index = (1 - min(speed_kmh, 50) / 50) * 0.6 + min(wait_s, 120) / 120 * 0.4
    sample = {
        "total_vehicles":   vehicles,
        "avg_speed_kmh":    speed_kmh,
        "avg_wait_s":       wait_s,
        "max_wait_s":       wait_s * 2,
        "throughput":       max(0, vehicles - int(vehicles * wait_s / 300)),
        "departed":         int(vehicles * 1.1),
        "speed_ma5":        speed_kmh,
        "wait_ma5":         wait_s,
        "vehicle_ma5":      vehicles,
        "efficiency":       0.5,
        "traffic_pressure": vehicles * wait_s / (speed_kmh + 1),
        "congestion_index": congestion_index,
    }

    # Lấy đúng thứ tự features (thử load từ model stats nếu có)
    try:
        stats_df = pd.read_csv(DATA_DIR / "model_stats.csv")
        feat_names = stats_df["feature_names"].iloc[0].split(", ")
    except Exception:
        feat_names = list(sample.keys())

    X = np.array([[sample.get(f, 0) for f in feat_names]])
    X_scaled = scaler.transform(X)
    pred  = encoder.inverse_transform(model.predict(X_scaled))[0]
    proba = model.predict_proba(X_scaled)[0]
    confidence = dict(zip(encoder.classes_, proba.round(3)))

    return {"state": pred, "confidence": confidence, "congestion_index": round(congestion_index, 3)}


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def run_training() -> dict:
    trainer = TrafficModelTrainer()
    (
        trainer
        .load()
        .prepare_features()
        .split()
        .build_model()
        .train()
        .evaluate()
        .save()
    )
    results_csv = trainer.export_results()
    plot_path   = trainer.plot_evaluation()
    stats_path  = trainer.export_model_stats()
    return {
        "model":   MODEL_PATH,
        "results": results_csv,
        "plot":    plot_path,
        "stats":   stats_path,
    }


if __name__ == "__main__":
    paths = run_training()
    print("\n🤖 Huấn luyện hoàn tất:")
    for k, v in paths.items():
        print(f"   {k}: {v}")