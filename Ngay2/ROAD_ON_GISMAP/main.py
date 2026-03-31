import os
import sys

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Lỗi: Khai báo SUMO_HOME!")

import traci

# ==========================================
# CẤU HÌNH
# ==========================================
sumoCmd = ["sumo-gui", "-c", "SUMO_ROAD_ON_GITMAP_BUOI2.sumocfg"]
TLS_ID  = "clusterJ10_J11_J15_J16_#4more"

LANES = {
    "E1": ["E1_0", "E1_1", "E1_2"],
    "E3": ["E3_0", "E3_1", "E3_2"],
    "E5": ["E5_0", "E5_1"],
    "E8": ["E8_0", "E8_1"],
}

# Giới hạn thời gian xanh (giây)
GREEN_MIN   = 12
GREEN_MAX   = 70
GREEN_TURN  = 10   # Pha rẽ trái tối thiểu
YELLOW_TIME = 4    # Thời gian vàng (giây) — phải khớp file .net.xml

# Ngưỡng & tham số
NGUONG_UN_TAC  = 5
SCAN_INTERVAL  = 15
EMA_ALPHA      = 0.4
CYCLE_MIN      = 40
CYCLE_MAX      = 180
LECH_RE_TRAI   = 5   # Chênh lệch xe giữa 2 chiều → nghi rẽ trái kẹt

# ==========================================
# TRẠNG THÁI NỘI BỘ
# ==========================================
ema_xe = {"E1": 0.0, "E3": 0.0, "E5": 0.0, "E8": 0.0}

# ==========================================
# KHỞI ĐỘNG & ĐỌC CẤU TRÚC PHA THỰC TẾ
# ==========================================
print("Khởi động SUMO...")
traci.start(sumoCmd)

tls_list = traci.trafficlight.getIDList()
if not tls_list:
    traci.close()
    sys.exit("Không tìm thấy traffic light!")
if TLS_ID not in tls_list:
    TLS_ID = tls_list[0]
    print(f"Dùng TLS: {TLS_ID}")

# Đọc pha thực tế từ file .net.xml
logics     = traci.trafficlight.getAllProgramLogics(TLS_ID)
logic      = logics[0]
PROGRAM_ID = logic.programID
PHASES     = logic.phases
N_PHASES   = len(PHASES)

print(f"\n[CẤU TRÚC PHA — Program '{PROGRAM_ID}']")
for i, ph in enumerate(PHASES):
    tag = "🟡 VÀNG" if ph.duration <= 6 else "🟢 XANH"
    print(f"  [{i}] {tag} | state={ph.state} | duration={ph.duration}s")

# Tự động phân loại pha xanh / vàng từ duration
PHASE_XANH = [i for i, ph in enumerate(PHASES) if ph.duration > 6]
PHASE_VANG = [i for i, ph in enumerate(PHASES) if ph.duration <= 6]
print(f"\n  → Pha xanh hợp lệ : {PHASE_XANH}")
print(f"  → Pha vàng (skip) : {PHASE_VANG}\n")

# Xác định 2 cặp pha đối lập (ngang / dọc) từ danh sách pha xanh
# Giả định thứ tự: [pha_ngang_thang, pha_ngang_re, pha_doc_thang, pha_doc_re]
# Nếu số pha khác 4 → dùng fallback theo index
if len(PHASE_XANH) >= 4:
    PHA_NGANG_THANG = PHASE_XANH[0]
    PHA_NGANG_RE    = PHASE_XANH[1]
    PHA_DOC_THANG   = PHASE_XANH[2]
    PHA_DOC_RE      = PHASE_XANH[3]
elif len(PHASE_XANH) == 2:
    PHA_NGANG_THANG = PHASE_XANH[0]
    PHA_NGANG_RE    = PHASE_XANH[0]   # Không có pha rẽ riêng
    PHA_DOC_THANG   = PHASE_XANH[1]
    PHA_DOC_RE      = PHASE_XANH[1]
else:
    PHA_NGANG_THANG = PHASE_XANH[0]
    PHA_NGANG_RE    = PHASE_XANH[0]
    PHA_DOC_THANG   = PHASE_XANH[0]
    PHA_DOC_RE      = PHASE_XANH[0]

print(f"  Ngang thẳng={PHA_NGANG_THANG} | Ngang rẽ={PHA_NGANG_RE}")
print(f"  Dọc thẳng={PHA_DOC_THANG}   | Dọc rẽ={PHA_DOC_RE}\n")

# ==========================================
# CÁC HÀM TIỆN ÍCH
# ==========================================

def dem_xe_dung(ten_nhanh: str) -> int:
    """Đếm xe đang dừng (v < 0.1 m/s) trên tất cả làn của nhánh."""
    return sum(traci.lane.getLastStepHaltingNumber(l)
               for l in LANES[ten_nhanh])

def cap_nhat_ema(xe_raw: dict) -> dict:
    """Làm mượt EMA để tránh điều chỉnh giật cục."""
    global ema_xe
    for k in ema_xe:
        ema_xe[k] = EMA_ALPHA * xe_raw[k] + (1 - EMA_ALPHA) * ema_xe[k]
    return dict(ema_xe)

def tinh_green_time(xe_nhanh: float, tong_xe: float,
                    la_re_trai: bool = False) -> int:
    """
    Tính thời gian xanh cho 1 nhánh theo tỉ lệ xe thực tế.
    Công thức: g = (C* - n*L) * (xe_nhanh / tong_xe)
    với C* tỉ lệ tuyến tính với tổng nhu cầu.
    """
    k      = (CYCLE_MAX - CYCLE_MIN) / 60.0
    C_star = int(min(CYCLE_MAX, max(CYCLE_MIN, CYCLE_MIN + k * tong_xe)))
    co_ich = max(C_star - len(PHASE_XANH) * YELLOW_TIME, len(PHASE_XANH) * 8)
    ratio  = xe_nhanh / tong_xe if tong_xe > 0 else 0.25
    g      = co_ich * ratio
    g_min  = GREEN_TURN if la_re_trai else GREEN_MIN
    return round(min(GREEN_MAX, max(g_min, g))), C_star

def la_pha_vang() -> bool:
    """Kiểm tra SUMO có đang ở pha vàng không."""
    try:
        return traci.trafficlight.getPhase(TLS_ID) in PHASE_VANG
    except:
        return False

def ap_dung_pha_an_toan(pha_muc_tieu: int, green: int):
    """
    Áp dụng pha an toàn:
    - Không set khi đang vàng
    - Không reset nếu đã đúng pha
    - Dùng setPhase + 1 bước delay trước setPhaseDuration
    """
    if la_pha_vang():
        print("  ⏳ Đang pha vàng → chờ chuyển tiếp tự nhiên")
        return False

    pha_hien = traci.trafficlight.getPhase(TLS_ID)

    if pha_hien == pha_muc_tieu:
        # Đúng pha rồi → chỉ gia hạn thời gian xanh
        traci.trafficlight.setPhaseDuration(TLS_ID, green)
        print(f"  ✅ Giữ Pha {pha_muc_tieu} | Gia hạn → {green}s")
    else:
        # Chuyển pha mới → SUMO cần 1 bước để nhận
        traci.trafficlight.setPhase(TLS_ID, pha_muc_tieu)
        print(f"  🔄 Pha {pha_hien} → Pha {pha_muc_tieu} | Xanh sẽ = {green}s")
        # Trả về green để vòng lặp chính set duration ở bước tiếp theo
        return green

    return False

def log_bang(step, xe, ema_s, C_star, pha_uu):
    tong = sum(ema_s.values())
    print(f"\n{'═'*58}")
    print(f"  ADAPTIVE CONTROL  |  Bước {step}s  |  C* = {C_star}s")
    print(f"{'─'*58}")
    print(f"  {'Nhánh':<6}{'Xe thực':>8}{'EMA':>8}{'Tỉ lệ':>8}{'Làn':>6}")
    print(f"{'─'*58}")
    for nhanh in ["E1","E3","E5","E8"]:
        tl  = ema_s[nhanh] / tong * 100 if tong > 0 else 0
        print(f"  {nhanh:<6}{xe[nhanh]:>8}{ema_s[nhanh]:>8.1f}{tl:>7.1f}%"
              f"{len(LANES[nhanh]):>6}")
    print(f"{'─'*58}")
    print(f"  Pha ưu tiên: {pha_uu}  |  Tổng xe: {sum(xe.values())}")
    print(f"{'═'*58}")

# ==========================================
# VÒNG LẶP CHÍNH
# ==========================================
step             = 0
pending_green    = None   # Thời gian xanh chờ set sau 1 bước
pending_phase    = None   # Pha chờ xác nhận
lich_su          = []

try:
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # ── Xác nhận setPhaseDuration sau 1 bước delay ──────────────
        if pending_phase is not None:
            pha_hien = traci.trafficlight.getPhase(TLS_ID)
            if pha_hien == pending_phase:
                traci.trafficlight.setPhaseDuration(TLS_ID, pending_green)
                print(f"  ✅ Đã set duration Pha {pending_phase} = {pending_green}s")
                pending_phase = None
                pending_green = None

        # ── Quét mỗi SCAN_INTERVAL giây ─────────────────────────────
        if step > 0 and step % SCAN_INTERVAL == 0:

            if la_pha_vang():
                print(f"[Bước {step}s] Pha vàng → bỏ qua")
            else:
                # BƯỚC 1: Thu thập xe thực tế
                xe = {n: dem_xe_dung(n) for n in LANES}

                # BƯỚC 2: Làm mượt EMA
                ema_s   = cap_nhat_ema(xe)
                tong_xe = sum(xe.values())
                max_xe  = max(xe.values())

                if tong_xe == 0 or max_xe <= NGUONG_UN_TAC:
                    print(f"[Bước {step}s] Thoáng ({tong_xe} xe) → giữ chu kỳ.")

                else:
                    # BƯỚC 3: Xác định trục kẹt & pha phù hợp
                    xe_ngang = ema_s["E1"] + ema_s["E3"]
                    xe_doc   = ema_s["E5"] + ema_s["E8"]

                    if xe_ngang >= xe_doc:
                        tong_truc  = xe_ngang
                        la_re_trai = abs(xe["E1"] - xe["E3"]) > LECH_RE_TRAI
                        pha_uu     = PHA_NGANG_RE if la_re_trai else PHA_NGANG_THANG
                        mo_ta      = "Ngang-Rẽ" if la_re_trai else "Ngang-Thẳng"
                    else:
                        tong_truc  = xe_doc
                        la_re_trai = abs(xe["E5"] - xe["E8"]) > LECH_RE_TRAI
                        pha_uu     = PHA_DOC_RE if la_re_trai else PHA_DOC_THANG
                        mo_ta      = "Dọc-Rẽ" if la_re_trai else "Dọc-Thẳng"

                    # BƯỚC 4: Tính thời gian xanh adaptive
                    green, C_star = tinh_green_time(tong_truc, tong_xe, la_re_trai)
                    log_bang(step, xe, ema_s, C_star, pha_uu)
                    print(f"  Hướng kẹt: {mo_ta} | Xanh = {green}s")

                    # BƯỚC 5: Áp dụng an toàn
                    result = ap_dung_pha_an_toan(pha_uu, green)
                    if result:   # Cần delay 1 bước
                        pending_phase = pha_uu
                        pending_green = result

                    lich_su.append({
                        "step": step, "C": C_star,
                        "pha": pha_uu, "xe": tong_xe,
                        "huong": mo_ta
                    })

        step += 1

        # Xuất file nghiệm thu sau 500 bước (nếu tới được bước này)
        if step == 500:
            filename = "nghiem_thu_500.csv"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("step,C,pha,xe,huong\n")
                for r in lich_su:
                    f.write(f"{r['step']},{r['C']},{r['pha']},{r['xe']},{r['huong']}\n")
            print(f"[Nghiệm thu] Đã ghi file: {filename} ({len(lich_su)} bản ghi)")

except Exception as e:
    print(f"\n[KẾT THÚC] {e}")

finally:
    if lich_su:
        # Ghi file nghiệm thu chung sau chạy xong
        filename_all = "nghiem_thu_tong.csv"
        with open(filename_all, "w", encoding="utf-8") as f:
            f.write("step,C,pha,xe,huong\n")
            for r in lich_su:
                f.write(f"{r['step']},{r['C']},{r['pha']},{r['xe']},{r['huong']}\n")
        print(f"[Nghiệm thu tổng] Đã ghi file: {filename_all} ({len(lich_su)} bản ghi)")

        from collections import Counter
        print(f"\n{'═'*45}")
        print("  THỐNG KÊ TOÀN BỘ MÔ PHỎNG")
        print(f"{'═'*45}")
        print(f"  Tổng lần điều chỉnh : {len(lich_su)}")
        chu_ky_tb = sum(r['C'] for r in lich_su) / len(lich_su)
        print(f"  Chu kỳ TB / Min / Max: "
              f"{chu_ky_tb:.1f}s / "
              f"{min(r['C'] for r in lich_su)}s / "
              f"{max(r['C'] for r in lich_su)}s")
        print(f"  Xe TB mỗi lần quét  : "
              f"{sum(r['xe'] for r in lich_su)/len(lich_su):.1f}")
        print(f"\n  Hướng được ưu tiên:")
        for huong, cnt in Counter(r['huong'] for r in lich_su).most_common():
            print(f"    {huong:<15} : {cnt} lần")
    try:
        traci.close()
    except:
        pass

print("\nHoàn tất!")