import xml.etree.ElementTree as ET
from xml.dom import minidom
import random

# ==========================================
# TEMPLATE 1: CẤU HÌNH THÔNG SỐ (CONFIG)
# ==========================================
TOTAL_TIME = 3600       # Tổng thời gian mô phỏng: 1 giờ = 3600 giây
CAR_COUNT = 500         # Số lượng xe ô tô (phân bổ đều trong 1 giờ)
MOTORBIKE_COUNT = 800   # Số lượng xe máy (chiếm đa số ở Thủ Đức)
BUS_COUNT = 40          # Số lượng xe buýt (tuyến cố định)
TRUCK_COUNT = 60        # Số lượng xe tải (giờ thấp điểm)
AMBULANCE_COUNT = 3     # Số lượng xe cứu thương (xuất hiện ngẫu nhiên)

# ------------------------------------------
# CÁC CẠNH (EDGES) TRÍCH XUẤT TỪ trips_trips.xml
# Đây là các cạnh xuất hiện nhiều nhất trong mạng lưới Thủ Đức,
# đại diện cho các trục đường chính vào/ra ngã tư.
# ------------------------------------------

# Cạnh đầu vào (START): Các đường trục chính dẫn vào ngã tư
# Trích từ top FROM edges trong trips_trips.xml
START_EDGES = [
    "1149832776",       # Trục chính phía Bắc (tần suất cao nhất: 75 trips)
    "1269264065#0",     # Đường vào từ hướng Đông Bắc (45 trips)
    "610137708",        # Đường vào từ hướng Tây (43 trips)
    "1175493775#0",     # Đường vào từ hướng Nam (35 trips)
    "-191937837#0",     # Đường vào từ hướng Đông (34 trips)
    "1213004136",       # Nhánh phụ phía Bắc (30 trips)
    "-606920724#5",     # Đường vào từ hướng Tây Bắc (28 trips)
    "802338945",        # Nhánh phụ phía Đông (25 trips)
]

# Cạnh đầu ra (END): Các đường trục chính rời khỏi ngã tư
# Trích từ top TO edges trong trips_trips.xml
END_EDGES = [
    "404368566",        # Đường ra chính hướng Đông Nam (tần suất cao nhất: 102 trips)
    "404368568",        # Đường ra chính hướng Nam (92 trips)
    "610099998",        # Đường ra hướng Tây Nam (60 trips)
    "722058131",        # Đường ra hướng Bắc (37 trips)
    "606240976",        # Nhánh ra phía Tây (31 trips)
    "-1269264065#0",    # Đường ra hướng Đông Bắc (31 trips)
    "-1175493775#0",    # Đường ra hướng Nam (26 trips)
    "224494489#2",      # Nhánh ra phía Đông (26 trips)
]

# ------------------------------------------
# KỊCH BẢN GIỜ CAO ĐIỂM (PEAK HOUR PROFILE)
# Chia thời gian thành 3 giai đoạn:
#   - Trước cao điểm (0–900s):   lưu lượng thấp
#   - Cao điểm sáng (900–2700s): lưu lượng cao nhất
#   - Sau cao điểm (2700–3600s): lưu lượng giảm dần
# ------------------------------------------
PEAK_START = 900
PEAK_END   = 2700


# ==========================================
# TEMPLATE 2: HÀM TẠO NỘI DUNG XML (XML GENERATOR)
# ==========================================
def create_scenario_xml(filename="thuduc_custom.rou.xml"):
    """
    Tạo kịch bản giao thông cho ngã tư Thủ Đức với:
    - Nhiều loại phương tiện: xe máy, ô tô con, xe buýt, xe tải, xe cứu thương
    - Lưu lượng biến đổi theo giờ (cao điểm sáng)
    - Xe buýt đi theo tuyến cố định
    - Xe cứu thương xuất hiện ngẫu nhiên và có quyền ưu tiên
    """
    routes = ET.Element("routes")

    # ------------------------------------------------------------------
    # 1. ĐỊNH NGHĨA CÁC LOẠI PHƯƠNG TIỆN (vType)
    # ------------------------------------------------------------------

    # Xe máy – chiếm tỉ lệ lớn nhất trong giao thông đô thị Việt Nam
    ET.SubElement(routes, "vType",
        id="motorbike",
        vClass="motorcycle",
        maxSpeed="12.0",
        accel="2.0",
        decel="4.0",
        length="2.2",
        minGap="1.0",
        color="0,128,255",
        guiShape="motorcycle")

    # Xe ô tô con – phương tiện phổ biến thứ hai
    ET.SubElement(routes, "vType",
        id="car",
        vClass="passenger",
        maxSpeed="15.0",
        accel="2.6",
        decel="4.5",
        length="5.0",
        minGap="2.5",
        color="200,200,200",
        guiShape="passenger")

    # Xe buýt – phương tiện công cộng, dừng đỗ thường xuyên
    ET.SubElement(routes, "vType",
        id="bus",
        vClass="bus",
        maxSpeed="10.0",
        accel="1.2",
        decel="3.0",
        length="12.0",
        minGap="3.0",
        color="255,165,0",
        guiShape="bus")

    # Xe tải – kích thước lớn, chạy chậm hơn
    ET.SubElement(routes, "vType",
        id="truck",
        vClass="truck",
        maxSpeed="10.0",
        accel="1.0",
        decel="3.5",
        length="8.0",
        minGap="3.0",
        color="139,90,43",
        guiShape="truck")

    # Xe cứu thương – vClass="emergency" cho phép vượt đèn đỏ và ưu tiên làn
    ET.SubElement(routes, "vType",
        id="ambulance",
        vClass="emergency",
        maxSpeed="25.0",
        accel="3.0",
        decel="5.0",
        length="6.5",
        minGap="2.0",
        speedFactor="1.5",
        color="255,0,0",
        guiShape="emergency")

    # ------------------------------------------------------------------
    # 2. XE CỨU THƯƠNG (AMBULANCE) – xuất hiện ngẫu nhiên
    # ------------------------------------------------------------------
    # Rải đều trong suốt 1 giờ, nhưng tránh trùng thời điểm
    ambulance_times = sorted(random.sample(range(0, TOTAL_TIME), AMBULANCE_COUNT))

    for i, t in enumerate(ambulance_times):
        frm = random.choice(START_EDGES)
        to  = random.choice([e for e in END_EDGES if e != frm])
        trip = ET.SubElement(routes, "trip",
            id=f"amb_{i}",
            type="ambulance",
            depart=str(t))
        trip.set("from", frm)
        trip.set("to", to)

    # ------------------------------------------------------------------
    # 3. LUỒNG XE BUÝT (BUS FLOWS) – 2 tuyến cố định
    #    Tuyến A: Bắc–Nam   (START_EDGES[0] → END_EDGES[1])
    #    Tuyến B: Đông–Tây  (START_EDGES[2] → END_EDGES[2])
    # ------------------------------------------------------------------
    bus_per_route = BUS_COUNT // 2

    bus_routes = [
        ("bus_routeA", START_EDGES[0], END_EDGES[1]),   # Tuyến Bắc–Nam
        ("bus_routeB", START_EDGES[2], END_EDGES[2]),   # Tuyến Đông–Tây
    ]
    for route_id, frm, to in bus_routes:
        flow = ET.SubElement(routes, "flow",
            id=route_id,
            type="bus",
            begin="0",
            end=str(TOTAL_TIME),
            number=str(bus_per_route))
        flow.set("from", frm)
        flow.set("to", to)

    # ------------------------------------------------------------------
    # 4. LUỒNG XE TẢI (TRUCK FLOWS) – chủ yếu ngoài giờ cao điểm
    #    Xe tải tập trung trong khoảng 0–900s và 2700–3600s
    # ------------------------------------------------------------------
    truck_per_flow = TRUCK_COUNT // len(START_EDGES)
    for i, (frm, to) in enumerate(zip(START_EDGES, END_EDGES)):
        # Trước cao điểm
        flow_before = ET.SubElement(routes, "flow",
            id=f"truck_flow_before_{i}",
            type="truck",
            begin="0",
            end=str(PEAK_START),
            number=str(truck_per_flow // 2))
        flow_before.set("from", frm)
        flow_before.set("to", to)

        # Sau cao điểm
        flow_after = ET.SubElement(routes, "flow",
            id=f"truck_flow_after_{i}",
            type="truck",
            begin=str(PEAK_END),
            end=str(TOTAL_TIME),
            number=str(truck_per_flow // 2))
        flow_after.set("from", frm)
        flow_after.set("to", to)

    # ------------------------------------------------------------------
    # 5. LUỒNG XE Ô TÔ (CAR FLOWS) – 3 giai đoạn theo giờ cao điểm
    # ------------------------------------------------------------------
    # Phân bổ: 20% trước cao điểm, 60% trong cao điểm, 20% sau cao điểm
    car_per_flow_total = CAR_COUNT // len(START_EDGES)
    car_before = int(car_per_flow_total * 0.20)
    car_peak   = int(car_per_flow_total * 0.60)
    car_after  = car_per_flow_total - car_before - car_peak

    for i, (frm, to) in enumerate(zip(START_EDGES, END_EDGES)):
        # Trước cao điểm (0 → 900s)
        f = ET.SubElement(routes, "flow",
            id=f"car_flow_before_{i}", type="car",
            begin="0", end=str(PEAK_START), number=str(car_before))
        f.set("from", frm); f.set("to", to)

        # Giờ cao điểm (900 → 2700s)
        f = ET.SubElement(routes, "flow",
            id=f"car_flow_peak_{i}", type="car",
            begin=str(PEAK_START), end=str(PEAK_END), number=str(car_peak))
        f.set("from", frm); f.set("to", to)

        # Sau cao điểm (2700 → 3600s)
        f = ET.SubElement(routes, "flow",
            id=f"car_flow_after_{i}", type="car",
            begin=str(PEAK_END), end=str(TOTAL_TIME), number=str(car_after))
        f.set("from", frm); f.set("to", to)

    # ------------------------------------------------------------------
    # 6. LUỒNG XE MÁY (MOTORBIKE FLOWS) – 3 giai đoạn, tỉ lệ cao nhất
    # ------------------------------------------------------------------
    # Phân bổ: 15% trước, 70% trong cao điểm, 15% sau
    mb_per_flow_total = MOTORBIKE_COUNT // len(START_EDGES)
    mb_before = int(mb_per_flow_total * 0.15)
    mb_peak   = int(mb_per_flow_total * 0.70)
    mb_after  = mb_per_flow_total - mb_before - mb_peak

    for i, (frm, to) in enumerate(zip(START_EDGES, END_EDGES)):
        # Trước cao điểm (0 → 900s)
        f = ET.SubElement(routes, "flow",
            id=f"mb_flow_before_{i}", type="motorbike",
            begin="0", end=str(PEAK_START), number=str(mb_before))
        f.set("from", frm); f.set("to", to)

        # Giờ cao điểm (900 → 2700s)
        f = ET.SubElement(routes, "flow",
            id=f"mb_flow_peak_{i}", type="motorbike",
            begin=str(PEAK_START), end=str(PEAK_END), number=str(mb_peak))
        f.set("from", frm); f.set("to", to)

        # Sau cao điểm (2700 → 3600s)
        f = ET.SubElement(routes, "flow",
            id=f"mb_flow_after_{i}", type="motorbike",
            begin=str(PEAK_END), end=str(TOTAL_TIME), number=str(mb_after))
        f.set("from", frm); f.set("to", to)

    # ==========================================
    # TEMPLATE 3: XUẤT FILE VÀ ĐỊNH DẠNG (EXPORT)
    # ==========================================
    xmlstr = minidom.parseString(ET.tostring(routes)).toprettyxml(indent="    ")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(xmlstr)

    # Tính tổng số phương tiện
    total_cars       = car_per_flow_total * len(START_EDGES)
    total_motorbikes = mb_per_flow_total * len(START_EDGES)
    total_buses      = bus_per_route * 2
    total_trucks     = truck_per_flow * len(START_EDGES)

    print(f"\n{'='*50}")
    print(f"  Kịch bản giao thông ngã tư Thủ Đức")
    print(f"{'='*50}")
    print(f"  File tạo ra    : {filename}")
    print(f"  Thời gian      : {TOTAL_TIME}s ({TOTAL_TIME//3600}h)")
    print(f"  Giờ cao điểm   : {PEAK_START}s – {PEAK_END}s")
    print(f"{'='*50}")
    print(f"  Xe máy         : ~{total_motorbikes} xe")
    print(f"  Ô tô con       : ~{total_cars} xe")
    print(f"  Xe buýt        : ~{total_buses} xe")
    print(f"  Xe tải         : ~{total_trucks} xe")
    print(f"  Xe cứu thương  : {AMBULANCE_COUNT} xe")
    print(f"  Tổng phương tiện: ~{total_motorbikes + total_cars + total_buses + total_trucks + AMBULANCE_COUNT} xe")
    print(f"{'='*50}\n")


# ==========================================
# CHẠY CHƯƠNG TRÌNH (MAIN)
# ==========================================
if __name__ == "__main__":
    random.seed(42)  # Đặt seed cố định để kết quả có thể tái tạo
    create_scenario_xml("thuduc_custom.rou.xml")