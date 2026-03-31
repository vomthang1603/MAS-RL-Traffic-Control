import os
import sys

# 1. Kiểm tra và nạp thư viện TraCI từ SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Lỗi: Vui lòng khai báo biến môi trường SUMO_HOME!")

import traci  # Import thành công giao diện TraCI

# 2. Khai báo lệnh chạy SUMO
# Bạn có thể đổi "sumo-gui" thành "sumo" nếu muốn chạy ngầm (không hiện cửa sổ đồ họa, giúp RL train nhanh hơn)
sumoBinary = "sumo-gui" 
sumoConfig = "MoPhongDothi.sumocfg"
sumoCmd = [sumoBinary, "-c", sumoConfig]

# 3. Khởi động kết nối TraCI
print("Đang khởi động SUMO...")
traci.start(sumoCmd)

# 4. Vòng lặp mô phỏng (Simulation Loop)
step = 0
while step < 1000: # Cho chạy thử 1000 bước (giây)
    traci.simulationStep() # Bấm nút "Next" cho SUMO chạy 1 bước
    
    # Đọc dữ liệu từ SUMO: Lấy tổng số xe đang có mặt trên đường
    so_xe = traci.vehicle.getIDCount()
    
    # In ra terminal để theo dõi
    print(f"Bước {step}: Đang có {so_xe} xe chạy trong mô phỏng")
    
    step += 1

# 5. Đóng kết nối an toàn
traci.close()
print("Đã đóng kết nối TraCI an toàn!")