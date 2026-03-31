# Step 1: Add modules to provide access to specific libraries and functions
import os # Module provides functions to handle file paths, directories, environment variables
import sys # Module provides access to Python-specific system parameters and functions

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME' before running this script.")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if not os.path.isdir(tools):
    sys.exit("SUMO_HOME/tools path does not exist: {}".format(tools))

sys.path.append(tools)

# Step 3: Add Traci module to provide access to specific libraries and functions
from sumolib import checkBinary
import traci

# Step 4: Define Sumo configuration
sumo_binary = checkBinary('sumo-gui')  # 'sumo' nếu bạn chạy không GUI
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, 'Traci.sumocfg')
if not os.path.isfile(config_file):
    sys.exit(f"Cannot access configuration file: {config_file}")

Sumo_config = [
    sumo_binary,
    '-c', config_file,
    '--step-length', '0.05',
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Step 5: Open connection between SUMO and Traci
try:
    traci.start(Sumo_config)
except Exception as e:
    sys.exit(f"Cannot start TraCI: {e}\nCheck SUMO_HOME and your SUMO configuration file path.")

# Step 6: Define Variables
vehicle_speed = 0.0
total_speed = 0.0

# Step 7: Define Functions
def update_vehicle_speed(veh_id):
    """
    Hàm này kiểm tra xem xe có trên bản đồ không.
    Nếu có, in ra vị trí, vận tốc và trả về giá trị vận tốc hiện tại.
    Nếu không, trả về 0.0.
    """
    # Lấy danh sách các xe đang chạy để kiểm tra
    if veh_id in traci.vehicle.getIDList():
        # Lấy các thông số của xe
        speed = traci.vehicle.getSpeed(veh_id)
        pos = traci.vehicle.getPosition(veh_id)
        edge = traci.vehicle.getRoadID(veh_id)
        
        # In kết quả ra màn hình
        print(f"Xe '{veh_id}' | Vận tốc: {speed:.2f} m/s | Tọa độ: ({pos[0]:.2f}, {pos[1]:.2f}) | Đường: {edge}")
        
        return speed
    else:
        # Trả về 0 nếu xe chưa xuất hiện hoặc đã đi mất
        return 0.0
# Step 8: Take simulation steps until there are no more vehicles in the network
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep() # Move simulation forward 1 step
    
    # Lấy danh sách TẤT CẢ các ID xe đang có mặt trên bản đồ ở thời điểm hiện tại
    danh_sach_xe_dang_chay = traci.vehicle.getIDList()
    
    # Dùng vòng lặp for để lấy thông tin của từng chiếc xe một
    for v_id in danh_sach_xe_dang_chay:
        vehicle_speed = update_vehicle_speed(v_id) 
        total_speed = total_speed + vehicle_speed

# In tổng kết sau khi vòng lặp kết thúc
print("-" * 50)
print(f"Mô phỏng hoàn tất! Tổng vận tốc tích lũy: {total_speed:.2f}")

# Step 9: Close connection between SUMO and Traci
traci.close()