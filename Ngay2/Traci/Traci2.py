# Step 1: Add modules to provide access to specific libraries and functions
import os # Module provides functions to handle file paths, directories, environment variables
import sys # Module provides access to Python-specific system parameters and functions

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if not os.path.isdir(tools):
    sys.exit("SUMO_HOME/tools path does not exist: {}".format(tools))

sys.path.append(tools)

# Step 3: Add Traci module to provide access to specific libraries and functions
from sumolib import checkBinary
import traci # Module for controlling SUMO simulations via TraCI

# Step 4: Define SUMO configuration
sumo_binary = checkBinary('sumo-gui')
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
    sys.exit(f"Cannot start TraCI: {e}\nCheck SUMO_HOME and SUMO config path.")

# Step 6: Define Variables
vehicle_speed = 0
total_speed = 0
step_count = 0

# Step 7: Define Functions
def process_vehicles():
    vehicle_ids = traci.vehicle.getIDList()
    if not vehicle_ids:
        print("No vehicles in the network this step.")
        return

    # Print header for table display
    print("{:^4} | {:^12} | {:^8} | {:^8} | {:^8} | {:^8}".format("Step", "Vehicle", "Edge", "X", "Y", "Speed"))
    print("-" * 60)

    for vid in vehicle_ids:
        x, y = traci.vehicle.getPosition(vid)
        edge_id = traci.vehicle.getRoadID(vid)
        speed = traci.vehicle.getSpeed(vid)
        print("{:>4} | {:<12} | {:<8} | {:8.2f} | {:8.2f} | {:8.2f}".format(
            step_count, vid, edge_id, x, y, speed
        ))

# Step 8: Take simulation steps until there are no more vehicles in the network
try:
    while True:
        remaining = traci.simulation.getMinExpectedNumber()
        if remaining <= 0:
            print(f"Simulation ended (remaining vehicles expected = {remaining}).")
            break

        try:
            traci.simulationStep()  # Move simulation forward 1 step
        except traci.exceptions.FatalTraCIError as fatal:
            print(f"TraCI connection terminated by SUMO: {fatal}")
            break

        # Here you can decide what to do with simulation data at each step
        process_vehicles()

        step_count += 1
       
finally:
    # Step 9: Close connection between SUMO and Traci
    traci.close()

