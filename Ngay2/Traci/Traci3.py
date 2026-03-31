# Step 1: Add modules to provide access to specific libraries and functions
import os
import sys

# Step 2: establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' not in os.environ:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if not os.path.isdir(tools):
    sys.exit(f"SUMO_HOME/tools path does not exist: {tools}")

sys.path.append(tools)

# Step 3: Add Traci module and sumolib
from sumolib import checkBinary
import traci

# Step 4: Define Sumo configuration
sumo_binary = checkBinary('sumo-gui')
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, 'Traci.sumocfg')
if not os.path.isfile(config_file):
    sys.exit(f"Cannot access configuration file: {config_file}")

sumo_args = [
    sumo_binary,
    '-c', config_file,
    '--step-length', '0.05',
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Step 5: Open connection between SUMO and TraCI
try:
    traci.start(sumo_args)
except Exception as e:
    sys.exit(f"Cannot start TraCI: {e}")

# Step 6: Define Variables
total_speed = 0.0
total_count = 0
step_count = 0

# Step 7: Main loop: collect live vehicle speeds per step
try:
    while True:
        remaining = traci.simulation.getMinExpectedNumber()
        if remaining <= 0:
            print(f"Simulation ended (remaining vehicles expected = {remaining}).")
            break

        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError as fatal:
            print(f"TraCI connection terminated by SUMO: {fatal}")
            break

        vehicle_ids = traci.vehicle.getIDList()
        if vehicle_ids:
            step_sum = 0.0
            for vid in vehicle_ids:
                speed = traci.vehicle.getSpeed(vid)
                step_sum += speed
                total_speed += speed
                total_count += 1
            step_avg = step_sum / len(vehicle_ids)
            print(f"Step {step_count:03d}: {len(vehicle_ids)} vehicles, step avg speed = {step_avg:.2f} m/s")
        else:
            print(f"Step {step_count:03d}: no vehicles")

        step_count += 1

finally:
    # Step 9: Close connection
    traci.close()

# Print overall average speed
if total_count > 0:
    overall_avg_speed = total_speed / total_count
    print(f"Overall average speed across all vehicle measurements: {overall_avg_speed:.2f} m/s")
else:
    print("No vehicle speed data collected.")
