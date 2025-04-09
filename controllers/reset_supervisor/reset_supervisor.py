from controller import Supervisor
import os
file_path = "/Users/marziehghayour/Library/CloudStorage/GoogleDrive-marzieh.ghayour.na@gmail.com/My Drive/U of A/ECE 720/Project/Simulation/Map2/controllers/city/reset_trigger.txt"
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
print("here1")
robot_node = supervisor.getFromDef("BMW")  # Replace with your robot DEF name
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

while supervisor.step(timestep) != -1:
    # print("here2")
    if os.path.exists(file_path):
        # print("here3")
        with open(file_path, "r") as f:
            # print("here4")
            trigger = f.read().strip()
        if trigger == "reset":
            # print("here5")
            # Reset position and orientation
            # translation_field.setSFVec3f([206.204, -107.615, 0.523752])     #r3 <-- Set to your desired coords
            translation_field.setSFVec3f([149.197, -107.476, 0.33])     #r4 <-- Set to your desired coords
            rotation_field.setSFRotation([0.0, 0, 1, -0.02])  # <-- Optional: reset rotation
            # print("Robot position reset by Supervisor.")
            os.remove(file_path)