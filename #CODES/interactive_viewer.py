import time
import mujoco
import mujoco.viewer

# 1. Point this directly to your newly generated XML file
xml_path = "/home/tahao_mw5iph3/linuxrobotic/mujoco_playground/#CODES/RobotDog/RobotDog.xml"

# Load the raw MuJoCo C++ model and create a fresh state data object
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. Launch the passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("3D UI Engine running. Close the window to exit.")
    
    # 3. Real-time event loop
    while viewer.is_running():
        step_start = time.time()
        
        # Advance the physics engine by one timestep
        mujoco.mj_step(model, data)
        
        # Sync the updated physics state to the 3D UI
        viewer.sync()
        
        # Lock the loop to real-time speed
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)