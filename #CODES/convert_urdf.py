import mujoco

urdf_path = "my_robot.urdf" # Point this to your file
xml_path = "my_robot.xml"

try:
    # Load the URDF into the engine
    model = mujoco.MjModel.from_xml_path(urdf_path)
    
    # Save it out as a native MJCF XML file
    mujoco.mj_saveLastXML(xml_path, model)
    print(f"Success! Model saved to {xml_path}")
    
except Exception as e:
    print(f"URDF Parsing Error: {e}")