import subprocess
import sys

#--------------------------------------------------------------------
ORIGIN_PATH = r"E:\tahao_mw5iph3\Desktop\Robotic\Solid\Assem1.SLDASM"
ROBOT_NAME = "SpiderBot"
#--------------------------------------------------------------------

import_script = "/home/tahao_mw5iph3/linuxrobotic/mujoco_playground/#CODES/import_urdf.py"
utils_script = "/home/tahao_mw5iph3/linuxrobotic/mujoco_playground/#CODES/add_robotils.py"

result = subprocess.run( [sys.executable, import_script, ORIGIN_PATH, ROBOT_NAME],
                        capture_output=True, text=True )

assert result.returncode == 0

result = subprocess.run( [sys.executable, utils_script, ORIGIN_PATH, ROBOT_NAME],
                        capture_output=True, text=True )