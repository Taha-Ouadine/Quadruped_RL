import mujoco
import re

urdf_path = "/home/tahao_mw5iph3/linuxrobotic/mujoco_playground/#CODES/RobotDog/Robot Final.SLDASM.urdf"
xml_path = "/home/tahao_mw5iph3/linuxrobotic/mujoco_playground/#CODES/RobotDog/RobotDog.xml"

def main():
    print(f"Converting to MuJoCo MJCF format...")
    
    def compile_model():
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
        mujoco.mj_saveLastXML(str(xml_path), model)
        print(f"\nSuccess! Flat 3D model ready at: {xml_path}")

    try:
        compile_model()
    except Exception as e:
        print(f"\nCompilation failed: {e}")
        print("Running automated URDF physical value sanitizer...")
        
        content = urdf_path.read_text()
        original_content = content
        
        # Regex callback: Parses the string to a float, bumps it if it's too small
        def bump_tiny(match):
            try:
                val = float(match.group(2))
                if val < 0.0001:
                    return f'{match.group(1)}="0.001"'
            except ValueError:
                pass
            return match.group(0)

        # Fix 1: The Zero-Axis Bug
        content = re.sub(r'axis\s+xyz=["\']0(\.0+)? 0(\.0+)? 0(\.0+)?["\']', 'axis xyz="0 0 1"', content)
        
        # Fix 2: Tiny Mass values
        content = re.sub(r'(mass\s+value)=["\']([^"\']+)["\']', bump_tiny, content)
        
        # Fix 3: Tiny Inertia main diagonals (ixx, iyy, izz)
        # (Off-diagonals like ixy can legally be zero, but diagonals must be positive)
        for attr in ['ixx', 'iyy', 'izz']:
            content = re.sub(rf'({attr})=["\']([^"\']+)["\']', bump_tiny, content)

        if content != original_content:
            urdf_path.write_text(content)
            print("Sanitization complete. Bumbed microscopic values to 0.001. Retrying compilation...")
            try:
                compile_model()
            except Exception as e2:
                print(f"URDF Parsing Error after applying fixes: {e2}")
        else:
            print("Could not find any obvious zero/tiny values to fix.")

    # Optionaly delete the urdf file
    # local_urdf_file.unlink()

if __name__ == "__main__":
    main()