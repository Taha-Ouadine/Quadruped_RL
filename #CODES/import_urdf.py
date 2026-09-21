import argparse
import shutil
import re
from pathlib import Path
import mujoco

def main():
    parser = argparse.ArgumentParser(description="Import SolidWorks URDF from Windows to WSL.")
    parser.add_argument("win_path", help="Windows path to the exported URDF folder (e.g., E:\\...)")
    parser.add_argument("name", help="Name for the clean Linux folder (e.g. my_dog)")
    args = parser.parse_args()

    # 1. Convert Windows path to WSL path (E:\folder -> /mnt/e/folder)
    win_path = args.win_path.strip("'\"")
    if ":" in win_path:
        drive, rest = win_path.split(":", 1)
        rest = rest.replace("\\", "/").lstrip("/")
        wsl_source = Path(f"/mnt/{drive.lower()}/{rest}")
    else:
        wsl_source = Path(win_path)

    if not wsl_source.exists():
        print(f"Error: Could not find source directory at {wsl_source}")
        return

    # 2. Create the clean, flat local directory
    local_dest = Path.cwd() / args.name
    local_dest.mkdir(parents=True, exist_ok=True)
    print(f"Importing {wsl_source} -> {local_dest}...")

    # 3. Locate the source URDF
    wsl_urdf_dir = wsl_source / "urdf"
    urdf_files = list(wsl_urdf_dir.glob("*.urdf"))
    if not urdf_files:
        print(f"Error: No .urdf file found in {wsl_urdf_dir}")
        return
        
    wsl_urdf_file = urdf_files[0]
    local_urdf_file = local_dest / wsl_urdf_file.name

    # 4. Clean up the URDF content to use bare filenames
    print(f"Cleaning mesh paths in {wsl_urdf_file.name}...")
    content = wsl_urdf_file.read_text()
    
    # Strip the entire SolidWorks package URI, leaving just the filename (e.g., "base_link.STL")
    clean_content = re.sub(r'package://[^/]+/meshes/', '', content)
    local_urdf_file.write_text(clean_content)

    # 5. Copy all STLs directly into the same folder as the URDF
    wsl_meshes_dir = wsl_source / "meshes"
    if wsl_meshes_dir.exists():
        print("Copying STLs directly to root...")
        for stl in wsl_meshes_dir.glob("*.STL"):
            shutil.copy2(stl, local_dest / stl.name)

    # 6. Convert to MuJoCo XML with automated physics sanitization
    xml_path = local_dest / f"{args.name}.xml"
    print(f"Converting to MuJoCo MJCF format...")
    
    def compile_model():
        model = mujoco.MjModel.from_xml_path(str(local_urdf_file))
        mujoco.mj_saveLastXML(str(xml_path), model)
        print(f"\nSuccess! Flat 3D model ready at: {xml_path}")

    try:
        compile_model()
    except Exception as e:
        print(f"\nCompilation failed: {e}")
        print("Running automated URDF physical value sanitizer...")
        
        content = local_urdf_file.read_text()
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
            local_urdf_file.write_text(content)
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