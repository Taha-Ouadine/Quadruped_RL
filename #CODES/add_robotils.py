#!/usr/bin/env python3
"""
add_actuators_sensors.py

Automatically adds actuators and sensors to a MuJoCo MJCF file that has
none yet (e.g. one produced by mj_saveLastXML() straight from a URDF import).

Usage:
    python add_actuators_sensors.py my_dog.xml
    python add_actuators_sensors.py my_dog.xml --actuator-type position --kp 50
    python add_actuators_sensors.py my_dog.xml --actuator-type motor --gear 5 --force-range -20 20
    python add_actuators_sensors.py my_dog.xml --no-sensors
    python add_actuators_sensors.py my_dog.xml --no-imu
    python add_actuators_sensors.py my_dog.xml --in-place

What it does:
  1. Walks <worldbody> and finds every real joint (hinge/slide), skipping
     any freejoint (a floating base can't be actuated).
  2. Adds one actuator per joint: motor, position, or velocity (your choice).
  3. Adds jointpos + jointvel sensors for every joint.
  4. Adds an IMU-style site + framequat/gyro/accelerometer sensors on the
     root body (the first body under worldbody), unless --no-imu is passed.
  5. Skips any joint that already has a matching actuator/sensor -- safe
     to re-run after you hand-edit the file.
  6. Writes to <name>_actuated.xml by default so it never clobbers your
     clean import; pass --in-place to overwrite instead.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def find_joints(worldbody):
    """Return [(joint_elem, body_elem), ...] for every real (non-free) joint."""
    joints = []
    for body in worldbody.iter("body"):
        for joint in body.findall("joint"):
            if joint.get("type") == "free":
                continue
            joints.append((joint, body))
    return joints


def get_or_create(parent, tag):
    el = parent.find(tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    return el


def existing_joint_names(section, tags):
    names = set()
    for tag in tags:
        for el in section.findall(tag):
            if el.get("joint"):
                names.add(el.get("joint"))
    return names


def add_actuators(root, joints, actuator_type, gear, kp, kv, force_range):
    actuator_section = get_or_create(root, "actuator")
    already = existing_joint_names(actuator_section, ("motor", "position", "velocity"))

    added = []
    for joint, body in joints:
        jname = joint.get("name")
        if not jname or jname in already:
            continue

        attrs = {"name": f"{jname}_act", "joint": jname}

        if actuator_type == "motor":
            attrs["gear"] = str(gear)
            if force_range:
                attrs["ctrlrange"] = f"{force_range[0]} {force_range[1]}"
                attrs["ctrllimited"] = "true"
            ET.SubElement(actuator_section, "motor", attrs)

        elif actuator_type == "position":
            attrs["kp"] = str(kp)
            jrange = joint.get("range")
            if jrange and joint.get("limited") != "false":
                attrs["ctrlrange"] = jrange
                attrs["ctrllimited"] = "true"
            ET.SubElement(actuator_section, "position", attrs)

        elif actuator_type == "velocity":
            attrs["kv"] = str(kv)
            if force_range:
                attrs["ctrlrange"] = f"{force_range[0]} {force_range[1]}"
                attrs["ctrllimited"] = "true"
            ET.SubElement(actuator_section, "velocity", attrs)

        added.append(jname)

    return added


def add_joint_sensors(root, joints):
    sensor_section = get_or_create(root, "sensor")
    already = existing_joint_names(sensor_section, ("jointpos", "jointvel"))

    added = []
    for joint, body in joints:
        jname = joint.get("name")
        if not jname or jname in already:
            continue
        ET.SubElement(sensor_section, "jointpos", {"name": f"{jname}_pos", "joint": jname})
        ET.SubElement(sensor_section, "jointvel", {"name": f"{jname}_vel", "joint": jname})
        added.append(jname)

    return added


def add_imu(root, worldbody):
    """Attach an IMU site + orientation/gyro/accelerometer sensors to the root body."""
    root_body = worldbody.find("body")
    if root_body is None:
        return None, []

    body_name = root_body.get("name", "root")
    site_name = f"{body_name}_imu"

    if not any(s.get("name") == site_name for s in root_body.findall("site")):
        ET.SubElement(root_body, "site", {"name": site_name, "pos": "0 0 0", "size": "0.01"})

    sensor_section = get_or_create(root, "sensor")
    existing_names = {el.get("name") for el in sensor_section}

    to_add = [
        ("framequat", f"{body_name}_orientation"),
        ("gyro", f"{body_name}_gyro"),
        ("accelerometer", f"{body_name}_accel"),
    ]
    added = []
    for tag, name in to_add:
        if name in existing_names:
            continue
        if tag == "framequat":
            ET.SubElement(sensor_section, tag, {"name": name, "objtype": "site", "objname": site_name})
        else:
            ET.SubElement(sensor_section, tag, {"name": name, "site": site_name})
        added.append(name)

    return body_name, added


def indent(elem, level=0):
    """Pretty-print helper (ElementTree.indent is 3.9+; this works on any version)."""
    i = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i + "  "
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def main():
    parser = argparse.ArgumentParser(description="Add actuators + sensors to an MJCF (MuJoCo XML) file.")
    parser.add_argument("xml_path", help="Path to the compiled MJCF .xml file")
    parser.add_argument("--actuator-type", choices=["motor", "position", "velocity"], default="motor",
                         help="Actuator type to add for every joint (default: motor)")
    parser.add_argument("--gear", type=float, default=1.0, help="Gear ratio for motor actuators (default: 1.0)")
    parser.add_argument("--kp", type=float, default=20.0, help="Position gain for position actuators (default: 20)")
    parser.add_argument("--kv", type=float, default=1.0, help="Velocity gain for velocity actuators (default: 1.0)")
    parser.add_argument("--force-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                         help="ctrlrange for motor/velocity actuators, e.g. --force-range -20 20")
    parser.add_argument("--no-sensors", action="store_true", help="Skip adding jointpos/jointvel sensors")
    parser.add_argument("--no-imu", action="store_true", help="Skip adding IMU site/sensors on the root body")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input file instead of writing *_actuated.xml")
    parser.add_argument("--output", help="Explicit output path (overrides default naming and --in-place)")
    args = parser.parse_args()

    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        print(f"Error: {xml_path} not found.")
        sys.exit(1)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        print("Error: no <worldbody> found -- is this a valid MJCF file?")
        sys.exit(1)

    joints = find_joints(worldbody)
    if not joints:
        print("No actuatable joints found (only a freejoint, or no joints at all).")

    added_act = add_actuators(root, joints, args.actuator_type, args.gear, args.kp, args.kv, args.force_range)
    print(f"Added {len(added_act)} '{args.actuator_type}' actuator(s)"
          + (f": {', '.join(added_act)}" if added_act else " (none new -- already present)."))

    if not args.no_sensors:
        added_sens = add_joint_sensors(root, joints)
        print(f"Added jointpos/jointvel sensors for {len(added_sens)} joint(s).")

    if not args.no_imu:
        body_name, added_imu = add_imu(root, worldbody)
        if body_name and added_imu:
            print(f"Added IMU sensors (orientation/gyro/accel) on root body '{body_name}'.")

    indent(root)

    if args.output:
        out_path = Path(args.output)
    elif args.in_place:
        out_path = xml_path
    else:
        out_path = xml_path.with_name(xml_path.stem + "_actuated.xml")

    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()