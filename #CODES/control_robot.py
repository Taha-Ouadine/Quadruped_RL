#!/usr/bin/env python3
"""
control_robot.py

Open your actuated MJCF model and drive the actuators to confirm
everything is wired up correctly.

Two modes:

  1. Manual (default) -- opens MuJoCo's full native viewer, which has
     built-in sliders for every actuator's control input. Just run:

         python control_robot.py path/to/RobotDog.xml

     In the viewer window: press Tab (or click the panel icon,
     top-left) to open the side panel, open the "Control" tab, and
     drag each actuator's slider. Watch the matching joint move live.

  2. Auto-test -- no clicking needed. Cycles through every actuator one
     at a time, driving it with a slow sine wave so you can watch each
     joint move on its own and confirm it's wired to the right thing:

         python control_robot.py path/to/RobotDog.xml --auto-test
         python control_robot.py path/to/RobotDog.xml --auto-test --seconds-per-actuator 5 --amplitude 0.5
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer


def actuator_names(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]


def manual_mode(model, data):
    print(f"Loaded model with {model.nu} actuator(s), {model.nsensor} sensor(s).")
    if model.nu:
        print("Actuators:", ", ".join(actuator_names(model)))
    print("\nOpening the MuJoCo viewer...")
    print("  - Press Tab (or click the panel icon, top-left) to show the side panel.")
    print("  - Open the 'Control' tab and drag sliders to drive each actuator live.")
    print("  - Close the window to exit.\n")
    mujoco.viewer.launch(model, data)  # blocking call; full native UI incl. control sliders


def actuator_amplitude_and_center(model, i, amplitude_arg):
    """Pick a safe sine amplitude/center for actuator i, respecting its ctrlrange if set."""
    if amplitude_arg is not None:
        return amplitude_arg, 0.0
    if model.actuator_ctrllimited[i]:
        lo, hi = model.actuator_ctrlrange[i]
        return (hi - lo) / 2 * 0.8, (hi + lo) / 2
    return 0.5, 0.0  # unlimited actuator (e.g. plain motor/torque) -- conservative default


def auto_test_mode(model, data, seconds_per_actuator, amplitude_arg):
    if model.nu == 0:
        print("Model has no actuators -- nothing to test.")
        return

    names = actuator_names(model)
    print(f"Auto-testing {model.nu} actuator(s), {seconds_per_actuator}s each: {', '.join(names)}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i, name in enumerate(names):
            amp, center = actuator_amplitude_and_center(model, i, amplitude_arg)
            print(f"\n[{i + 1}/{model.nu}] Driving actuator '{name}' "
                  f"(center={center:.3f}, amplitude={amp:.3f}) ...")

            t0 = time.time()
            while viewer.is_running() and (time.time() - t0) < seconds_per_actuator:
                step_start = time.time()
                t = time.time() - t0

                data.ctrl[:] = 0.0
                data.ctrl[i] = center + amp * np.sin(2 * np.pi * 0.5 * t)

                mujoco.mj_step(model, data)
                viewer.sync()

                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)

            if not viewer.is_running():
                break

        data.ctrl[:] = 0.0
        print("\nDone testing all actuators. Window stays open, close it to exit "
              "(or Ctrl+C in the terminal).")
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


def main():
    parser = argparse.ArgumentParser(description="View and control an actuated MJCF model.")
    parser.add_argument("xml_path", help="Path to the MJCF .xml file")
    parser.add_argument("--auto-test", action="store_true",
                         help="Automatically sine-sweep each actuator in turn instead of manual sliders")
    parser.add_argument("--seconds-per-actuator", type=float, default=3.0,
                         help="Seconds to drive each actuator in --auto-test mode (default: 3.0)")
    parser.add_argument("--amplitude", type=float, default=None,
                         help="Sine amplitude for --auto-test mode. Defaults to ~80%% of the actuator's "
                              "ctrlrange when it has one, else a conservative 0.5")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    if args.auto_test:
        auto_test_mode(model, data, args.seconds_per_actuator, args.amplitude)
    else:
        manual_mode(model, data)


if __name__ == "__main__":
    main()