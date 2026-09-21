import time
import os
import json
import shutil as _shutil
import tempfile
import glob

TRAINING_IS_ON = False # PUT THIS TO TRUE IF YOU'RE VIEWING DURING TRAINING !!!
PRINT_DELAYS = False

if not TRAINING_IS_ON:
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    print("JAX backend:", jax.default_backend())
else: import jax

from mujoco_playground import registry
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

import mujoco
from mujoco import mjx
import mujoco.viewer

env_name = "Go1JoystickFlatTerrain"
RUN_DIR = "/home/tahao_mw5iph3/linuxrobotic/mujoco_playground/logs/Go1JoystickFlatTerrain-20260918-234036"
CHECKPOINTS_DIR = os.path.join(RUN_DIR, "checkpoints")

env_cfg = registry.get_default_config(env_name)
env_cfg.impl = "jax"
env = registry.load(env_name, config=env_cfg)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

mj_model = env.mj_model
mj_data = mujoco.MjData(mj_model)


def latest_checkpoint_path():
    subdirs = [d for d in glob.glob(os.path.join(CHECKPOINTS_DIR, "*")) if os.path.isdir(d)]
    if not subdirs:
        return None
    # Checkpoint dirs are zero-padded step numbers, e.g. "000045875200" -> sort numerically.
    return max(subdirs, key=lambda d: int(os.path.basename(d)))

def sanitized_checkpoint_copy(ckpt_path):
    """Copy a checkpoint dir and strip None-valued kernel-init keys
    that trigger brax's load_config bug (KERNEL_INITIALIZER[None])."""
    tmp_dir = tempfile.mkdtemp(prefix="rscope_ckpt_fix_")
    dest = os.path.join(tmp_dir, os.path.basename(ckpt_path))
    _shutil.copytree(ckpt_path, dest)

    config_path = os.path.join(dest, "ppo_network_config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    nfk = cfg.get("network_factory_kwargs", {})
    for key in list(nfk.keys()):
        if key.endswith("_kernel_init_fn") and nfk[key] is None:
            del nfk[key]  # let load_config's `continue` skip it entirely

    with open(config_path, "w") as f:
        json.dump(cfg, f)

    return dest


print("Waiting for first checkpoint...")
ckpt_path = None
while ckpt_path is None:
    ckpt_path = latest_checkpoint_path()
    time.sleep(2.0)

print(f"Loading policy from {ckpt_path}")
fixed_path = sanitized_checkpoint_copy(ckpt_path)
print(f"Path sanitized, loading the policy...")
jit_inference_fn = jax.jit(ppo_checkpoint.load_policy(fixed_path, deterministic=True))
last_ckpt_path = ckpt_path

rng = jax.random.PRNGKey(0)
state = jit_reset(rng)

print("Launching Viewer")
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    print("Live viewer running. Close window to exit.")

    # TRACK ROBOT WITH CAMERA
    tracked_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if tracked_body_id == -1:
        print("WARNING: 'trunk' body not found, falling back to body 1")
        tracked_body_id = 1
    viewer.cam.trackbodyid = tracked_body_id
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.distance = 3.0    # how far the camera sits from the robot
    viewer.cam.azimuth = 90      # horizontal angle
    viewer.cam.elevation = -15   # vertical angle

    while viewer.is_running():
        step_start = time.time()

        # Hot-reload if a newer checkpoint has appeared.
        newest = latest_checkpoint_path()
        if PRINT_DELAYS: t0 = time.time()

        if newest is not None and newest != last_ckpt_path:
            try:
                print(f"Reloading newer checkpoint: {newest}")
                fixed_path = sanitized_checkpoint_copy(newest)
                jit_inference_fn = jax.jit(ppo_checkpoint.load_policy(fixed_path, deterministic=True))
                if 'prev_tmp_root' in dir():
                    _shutil.rmtree(os.path.dirname(fixed_path), ignore_errors=True)
                prev_tmp_root = fixed_path
                last_ckpt_path = newest
            except Exception as e:
                # Checkpoint dir may still be mid-write; try again next loop.
                print(f"Reload failed (probably mid-write), will retry: {e}")

        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        if PRINT_DELAYS: t1 = time.time()

        state = jit_step(state, ctrl)
        if PRINT_DELAYS: t2 = time.time()

        if state.done:
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)
        if PRINT_DELAYS: t3 = time.time()

        # Sync mjx physics state into a real mjData for the viewer.
        mj_data_synced = mjx.get_data(mj_model, state.data)
        mj_data.qpos[:] = mj_data_synced.qpos
        mj_data.qvel[:] = mj_data_synced.qvel
        mujoco.mj_forward(mj_model, mj_data)
        if PRINT_DELAYS: t4 = time.time()

        viewer.sync()
        if PRINT_DELAYS: t5 = time.time()

        if PRINT_DELAYS: print(f"checkpoint_check={t0-step_start:.4f} inference={t1-t0:.4f} step={t2-t1:.4f} reset={t3-t2:.4f} sync_to_mjdata={t4-t3:.4f} viewer_sync={t5-t4:.4f}")

        elapsed = time.time() - step_start
        sleep_time = env.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)