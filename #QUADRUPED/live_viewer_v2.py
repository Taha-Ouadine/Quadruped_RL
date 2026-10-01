import time
T_START = time.time()

def log(msg):
    print(f"[{time.time() - T_START:7.2f}s] {msg}", flush=True)

import os
import json
import shutil as _shutil
import tempfile
import glob
import struct
import threading

TRAINING_IS_ON = False  # PUT THIS TO TRUE IF YOU'RE VIEWING DURING TRAINING !!!
PRINT_DELAYS = False

if TRAINING_IS_ON:
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jp
    print("JAX backend:", jax.default_backend())
else:
    import jax
    import jax.numpy as jp

# Persistent compilation cache: the 2nd launch skips most XLA compilation.
# (CPU and GPU builds are cached separately, so each needs one warm run.)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

from mujoco_playground import registry
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

import mujoco
from mujoco import mjx
import mujoco.viewer

log("imports done")


# --------------------------------------------------------------
choice = 2
if choice==1:
    env_name = "SpiderWalk"
    RUN_DIR = "/home/tahao_mw5iph3/linuxrobotic/mujoco_playground/logs/SpiderWalk-20260928-164614"
elif choice==2:
    env_name = "Go1JoystickFlatTerrain"
    RUN_DIR = "/home/tahao_mw5iph3/linuxrobotic/mujoco_playground/logs/Go1JoystickFlatTerrain-20260918-234036"
# --------------------------------------------------------------


CHECKPOINTS_DIR = os.path.join(RUN_DIR, "checkpoints")

env_cfg = registry.get_default_config(env_name)
env_cfg.impl = "jax"
env = registry.load(env_name, config=env_cfg)
log("env loaded")

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

mj_model = env.mj_model
mj_data = mujoco.MjData(mj_model)


#-----------------------------------------------------------------
class LinuxJoystick:
    """Read standard Linux joystick events without blocking the viewer loop."""

    EVENT = struct.Struct("<IhBB")
    AXIS_EVENT = 0x02
    AXIS_IDS = {0: "left_x", 1: "left_y", 3: "right_x"}
    DEADZONE = 0.08

    def __init__(self, device):
        self.device = device
        self.fd = None
        self.axes = {axis: 0.0 for axis in self.AXIS_IDS}
        self.buffer = bytearray()
        self.retry_at = 0.0
        self.warned_missing = False
        self._connect()

    def _connect(self):
        if time.monotonic() < self.retry_at:
            return
        self.retry_at = time.monotonic() + 1.0
        try:
            self.fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
            log(f"joystick connected: {self.device}")
            self.warned_missing = False
        except OSError as exc:
            if not self.warned_missing:
                log(f"joystick unavailable at {self.device}; retrying ({exc})")
                self.warned_missing = True

    def read_command(self, limits):
        if self.fd is None:
            self._connect()
            if self.fd is None:
                return (0.0, 0.0, 0.0)

        try:
            while True:
                try:
                    chunk = os.read(self.fd, 64)
                except BlockingIOError:
                    break
                if not chunk:
                    raise OSError("joystick disconnected")
                self.buffer.extend(chunk)
        except OSError as exc:
            os.close(self.fd)
            self.fd = None
            self.buffer.clear()
            self.axes = {axis: 0.0 for axis in self.AXIS_IDS}
            self.retry_at = time.monotonic() + 1.0
            log(f"joystick disconnected; commands neutralized ({exc})")
            return (0.0, 0.0, 0.0)

        while len(self.buffer) >= self.EVENT.size:
            _, value, event_type, number = self.EVENT.unpack_from(self.buffer)
            del self.buffer[:self.EVENT.size]
            event_type &= 0x7F
            if event_type == self.AXIS_EVENT and number in self.AXIS_IDS:
                self.axes[number] = max(-1.0, min(1.0, value / 32767.0))

        def apply_deadzone(value):
            magnitude = abs(value)
            if magnitude <= self.DEADZONE:
                return 0.0
            return (1.0 if value > 0 else -1.0) * (
                (magnitude - self.DEADZONE) / (1.0 - self.DEADZONE)
            )

        forward = apply_deadzone(-self.axes[1])
        lateral = apply_deadzone(-self.axes[0])
        yaw = apply_deadzone(-self.axes[3])
        return (
            forward * limits[0],
            lateral * limits[1],
            yaw * limits[2],
        )

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


JOYSTICK_DEVICE = os.environ.get("MUJOCO_JOYSTICK_DEVICE", "/dev/input/js0")
joystick = LinuxJoystick(JOYSTICK_DEVICE)
COMMAND_LIMITS = tuple(float(value) for value in env_cfg.command_config.a)
#-----------------------------------------------------------------



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


def build_policy(ckpt_path, warmup_obs, warmup_rng):
    """Load a checkpoint, jit the policy, and run it once so it is fully
    compiled before anyone uses it. Returns the ready-to-use function."""
    fixed_path = sanitized_checkpoint_copy(ckpt_path)
    try:
        fn = jax.jit(ppo_checkpoint.load_policy(fixed_path, deterministic=True))
    finally:
        # Params are in memory after load_policy, the temp copy can go.
        _shutil.rmtree(os.path.dirname(fixed_path), ignore_errors=True)
    out = fn(warmup_obs, warmup_rng)
    jax.block_until_ready(out)
    return fn


# ---------------------------------------------------------------- startup
log("waiting for first checkpoint...")
ckpt_path = latest_checkpoint_path()
while ckpt_path is None:
    time.sleep(0.5)
    ckpt_path = latest_checkpoint_path()

rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
jax.block_until_ready(state)
log("reset compiled")

# Warm up EVERYTHING before opening the window, so the robot doesn't freeze
# at the start while XLA compiles step/inference inside the main loop.
log(f"loading policy from {ckpt_path}")
rng, warm_rng = jax.random.split(rng)
jit_inference_fn = build_policy(ckpt_path, state.obs, warm_rng)
last_ckpt_path = ckpt_path
log("policy compiled")

ctrl, _ = jit_inference_fn(state.obs, warm_rng)
state = jit_step(state, ctrl)
jax.block_until_ready(state)
log("step compiled")

rng, reset_rng = jax.random.split(rng)
state = jit_reset(reset_rng)  # start fresh

# ------------------------------------------------- background hot-reload
# New checkpoints are loaded + compiled in a background thread, and swapped
# in only once ready, so the simulation never stalls while reloading.
_pending = {"fn": None, "path": None}
_pending_lock = threading.Lock()
_loading = {"active": False}


def _reload_worker(newest, warmup_obs):
    try:
        log(f"reloading newer checkpoint: {newest}")
        wrng = jax.random.PRNGKey(1)
        fn = build_policy(newest, warmup_obs, wrng)
        with _pending_lock:
            _pending["fn"] = fn
            _pending["path"] = newest
        log("new policy ready")
    except Exception as e:
        # Checkpoint dir may still be mid-write; retry on a later check.
        print(f"Reload failed (probably mid-write), will retry: {e}", flush=True)
    finally:
        _loading["active"] = False


CHECK_EVERY_S = 5.0  # don't hit the filesystem every frame
last_check = time.time()

log("launching viewer")
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    print("Live viewer running. Close window to exit.")
    print("Controls: left stick = forward/strafe, right stick X = turn")

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

        #-----------------------------------------------------------------
        command = jp.asarray(
            joystick.read_command(COMMAND_LIMITS),
            dtype=state.obs["state"].dtype,
        )
        info = dict(state.info)
        info["command"] = command
        info["steps_until_next_cmd"] = jp.full_like(
            info["steps_until_next_cmd"], 2**31 - 1
        )
        obs = dict(state.obs)
        obs["state"] = obs["state"].at[-3:].set(command)
        state_dim = obs["state"].shape[-1]
        obs["privileged_state"] = obs["privileged_state"].at[
            state_dim - 3:state_dim
        ].set(command)
        state = state.replace(info=info, obs=obs)
        #-----------------------------------------------------------------

        # Swap in a freshly compiled policy if the worker finished one.
        with _pending_lock:
            if _pending["fn"] is not None:
                jit_inference_fn = _pending["fn"]
                last_ckpt_path = _pending["path"]
                _pending["fn"] = None

        # Periodically check for a newer checkpoint (non-blocking).
        if step_start - last_check > CHECK_EVERY_S and not _loading["active"]:
            last_check = step_start
            newest = latest_checkpoint_path()
            if newest is not None and newest != last_ckpt_path:
                _loading["active"] = True
                threading.Thread(
                    target=_reload_worker, args=(newest, state.obs), daemon=True
                ).start()

        t0 = time.time()
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

        if PRINT_DELAYS:
            print(f"inference={t1-t0:.4f} step={t2-t1:.4f} reset={t3-t2:.4f} "
                  f"sync_to_mjdata={t4-t3:.4f} viewer_sync={t5-t4:.4f}")

        elapsed = time.time() - step_start
        sleep_time = env.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

joystick.close()