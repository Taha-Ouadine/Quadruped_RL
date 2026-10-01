print("IMPORTING PACKAGES")
from datetime import datetime
import csv
import functools
import json
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp

from mujoco_playground import wrapper
from mujoco_playground import registry
from IPython.display import HTML, clear_output, display
import time

import pickle
import tempfile
print("FINISHED IMPORTING")


# Env and RL Algorithm ---------------------------------------------------------------------------------------


env_name = 'SpiderWalk'
env_cfg = registry.get_default_config(env_name)
env_cfg.impl = "jax"
env = registry.load(env_name, config=env_cfg)

from mujoco_playground.config import locomotion_params
ppo_params = locomotion_params.brax_ppo_config(env_name)
print('Number of Timesteps:', ppo_params['num_timesteps'])
print('Number of Evals:', ppo_params['num_evals'])

run_name = f"{env_name}-{datetime.now():%Y%m%d-%H%M%S}"
run_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", run_name))
checkpoints_dir = os.path.join(run_dir, "checkpoints")
os.makedirs(checkpoints_dir, exist_ok=True)
with open(os.path.join(run_dir, "config.json"), "w") as config_file:
    json.dump(ppo_params.to_dict(), config_file, indent=2)
metrics_path = os.path.join(run_dir, "metrics.csv")
with open(metrics_path, "w", newline="") as metrics_file:
    csv.writer(metrics_file).writerow(("step", "eval_episode_reward", "eval_episode_reward_std"))
print(f"Run logs and checkpoints: {run_dir}")


# Display ---------------------------------------------------------------------------------------


x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    print(f"[progress] step={num_steps} reward={metrics['eval/episode_reward']:.3f}", flush=True)

    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    plt.figure()
    plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    plt.savefig(os.path.join(run_dir, "training_progress.png"))
    plt.close()
    with open(metrics_path, "a", newline="") as metrics_file:
        csv.writer(metrics_file).writerow((
            num_steps,
            float(metrics["eval/episode_reward"]),
            float(metrics["eval/episode_reward_std"]),
        ))


# Setup ---------------------------------------------------------------------------------------


#randomizer = None if env_name == "SpiderWalk" else registry.get_domain_randomizer(env_name)
randomizer = registry.get_domain_randomizer(env_name) or None
ppo_training_params = dict(ppo_params)
ppo_training_params["save_checkpoint_path"] = checkpoints_dir
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

# TRAIN ---------------------------------------------------------------------------------------


print("TRAINING START")
train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress
)


# EVAL ---------------------------------------------------------------------------------------


make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# Enable perturbation in the eval env.
env_cfg.pert_config.enable = True
env_cfg.pert_config.velocity_kick = [3.0, 6.0]
env_cfg.pert_config.kick_wait_times = [5.0, 15.0]
env_cfg.command_config.a = [1.5, 0.8, 2*jp.pi]
eval_env = registry.load(env_name, config=env_cfg)
velocity_kick_range = [0.0, 0.0]  # Disable velocity kick.
kick_duration_range = [0.05, 0.2]

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))