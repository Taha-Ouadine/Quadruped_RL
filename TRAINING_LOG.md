# Training Progress Log

Append a dated section for each notable training issue, change, and validation result.

## 2026-09-28: SpiderWalk NaN evaluation reward

### Problem

SpiderWalk evaluation rewards were `nan` from step 0, including in `logs/SpiderWalk-20260928-211206/metrics.csv`. Since the first evaluation was already invalid, PPO updates were not the initial source of the failure.

### Root cause

The SpiderWalk domain randomizer added a uniform `[-1, 1] kg` offset to the trunk mass. The model's base trunk mass is approximately `0.0679 kg`, so this produced non-positive trunk masses; 35 of 64 sampled models had invalid masses. These invalid randomized physics models caused non-finite rollouts and rewards.

### Fix

In `mujoco_playground/_src/locomotion/spider/randomize.py`, replace the absolute mass offset with a multiplicative `0.9..1.1` scale, keeping the lightweight trunk mass positive.

### Validation and next check

- All randomized physical-body masses were positive across 64 sampled models after the change.
- A reset and zero-action step using a sampled randomized model produced finite state and reward.
- The full PPO training run has not yet been rerun; confirm evaluation rewards are finite in a new run.