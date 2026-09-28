# `collision_free_motion_retargeting/` — Core Library

Core Python package for retargeting human motion (LAFAN1 BVH / SMPL-X) onto the
Unitree **G1** humanoid.

## Pipeline at a glance

```
human motion file
  → format loader (utils/*)  →  per-frame dict {body_name: (position, quaternion)}
  → CollisionFreeMotionRetargeting.retarget()  (mink IK QP)
  → robot qpos  →  viewer / saved .pkl
```

## Core modules

| File | Role |
|------|------|
| `motion_retarget.py` | **Heart of the pipeline.** `CollisionFreeMotionRetargeting.retarget()` scales/offsets each human frame, sets mink `FrameTask` targets, and solves a 2-stage (table1/table2) IK QP to produce robot `qpos`. Includes CBF / ISSf self-collision avoidance (hard QP inequalities), a **soft** per-frame acceleration limit (DAQP `sense=8`), and a foot-contact zero-velocity limit. Reads runtime params from `assets/<robot>/collision_cfg.yaml`. |
| `params.py` | Registry constants: robot MuJoCo XML path (`ROBOT_XML_DICT`), input-source × robot IK-config JSON map (`IK_CONFIG_DICT`), robot base body name (`ROBOT_BASE_DICT`), viewer camera distance (`VIEWER_CAM_DISTANCE_DICT`). Robot: `unitree_g1`; sources: `smplx`, `bvh_lafan1`. |
| `__init__.py` | Public API. Re-exports the constants plus `CollisionFreeMotionRetargeting`, `RobotMotionViewer`, `draw_frame`, `load_robot_motion`, `KinematicsModel`. |
| `robot_motion_viewer.py` | MuJoCo passive-viewer visualization. `RobotMotionViewer.step()` sets robot qpos and renders, overlaying human coordinate frames (`draw_frame`) and foot-contact points; supports camera follow, fps rate-limiting, mp4 recording (imageio), and collision-geom toggling. |
| `data_loader.py` | Reader for saved robot-motion pickles. `load_robot_motion()` unpacks fps / root_pos / root_rot / dof_pos and converts root_rot xyzw → wxyz (MuJoCo scalar-first). Consumed by the playback scripts. |

## Kinematics & math

| File | Role |
|------|------|
| `kinematics_model.py` | Pure-PyTorch (batched, differentiable) forward-kinematics model parsed directly from the robot MuJoCo XML. `forward_kinematics(root_pos, root_rot, dof_pos) → body_pos/body_rot`. **Not** part of the IK; used by the batch dataset scripts to compute `local_body_pos` for the saved pkl. |
| `torch_utils.py` | IsaacGym-derived batched PyTorch quaternion ops (scalar-last / xyzw). Used by `kinematics_model.py`. |

## `utils/` — human-format loaders

| File | Role |
|------|------|
| `utils/lafan1.py` | **LAFAN1 BVH loader.** Computes global pose, applies Y-up→Z-up + cm→m, adds `LeftFootMod`/`RightFootMod` foot keys; returns `(frames, human_height)`. |
| `utils/smpl.py` | **SMPL-X loader.** Runs the SMPL-X body model, builds per-joint global (pos, quat) dicts, and offers fps down-sampling (slerp). |
| `utils/lafan_vendor/extract.py` | Original-author LAFAN1 BVH parser. `read_bvh` (used by `lafan1.py`). |
| `utils/lafan_vendor/utils.py` | LAFAN1 vendor quaternion/FK math (`quat_fk`, etc.), used by `lafan1.py` / `extract.py`. |
| `utils/__init__.py`, `utils/lafan_vendor/__init__.py` | Empty package initializers. |
