# COLMO: Collision-Free Motion Retargeting

COLMO retargets human motion (LAFAN1 BVH and SMPL-X) to the Unitree G1 humanoid and produces
kinematic reference motions for whole-body control and RL-based motion tracking.

**Key features**
- Collision-free retargeting: CBF-based self-collision avoidance and ground-penetration
  prevention as hard QP constraints.
- Soft acceleration-limit constraints that reduce velocity spikes, abrupt corrective motions,
  and the infeasibility caused by hard acceleration bounds.
- A per-frame cap on the base horizontal speed, so fast human root motion stays trackable by
  the robot.
- Built on the code structure of [GMR](https://github.com/YanjieZe/GeneralMotionRetargeting),
  including its two-stage differential IK pipeline, and extended with the collision-free
  formulation above.

This repository is licensed under the [MIT License](LICENSE).

## Installation

Tested on Ubuntu with Python 3.10.

```bash
conda create -n colmo python=3.10 -y
conda activate colmo
pip install -e .

# fixes a possible MuJoCo rendering issue
conda install -c conda-forge libstdcxx-ng -y
```

## Data Preparation

**SMPL-X body models** (needed for SMPL-X input). Download them from
[SMPL-X](https://smpl-x.is.tue.mpg.de/) and place them as follows:

```
assets/body_models/smplx/
├── SMPLX_NEUTRAL.npz
├── SMPLX_FEMALE.npz
└── SMPLX_MALE.npz
```

If you only have the `.pkl` models, change `ext` in `smplx/body_models.py` from `npz` to `pkl`.

**LAFAN1 motions.** Download the BVH files from the
[official repository](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
([lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip)).

**SMPL-X motions.** Any SMPL-X `.npz` in the AMASS layout works (keys `pose_body`,
`root_orient`, `trans`, `betas`, `gender`, `mocap_frame_rate`), e.g. the SMPL-X data from
[AMASS](https://amass.is.tue.mpg.de/) (do not download the SMPL+H data).

## Usage

All scripts target `unitree_g1` (the default `--robot`). Runtime parameters are read from
`assets/unitree_g1/collision_cfg.yaml`; see [Configuration](#configuration).

### LAFAN1 BVH → robot

```bash
# single motion: retarget and view in MuJoCo (optionally save the result)
python scripts/bvh_to_robot.py --bvh_file <motion.bvh> --save_path <out.pkl> --rate_limit

# a whole folder (or one file with --bvh_file/--save_path), no viewer
python scripts/bvh_to_robot_dataset.py --src_folder <bvh_dir> --tgt_folder <out_dir>
```

### SMPL-X → robot

```bash
# single motion: retarget and view in MuJoCo (optionally save the result)
python scripts/smplx_to_robot.py --smplx_file <motion.npz> --save_path <out.pkl> --rate_limit

# a whole folder (or one file with --smplx_file/--save_path), no viewer
python scripts/smplx_to_robot_dataset.py --src_folder <smplx_dir> --tgt_folder <out_dir>
```

In folder mode, `.npz` files without SMPL-X motion keys (e.g. AMASS `*_stagei.npz` shape
files) are skipped.

### Common options

| Option | Scripts | Meaning |
|---|---|---|
| `--collision_mode {cbf,off}` | all four | Collision avoidance mode (default: YAML value, `cbf`). |
| `--rate_limit` | viewer scripts | Play at the motion's frame rate instead of as fast as possible. |
| `--record_video` | viewer scripts | Record the viewer to mp4 (`--video_path` sets the file for `bvh_to_robot.py`; `smplx_to_robot.py` writes `videos/<robot>_<motion>.mp4`). |
| `--loop` | viewer scripts | Loop the motion. |
| `--max_base_horizontal_speed M` / `--no_speed_cap` | dataset scripts | Override or disable the base horizontal-speed cap. |
| `--max_files N` / `--max_frames N` | dataset scripts | Limit the number of files / frames per file. |
| `--override` | dataset scripts | Re-run motions whose output already exists (skipped by default). |
| `--device {cuda:0,cpu}` | dataset scripts | Device for the forward kinematics that fills `local_body_pos`. |

In `bvh_to_robot.py`, press `space` in the viewer to pause/resume.

### Visualize saved robot motion

```bash
# one motion
python scripts/vis_robot_motion.py --robot_motion_path <motion.pkl>

# record a video, or save snapshots offscreen (use MUJOCO_GL=egl on a headless machine)
python scripts/vis_robot_motion.py --robot_motion_path <motion.pkl> --record_video --video_path <out.mp4>
python scripts/vis_robot_motion.py --robot_motion_path <motion.pkl> --snapshot_frames 0 60 100-200:20

# a folder of motions: browse interactively, or record one video per motion
python scripts/vis_robot_motion_dataset.py --mode interactive --robot_motion_folder <out_dir>
python scripts/vis_robot_motion_dataset.py --mode batch --robot_motion_folder <out_dir> --video_dir <video_dir>
```

In interactive mode, `[` / `]` switch to the previous / next motion and `space` pauses.

## Motion Data Format

Each frame of **human motion** is a dict `{human_body_name: (position, quaternion)}` in the
world frame, with quaternions in wxyz order (MuJoCo convention).

Each frame of **robot motion** is `(base_position, base_rotation, joint_positions)`. Saved
`.pkl` files contain:

| Key | Shape | Content |
|---|---|---|
| `fps` | scalar | Frame rate of the saved motion (30 by default). |
| `root_pos` | `(T, 3)` | Base position [m]. |
| `root_rot` | `(T, 4)` | Base orientation, quaternion **xyzw**. |
| `dof_pos` | `(T, 29)` | Joint positions [rad], in MuJoCo joint order. |
| `local_body_pos` | `(T, N, 3)` or `None` | Body positions with the base at the origin (dataset scripts only). |
| `link_body_list` | `N` names or `None` | Body names for `local_body_pos` (dataset scripts only). |

`collision_free_motion_retargeting.load_robot_motion()` loads a `.pkl` and converts `root_rot`
back to wxyz.

## Configuration

**`assets/unitree_g1/collision_cfg.yaml`** holds the runtime parameters, the robot collision
body groups, and the collision constraints. The most commonly changed parameters:

| Parameter | Meaning |
|---|---|
| `collision_mode` | `cbf` (hard CBF, default) or `off`. |
| `max_base_horizontal_speed` | Cap on the base horizontal speed [m/s]; comment out to disable. |
| `base_speed_cap_mode` | `per_frame` (compress only the frames above the cap) or `clip` (one scale for the whole motion). |
| `damping`, `max_iter`, `warmup_iters` | IK regularization, iterations per frame, and iterations on the first frame. |
| `use_velocity_limit`, `use_acceleration_limit`, `use_ik_step_limit` | Enable the per-joint velocity, soft acceleration and IK step limits. |
| `frame_velocity_limit_soft`, `frame_acceleration_limit_soft` | Per-joint velocity / acceleration limits. |

**IK configs** (`collision_free_motion_retargeting/ik_configs/`) map human bodies to robot
bodies: `bvh_lafan1_to_g1.json` and `smplx_to_g1.json`.

| Field | Meaning |
|---|---|
| `human_root_name`, `robot_root_name` | Root bodies of the human and the robot. |
| `human_height_assumption` | Human height [m] the scale table assumes. |
| `human_scale_table` | Per-body scale applied to the human motion (a scalar, or `[x, y, z]`). |
| `ground_height`, `ground_anchor_bodies` | Floor height and the robot bodies (feet) used to calibrate it on the first frame. |
| `use_ik_match_table1`, `use_ik_match_table2` | Enable the two IK stages. |
| `ik_match_table1`, `ik_match_table2` | Robot body → human body targets for the two IK stages. |

Each IK match table entry has the form:

```jsonc
"pelvis": [          // robot body name
    "pelvis",        // corresponding human body name
    0,               // weight on the 3D position error
    100,             // weight on the 3D rotation error
    [0.0, 0.0, 0.0], // position offset added to the human body
    [0.5, -0.5, -0.5, -0.5]  // rotation offset applied to the human body (wxyz)
]
```

## Acknowledgement

This repository is built on the overall code structure of
[GMR](https://github.com/YanjieZe/GeneralMotionRetargeting); we thank the GMR authors for
open-sourcing their implementation. The IK solver builds on [mink](https://github.com/kevinzakka/mink)
and [MuJoCo](https://github.com/google-deepmind/mujoco), and visualization on MuJoCo. The human
motion data we use includes [AMASS](https://amass.is.tue.mpg.de/) and
[LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

The robot model comes from [Unitree G1](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description).
