# `scripts/` — Entry Points

Command-line entry points for retargeting human motion to the Unitree **G1** and viewing
the results. Runtime parameters are read from `assets/unitree_g1/collision_cfg.yaml`.

Common flags: `--collision_mode {cbf, issf, off}`, `--save_path <out.pkl>`,
`--record_video --video_path <out.mp4>`.

## Retargeting (input format → robot)

| Script | Role |
|--------|------|
| `bvh_to_robot.py` | Retarget **one** LAFAN1 BVH file and view it live (+ optional `.pkl` save). |
| `bvh_to_robot_dataset.py` | **Batch** retarget a LAFAN1 BVH folder (or one file) to `.pkl`; also runs `KinematicsModel` FK to store `local_body_pos`. Supports `--frame_stride`, `--max_frames`, resume/skip. |
| `smplx_to_robot.py` | Retarget one SMPL-X `.npz` motion (AMASS layout) and view it (+ optional `.pkl` save). |
| `smplx_to_robot_dataset.py` | **Batch** retarget an SMPL-X folder (or one file) to `.pkl`, same options and pkl layout as `bvh_to_robot_dataset.py`. `.npz` files without SMPL-X motion keys (e.g. AMASS `*_stagei.npz`) are skipped. |

## Visualization

| Script | Role |
|--------|------|
| `vis_robot_motion.py` | Play back a single robot-motion `.pkl` (optional video / snapshot capture). |
| `vis_robot_motion_dataset.py` | Viewer for a folder of `.pkl`s: `interactive` (browse with `[`/`]`), `batch` (record each to mp4), `single`. |
| `render_style.py` | Rendering helpers shared by the viewers (not an entry point). |

## Typical workflow

```bash
# 1. retarget a single BVH and preview
python scripts/bvh_to_robot.py --bvh_file <file.bvh> --save_path results/out.pkl

# 2. batch a whole BVH / SMPL-X folder into .pkl
python scripts/bvh_to_robot_dataset.py --src_folder <bvh_dir> --tgt_folder results/out
python scripts/smplx_to_robot_dataset.py --src_folder <smplx_dir> --tgt_folder results/out

# 3. review the results
python scripts/vis_robot_motion.py --robot_motion_path results/out.pkl
python scripts/vis_robot_motion_dataset.py --mode batch \
    --robot_motion_folder results/out --video_dir videos/out
```
