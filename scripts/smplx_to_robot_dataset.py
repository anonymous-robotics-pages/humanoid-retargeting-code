import argparse
import pathlib
import os
import time
import pickle
import gc

import numpy as np
import torch
from tqdm import tqdm
from rich import print

from collision_free_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from collision_free_motion_retargeting.kinematics_model import KinematicsModel
from collision_free_motion_retargeting import CollisionFreeMotionRetargeting as COLMO


HERE = pathlib.Path(__file__).parent
SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

# Keys load_smplx_file reads. AMASS *_stagei.npz / shape.npz (body shape only) and
# non-SMPL-X .npz files that share a folder with the motions lack them and are skipped.
SMPLX_MOTION_KEYS = {"pose_body", "root_orient", "trans", "betas", "gender", "mocap_frame_rate"}


def is_smplx_motion(path: str) -> bool:
    try:
        with np.load(path, allow_pickle=True) as data:
            return SMPLX_MOTION_KEYS.issubset(data.files)
    except Exception:
        return False


def collect_smplx_files(src_folder: str):
    smplx_files, skipped = [], []
    for dirpath, _, filenames in os.walk(src_folder):
        for fn in filenames:
            if not fn.lower().endswith(".npz"):
                continue
            path = os.path.join(dirpath, fn)
            (smplx_files if is_smplx_motion(path) else skipped).append(path)
    smplx_files.sort()
    return smplx_files, skipped


def process_one_smplx(
    smplx_file_path: str,
    tgt_file_path: str,
    robot: str,
    max_frames: int,
    device: str = "cuda:0",
    collision_mode: str = None,
    max_base_horizontal_speed: float = None,
    ik_config: str = None,
    exclude_collision_limits=(),
):
    # Load SMPL-X and resample to 30 fps
    smplx_data, body_model, smplx_output, _ = load_smplx_file(smplx_file_path, SMPLX_FOLDER)
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30)

    if max_frames and max_frames > 0:
        smplx_data_frames = smplx_data_frames[:max_frames]

    num_frames = len(smplx_data_frames)
    if num_frames == 0:
        raise RuntimeError("No frames after slicing")

    # Init retarget. The JSON human_scale_table is applied as-is (runtime height ratio
    # kept at 1.0 via actual_human_height=None), same as smplx_to_robot.py.
    retarget = COLMO(
        src_human="smplx",
        tgt_robot=robot,
        actual_human_height=None,
        collision_mode=collision_mode,
        ik_config=ik_config,
        exclude_collision_limits=exclude_collision_limits,
    )
    # Per-motion base horizontal-speed cap (collision_cfg max_base_horizontal_speed).
    # A CLI override replaces the YAML value; <= 0 disables the cap entirely. It MUST be
    # applied before adjust_hips_scale_for_motion, which bakes the cap into the root trajectory.
    if max_base_horizontal_speed is not None:
        retarget.max_base_horizontal_speed = (
            None if max_base_horizontal_speed <= 0 else float(max_base_horizontal_speed))
    retarget.adjust_hips_scale_for_motion(smplx_data_frames)

    # Retarget per frame
    qpos_list = []
    t0 = time.time()

    frame_pbar = tqdm(
        enumerate(smplx_data_frames),
        total=num_frames,
        desc=f"Frames ({os.path.basename(smplx_file_path)})",
        unit="frame",
        leave=False,
        mininterval=0.5,  # reduce overhead
    )

    for i, smplx_frame in frame_pbar:
        qpos = retarget.retarget(smplx_frame, frame_idx=i)
        qpos_list.append(qpos.copy())

        elapsed = time.time() - t0
        fps_eff = (i + 1) / max(elapsed, 1e-9)
        eta_sec = (num_frames - (i + 1)) / max(fps_eff, 1e-9)
        frame_pbar.set_postfix_str(f"{fps_eff:.2f} it/s, ETA {eta_sec/60:.1f}m")

    qpos_list = np.asarray(qpos_list)

    # FK (batched)
    kinematics_model = KinematicsModel(retarget.xml_file, device=device)

    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    dof_pos = qpos_list[:, 7:]

    identity_root_pos = torch.zeros((num_frames, 3), device=device)
    identity_root_rot = torch.zeros((num_frames, 4), device=device)
    identity_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        identity_root_pos,
        identity_root_rot,
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
    )
    body_names = kinematics_model.body_names

    motion_data = {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "fps": aligned_fps,
        "link_body_list": body_names,
    }

    os.makedirs(os.path.dirname(tgt_file_path) or ".", exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)

    total_elapsed = time.time() - t0

    # aggressive cleanup (helps long batch runs)
    del kinematics_model, local_body_pos, qpos_list
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    return num_frames, aligned_fps, total_elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--smplx_file", type=str, default=None, help="Single SMPL-X .npz file to process.")
    parser.add_argument("--src_folder", type=str, default=None,
                        help="Folder containing SMPL-X .npz files (process all, recursively).")

    parser.add_argument("--save_path", type=str, default=None, help="Output pkl path for single-file mode.")
    parser.add_argument("--tgt_folder", type=str, default="results", help="Output root folder for folder mode.")

    parser.add_argument("--robot", default="unitree_g1", choices=["unitree_g1"])
    parser.add_argument("--override", action="store_true")

    parser.add_argument("--max_files", default=0, type=int)
    parser.add_argument("--max_frames", default=0, type=int)

    parser.add_argument("--device", default="cuda:0", type=str, help="FK device: cuda:0 or cpu")

    parser.add_argument("--collision_mode", choices=["cbf", "issf", "off"],
                        default=None,
                        help="Collision avoidance mode. cbf: hard QP inequality "
                             "(mink.CollisionAvoidanceLimit); issf: robustified hard CBF "
                             "(ISSfCollisionAvoidanceLimit); off. Unset -> YAML "
                             "parameters.collision_mode or 'cbf'.")

    parser.add_argument("--max_base_horizontal_speed", type=float, default=None,
                        metavar="M_PER_S",
                        help="Override collision_cfg.yaml parameters.max_base_horizontal_speed "
                             "[m/s] for this run, e.g. 2.0. Unset keeps the YAML value "
                             "(unitree_g1 3.0). To turn the cap OFF use "
                             "--no_speed_cap, not a value of 0.")
    parser.add_argument("--ik_config", type=str, default=None, metavar="JSON",
                        help="Path to an IK-config JSON that overrides the built-in "
                             "IK_CONFIG_DICT entry for (smplx, --robot). The robot "
                             "model and assets/<robot>/collision_cfg.yaml are still taken "
                             "from --robot, so the variant JSON must target the same robot.")
    parser.add_argument("--no_speed_cap", action="store_true",
                        help="Disable the base horizontal-speed saturation entirely. The root "
                             "then follows the nominally scaled human trajectory at full speed. "
                             "Mutually exclusive with a positive --max_base_horizontal_speed.")
    parser.add_argument("--exclude_collision_limits", nargs="+", default=(), metavar="NAME",
                        help="collision_cfg.yaml collision_limits entries to drop, by name, "
                             "e.g. 'ground_collision' (self-collision limits stay active).")

    args = parser.parse_args()

    if (args.smplx_file is None) == (args.src_folder is None):
        raise ValueError("Provide exactly one of --smplx_file or --src_folder")

    # Resolve the two cap flags into the single value process_one_smplx takes:
    #   None -> keep the YAML value,  <= 0 -> cap disabled,  > 0 -> that cap [m/s].
    speed_cap = args.max_base_horizontal_speed
    if args.no_speed_cap:
        if speed_cap is not None and speed_cap > 0:
            raise ValueError(
                f"--no_speed_cap conflicts with --max_base_horizontal_speed {speed_cap}: "
                f"pass one or the other.")
        speed_cap = 0.0
    if speed_cap is not None and speed_cap <= 0 and not args.no_speed_cap:
        print("[yellow]--max_base_horizontal_speed <= 0 disables the cap; "
              "--no_speed_cap says so explicitly.[/yellow]")

    process_kwargs = dict(
        device=args.device,
        collision_mode=args.collision_mode,
        max_base_horizontal_speed=speed_cap,
        ik_config=args.ik_config,
        exclude_collision_limits=args.exclude_collision_limits,
    )

    if args.smplx_file:
        if args.save_path is None:
            raise ValueError("--save_path is required when using --smplx_file")

        if os.path.exists(args.save_path) and not args.override:
            print(f"[yellow]Skip (exists): {args.save_path}[/yellow]")
        else:
            nframes, fps, tel = process_one_smplx(
                args.smplx_file, args.save_path, args.robot, args.max_frames, **process_kwargs)
            print(f"[green]Saved[/green]: {args.save_path} | frames={nframes} | fps={fps:g} | time={tel:.1f}s")

    else:
        smplx_files, skipped = collect_smplx_files(args.src_folder)
        if args.max_files and args.max_files > 0:
            smplx_files = smplx_files[:args.max_files]

        if len(smplx_files) == 0:
            raise RuntimeError(f"No SMPL-X motion .npz files found under: {args.src_folder}")

        print(f"[bold]Found {len(smplx_files)} SMPL-X motion files[/bold]"
              + (f" (skipped {len(skipped)} .npz without SMPL-X motion keys)" if skipped else ""))

        for smplx_file_path in tqdm(smplx_files, desc="Files", unit="file"):
            rel_path = os.path.relpath(smplx_file_path, args.src_folder)
            tgt_file_path = os.path.join(args.tgt_folder, os.path.splitext(rel_path)[0] + ".pkl")

            if os.path.exists(tgt_file_path) and not args.override:
                tqdm.write(f"Skipping (exists): {tgt_file_path}")
                continue

            try:
                nframes, fps, tel = process_one_smplx(
                    smplx_file_path, tgt_file_path, args.robot, args.max_frames, **process_kwargs)
                tqdm.write(f"Saved: {tgt_file_path} | frames={nframes} | fps={fps:g} | time={tel:.1f}s")
            except Exception as e:
                tqdm.write(f"[ERROR] {smplx_file_path}: {e}")

        print("Done. saved to ", args.tgt_folder)
