import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from scipy.interpolate import interp1d


def load_smplx_file(smplx_file, smplx_body_model_path):
    smplx_data = np.load(smplx_file, allow_pickle=True)

    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
    )

    T = int(smplx_data["pose_body"].shape[0])

    # ---------- match betas dim to the model's num_betas ----------
    betas = torch.tensor(smplx_data["betas"]).float()
    if betas.ndim == 1:
        betas = betas.view(1, -1)  # (1, B)

    # num_betas the body model actually expects (e.g. 20)
    if hasattr(body_model, "num_betas"):
        nb = int(body_model.num_betas)
    else:
        # fallback: last dim of shapedirs
        nb = int(body_model.shapedirs.shape[-1])

    B = int(betas.shape[1])
    if B > nb:
        betas = betas[:, :nb].contiguous()  # slice
    elif B < nb:
        pad = torch.zeros((1, nb - B), dtype=betas.dtype)
        betas = torch.cat([betas, pad], dim=1)  # pad to nb

    # broadcast the batch dim to T: (1,nb) -> (T,nb)
    if betas.shape[0] == 1 and T > 1:
        betas = betas.repeat(T, 1)

    # ---------- remaining inputs ----------
    global_orient = torch.tensor(smplx_data["root_orient"]).float()  # (T,3)
    body_pose     = torch.tensor(smplx_data["pose_body"]).float()    # (T,63)
    transl        = torch.tensor(smplx_data["trans"]).float()        # (T,3)

    # hand/jaw/eye
    if "pose_hand" in smplx_data.files:
        pose_hand = torch.tensor(smplx_data["pose_hand"]).float()    # (T,90)
        left_hand_pose  = pose_hand[:, :45].contiguous()
        right_hand_pose = pose_hand[:, 45:90].contiguous()
    else:
        left_hand_pose  = torch.zeros((T, 45), dtype=torch.float32)
        right_hand_pose = torch.zeros((T, 45), dtype=torch.float32)

    jaw_pose = torch.tensor(smplx_data["pose_jaw"]).float() if "pose_jaw" in smplx_data.files \
        else torch.zeros((T, 3), dtype=torch.float32)

    if "pose_eye" in smplx_data.files:
        pose_eye = torch.tensor(smplx_data["pose_eye"]).float()      # (T,6)
        leye_pose = pose_eye[:, :3].contiguous()
        reye_pose = pose_eye[:, 3:6].contiguous()
    else:
        leye_pose = torch.zeros((T, 3), dtype=torch.float32)
        reye_pose = torch.zeros((T, 3), dtype=torch.float32)

    # set expression batch to T explicitly (some smplx versions default it to batch=1)
    expr_dim = int(getattr(body_model, "num_expression_coeffs", 10))
    expression = torch.zeros((T, expr_dim), dtype=torch.float32)

    # sanity check: every input has batch T
    assert betas.shape[0] == T and betas.shape[1] == nb
    assert global_orient.shape[0] == T
    assert body_pose.shape[0] == T
    assert transl.shape[0] == T

    smplx_output = body_model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        expression=expression,
        return_full_pose=True,
    )

    # height from betas[0], using the original value before slice/pad
    b0 = smplx_data["betas"][0] if len(smplx_data["betas"].shape) == 1 else smplx_data["betas"][0, 0]
    human_height = 1.66 + 0.1 * float(b0)

    return smplx_data, body_model, smplx_output, human_height


def get_smplx_data(smplx_data, body_model, smplx_output, curr_frame):
    global_orient = smplx_output.global_orient[curr_frame].squeeze()
    full_body_pose = smplx_output.full_pose[curr_frame].reshape(-1, 3)
    joints = smplx_output.joints[curr_frame].detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    result = {}
    joint_orientations = []
    for i, joint_name in enumerate(joint_names):
        if i == 0:
            rot = R.from_rotvec(global_orient)
        else:
            rot = joint_orientations[parents[i]] * R.from_rotvec(full_body_pose[i].squeeze())
        joint_orientations.append(rot)
        result[joint_name] = (joints[i], rot.as_quat(scalar_first=True))

    return result


def slerp(rot1, rot2, t):
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.sum(q1 * q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        return R.from_quat(q1 + t * (q2 - q1))
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q1 + s1 * q2
    return R.from_quat(q)


def get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    src_fps = float(smplx_data["mocap_frame_rate"]) if not hasattr(smplx_data["mocap_frame_rate"], "item") else float(smplx_data["mocap_frame_rate"].item())
    frame_skip = max(1, int(src_fps / tgt_fps))

    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    if tgt_fps < src_fps:
        new_num_frames = num_frames // frame_skip
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames - 1, new_num_frames)

        global_orient_interp = []
        for t in target_time:
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            global_orient_interp.append(slerp(rot1, rot2, alpha).as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)

        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):
            joint_rots = []
            for t in target_time:
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                joint_rots.append(slerp(rot1, rot2, alpha).as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)

        joints_interp = []
        for i in range(joints.shape[1]):
            for j in range(3):
                interp_func = interp1d(original_time, joints[:, i, j], kind="linear")
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)

        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps

    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(single_full_body_pose[i].squeeze())
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))
        smplx_data_frames.append(result)

    return smplx_data_frames, aligned_fps
