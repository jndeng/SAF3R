"""
Utility functions for DA3Bench evaluation.
"""

import os
import random
from addict import Dict

from scipy.io import loadmat
from sklearn import neighbors as skln
import numpy as np
import cv2
import imageio
import open3d as o3d
import torch

from .bench.utils import create_tsdf_volume, fuse_depth_to_tsdf,sample_points_from_mesh


# ========================================================== #
# ========================== Data ========================== #
# ========================================================== #
def sample_frames(
    scene_data: Dict, scene: str, max_frames=100, indices=None, sample_mode="uniform",
    seq_multiple=1, seed=42,
) -> Dict:
    """
    Modified from Evaluator._sample_frames() to support deterministic sampling.

    ===
    Sample frames if scene has more than max_frames.

    Args:
        scene_data: Scene data dict with image_files, extrinsics, intrinsics, aux
        scene: Scene name (for logging)
        max_frames: Number of frames to sample.
        indices: Frame indices. If given, sample the specified frames.
        sample_mode: "random" | "uniform"

    Returns:
        Sampled scene_data if num_frames > max_frames, otherwise original
    """
    num_frames = len(scene_data.image_files)

    # Sample with fixed seed for reproducibility
    if indices is None:
        if num_frames <= max_frames:
            sampled_indices = list(range(num_frames))
        else:
            if sample_mode == "random":
                random.seed(seed)
                indices = list(range(num_frames))
                random.shuffle(indices)
                sampled_indices = sorted(indices[:max_frames])
            elif sample_mode == "uniform":
                sampled_indices = np.linspace(
                    0, num_frames - 1, num=max_frames, dtype=np.int64
                ).tolist()
            
        # NOTE: for LiteVGGT which requires sequence length to be a multiple of 8
        #       we force using the first N frames that satisfy this constraint
        if seq_multiple > 1 and len(sampled_indices) % seq_multiple != 0:
            org_len = len(sampled_indices)
            new_len = org_len // seq_multiple * seq_multiple
            sampled_indices = sampled_indices[:new_len]
            print(f"[Warning] Truncated sequence length from {org_len} to {new_len} to satisfy TE FP8 requirements.")

    else:
        sampled_indices = sorted(indices)
    
    print(f"Sampled {scene} ({sample_mode}): {num_frames} -> {len(sampled_indices)} frames")

    # Create new scene_data with sampled frames
    sampled = Dict()
    sampled.image_files = [scene_data.image_files[i] for i in sampled_indices]
    sampled.extrinsics = scene_data.extrinsics[sampled_indices]
    sampled.intrinsics = scene_data.intrinsics[sampled_indices]

    # Copy aux data, sampling lists if needed
    sampled.aux = Dict()
    for key, val in scene_data.aux.items():
        if isinstance(val, list) and len(val) == num_frames:
            sampled.aux[key] = [val[i] for i in sampled_indices]
        elif isinstance(val, np.ndarray) and len(val) == num_frames:
            sampled.aux[key] = val[sampled_indices]
        else:
            sampled.aux[key] = val

    return sampled_indices, sampled


# ====================================================================== #
# ========================== Fuse3D functions ========================== #
# ====================================================================== #
def fuse3d(dataset, scene, gt_data, pred_data, mode="recon_unposed"):
    dataset_name = gt_data.dataset_name.lower()
    if dataset_name == "7scenes":
        return fuse3d_7scenes(dataset, scene, gt_data, pred_data, mode)
    elif dataset_name == "eth3d":
        return fuse3d_eth3d(dataset, scene, gt_data, pred_data, mode)
    elif dataset_name == "hiroom":
        return fuse3d_hiroom(dataset, scene, gt_data, pred_data, mode)
    elif dataset_name == "scannetpp":
        return fuse3d_scannetpp(dataset, scene, gt_data, pred_data, mode)
    elif dataset_name == "dtu":
        return fuse3d_dtu(dataset, scene, gt_data, pred_data, mode)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def fuse3d_7scenes(dataset, scene, gt_data, pred_data, mode):
    """
    Modified from SevenScenes.fuse3d()

    ===
    Fuse per-view depths into a point cloud using TSDF fusion.

    Args:
        scene: Scene identifier
        gt_data: gt data
        pred_data: predicted data
        mode: "recon_unposed" or "recon_posed"
    """
    # Load original images (keep original size)
    images = []
    orig_sizes = []
    for img_path in gt_data.image_files:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        orig_sizes.append((img.shape[0], img.shape[1]))

    # Prepare depths, intrinsics, extrinsics
    if mode == "recon_unposed":
        depths, intrinsics, extrinsics = dataset._prep_unposed(
            pred_data, gt_data, orig_sizes, scene=scene
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    images = np.stack(images, axis=0)

    # Create TSDF volume and fuse
    volume = create_tsdf_volume(
        voxel_length=dataset.voxel_length,
        sdf_trunc=dataset.sdf_trunc,
    )
    mesh = fuse_depth_to_tsdf(
        volume, depths, images, intrinsics, extrinsics, max_depth=dataset.max_depth
    )

    # Sample points from mesh
    pcd = sample_points_from_mesh(mesh, dataset.sampling_number)

    return mesh, pcd, intrinsics, extrinsics


def fuse3d_eth3d(dataset, scene, gt_data, pred_data, mode):
    """
    Modified from ETH3D.fuse3d()

    ===
    Fuse per-view depths into a point cloud using TSDF fusion.

    Pipeline:
    1. Load original images (keep original size)
    2. Resize depth to original image size (nearest interpolation)
    3. Adjust intrinsics to original image size
    4. Apply scale alignment and mask invalid depths
    5. TSDF fusion

    Args:
        scene: Scene identifier
        result_path: Path to npz file with predicted depths/poses
        fuse_path: Output path for fused point cloud (.ply)
        mode: "recon_unposed" or "recon_posed"
    """
    # Load original images (keep original size)
    images = []
    orig_sizes = []  # (H, W) for each image
    for img_path in gt_data.image_files:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        orig_sizes.append((img.shape[0], img.shape[1]))

    # Prepare depths, intrinsics, extrinsics
    if mode == "recon_unposed":
        depths, intrinsics, extrinsics = dataset._prep_unposed(
            pred_data, gt_data, orig_sizes, scene=scene
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    images = np.stack(images, axis=0)

    # Create TSDF volume and fuse
    volume = create_tsdf_volume(
        voxel_length=dataset.voxel_length,
        sdf_trunc=dataset.sdf_trunc,
    )
    mesh = fuse_depth_to_tsdf(
        volume, depths, images, intrinsics, extrinsics, max_depth=dataset.max_depth
    )

    # Sample points from mesh
    pcd = sample_points_from_mesh(mesh, dataset.sampling_number)

    return mesh, pcd, intrinsics, extrinsics


def fuse3d_hiroom(dataset, scene, gt_data, pred_data, mode):
    """
    Modified from HiRoomDataset.fuse3d()

    ===
    Fuse per-view depths into a point cloud using TSDF fusion.

    Args:
        scene: Scene identifier
        result_path: Path to npz file with predicted depths/poses
        fuse_path: Output path for fused point cloud (.ply)
        mode: "recon_unposed" or "recon_posed"
    """
    # Get full GT data
    full_gt_data = dataset.get_data(scene)

    # Try to load saved GT meta (handles frame sampling)
    image_indices = [
        full_gt_data.image_files.index(f)
        for f in gt_data.image_files
        if f in full_gt_data.image_files
    ]
    
    # Load original images (keep original size)
    images = []
    orig_sizes = []
    for img_idx in image_indices:
        img_path = full_gt_data.image_files[img_idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        orig_sizes.append((img.shape[0], img.shape[1]))
    
    images = np.stack(images, axis=0)

    # Prepare depths, intrinsics, extrinsics
    if mode == "recon_unposed":
        depths, intrinsics, extrinsics = dataset._prep_unposed(
            pred_data, gt_data, full_gt_data, image_indices, orig_sizes, scene=scene
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Create TSDF volume and fuse
    volume = create_tsdf_volume(
        voxel_length=dataset.voxel_length,
        sdf_trunc=dataset.sdf_trunc,
    )
    mesh = fuse_depth_to_tsdf(
        volume, depths, images, intrinsics, extrinsics, max_depth=dataset.max_depth
    )

    # Sample points from mesh
    pcd = sample_points_from_mesh(mesh, dataset.sampling_number)

    return mesh, pcd, intrinsics, extrinsics


def fuse3d_scannetpp(dataset, scene, gt_data, pred_data, mode):
    """
    Modified from ScanNetPP.fuse3d()

    ===
    Fuse per-view depths into a point cloud using TSDF fusion.

    Args:
        scene: Scene identifier
        result_path: Path to npz file with predicted depths/poses
        fuse_path: Output path for fused point cloud (.ply)
        mode: "recon_unposed" or "recon_posed"
    """
    # Get GT data
    full_gt_data = dataset.get_data(scene)

    # Need to rebuild aux from full GT data based on image indices
    image_indices = [
        full_gt_data.image_files.index(f)
        for f in gt_data.image_files
        if f in full_gt_data.image_files
    ]
    
    # Load and preprocess images
    images = []
    for idx, img_idx in enumerate(image_indices):
        img_path = full_gt_data.image_files[img_idx]
        image = imageio.imread(img_path).astype(np.uint8)

        # Undistort and crop
        ixt_raw = full_gt_data.aux.ixt_raw_list[img_idx]
        ixt = full_gt_data.intrinsics[img_idx]
        dist = full_gt_data.aux.dist_list[img_idx]
        roi = full_gt_data.aux.roi_list[img_idx]

        image = cv2.undistort(image, ixt_raw, dist, newCameraMatrix=ixt)
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]
        image = cv2.resize(image, (dataset.input_w, dataset.input_h), interpolation=cv2.INTER_AREA)

        images.append(image)

    images = np.stack(images, axis=0)

    # Prepare depths, intrinsics, extrinsics
    if mode == "recon_unposed":
        depths, intrinsics, extrinsics = dataset._prep_unposed(
            pred_data, gt_data, full_gt_data, image_indices, scene=scene
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Create TSDF volume and fuse
    volume = create_tsdf_volume(
        voxel_length=dataset.voxel_length,
        sdf_trunc=dataset.sdf_trunc,
    )
    mesh = fuse_depth_to_tsdf(
        volume, depths, images, intrinsics, extrinsics, max_depth=dataset.max_depth
    )

    # Sample points from mesh
    pcd = sample_points_from_mesh(mesh, dataset.sampling_number)

    return mesh, pcd, intrinsics, extrinsics


def fuse3d_dtu(dataset, scene, gt_data, pred_data, mode):
    """
    Modified from DTU.fuse3d()

    ---
    Fuse per-view depths into a point cloud and save to PLY.

    Args:
        scene: Scene identifier (e.g., "scan114")
        result_path: Path to npz file containing predicted depths/poses
        fuse_path: Output path for fused point cloud (.ply)
        mode: "recon_unposed" or "recon_posed"
    """
    gt_data = dataset.get_data(scene) # reload GT data to get info of all frames
    masks = dataset.load_masks(gt_data.aux.mask_files)

    if mode == "recon_unposed":
        depths, intrinsics, extrinsics = dataset._prep_unposed(pred_data, gt_data, masks)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    proj_mat = dataset._build_proj_mats(intrinsics, extrinsics)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    depths_t = torch.from_numpy(depths).to(device=device, dtype=dtype).unsqueeze(1)
    proj_t = torch.from_numpy(proj_mat).to(device=device, dtype=dtype)
    height, width = depths_t.shape[-2:]

    points = []
    for idx in range(len(gt_data.image_files)):
        if mode == "recon_unposed":
            # Simple unfiltered back-projection per frame
            cur_p_pcd = dataset._generate_points_from_depth(
                depths_t[idx : idx + 1], proj_t[idx : idx + 1]
            )
            mask = (depths_t[idx : idx + 1] > 0.001).squeeze()
            cur_p_pcd = cur_p_pcd[:, :, mask]
            no_filter_pc = cur_p_pcd.squeeze(0).permute(1, 0).cpu().numpy()
            points.append(no_filter_pc)
        else:  # recon_posed
            final_pc = dataset._fuse_consistent_points(depths_t, proj_t, idx, height, width)
            points.append(final_pc)

    # Concatenate and optionally downsample to hard cap
    points_np = np.concatenate(points, axis=0)
    points_np = dataset._cap_points(points_np, max_points=4_000_000)  # DTU_MAX_POINTS

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    mesh = None  # Not available for DTU since we don't do TSDF fusion

    return mesh, pcd, intrinsics, extrinsics


# ====================================================================== #
# ========================== Eval3D functions ========================== #
# ====================================================================== #
# NOTE: Only DTU uses a different evaluation protocol for point-cloud reconstruction
#       other datasets use the unified api `evaluate_3d_reconstruction`
def evaluate_reconstruction_dtu(
    dataset,
    pred_pcd,
    gt_pcd,
    mask_file,
    plane_file,
    down_dense: float = 0.2,
    patch: int = 60,
    max_dist: int = 20,
    use_gpu: bool = False,
) -> tuple:
    """
    Modified from DTU._evaluate_reconstruction()

    --
    Compute accuracy, completeness, and overall metrics for one scan.

    Args:
        scanid: Scan identifier
        pred_ply: Predicted point cloud path or array
        gt_ply: Ground truth point cloud path or array
        mask_file: ObsMask file path
        plane_file: Plane file path
        down_dense: Downsample density (min distance between points)
        patch: Patch size for boundary
        max_dist: Outlier threshold in mm
        use_gpu: If True, use GPU-accelerated distance computation

    Returns:
        Tuple of (mean_d2s, mean_s2d, overall)
    """
    thresh = down_dense

    # Load and downsample predicted point cloud
    data_pcd = np.asarray(pred_pcd.points)
    # Use fixed seed for reproducibility
    shuffle_rng = np.random.default_rng(seed=42)
    shuffle_rng.shuffle(data_pcd, axis=0)

    # Downsample point cloud
    nn_engine = skln.NearestNeighbors(
        n_neighbors=1, radius=thresh, algorithm="kd_tree", n_jobs=-1
    )
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    # Restrict to observed volume (ObsMask)
    obs_mask_file = loadmat(mask_file)
    ObsMask, BB, Res = (obs_mask_file[attr] for attr in ["ObsMask", "BB", "Res"])
    BB = BB.astype(np.float32)

    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(
        axis=-1
    ) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(
        axis=-1
    ) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(
        np.bool_
    )
    data_in_obs = data_in[grid_inbound][in_obs]

    # Compute accuracy (pred -> GT) and completeness (GT -> pred)
    stl = np.asarray(gt_pcd.points)

    if use_gpu and torch.cuda.is_available():
        # GPU-accelerated distance computation
        mean_d2s = dataset._knn_dist_gpu(data_in_obs, stl, max_dist)
    else:
        # CPU version (original, for exact reproduction)
        nn_engine.fit(stl)
        dist_d2s, _ = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
        mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    ground_plane = loadmat(plane_file)["P"]
    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    if use_gpu and torch.cuda.is_available():
        # GPU-accelerated distance computation
        mean_s2d = dataset._knn_dist_gpu(stl_above, data_in, max_dist)
    else:
        # CPU version (original, for exact reproduction)
        nn_engine.fit(data_in)
        dist_s2d, _ = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
        mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    overall = (mean_d2s + mean_s2d) / 2
    return mean_d2s, mean_s2d, overall