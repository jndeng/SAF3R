import os
import time
from contextlib import contextmanager
from addict import Dict

import numpy as np
import torch

from .data_utils import load_images
from .geometry_utils import estimate_intrinsics_from_pointmaps
from ..models.official_pi3.utils.geometry import se3_inverse
from ..models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
from ..patch import patch_model, update_args_per_forward


# Model init
def build_model(model_cfg, device):
    """
    A unified interface for building different F3R models based on the provided config.
    NOTE: checkpoints will download automatically during the first run
    """
    if model_cfg.name.lower() == "vggt":
        from ..models.official_vggt.models.vggt import VGGT
        model = VGGT(save_intermediates=not model_cfg.get("dpt_only", True))
        
        # AUTO download the checkpoint if not provided
        if not os.path.exists(model_cfg.ckpt_path):
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="facebook/VGGT_tracker_fixed",
                filename=os.path.basename(model_cfg.ckpt_path),
                local_dir=os.path.dirname(model_cfg.ckpt_path)
            )

        model.load_state_dict(torch.load(model_cfg.ckpt_path, map_location="cpu"))
        model = model.to(device).eval()

    elif model_cfg.name.lower() == "pi3":
        from ..models.official_pi3.models.pi3 import Pi3
        model = Pi3.from_pretrained(
            "yyfz233/Pi3", cache_dir=model_cfg.ckpt_path
        ).to(device).eval()

    elif model_cfg.name.lower() == "da3":
        from ..models.official_da3.api import DepthAnything3
        model = DepthAnything3.from_pretrained(
            "depth-anything/DA3-GIANT", cache_dir=model_cfg.ckpt_path
        ).to(device).eval()

    elif model_cfg.name.lower() == "streamvggt":
        from saf3r.models.official_streamvggt.models.streamvggt import StreamVGGT
        model = StreamVGGT()

        # AUTO download the checkpoint if not provided
        if not os.path.exists(model_cfg.ckpt_path):
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="lch01/StreamVGGT",
                filename=os.path.basename(model_cfg.ckpt_path),
                local_dir=os.path.dirname(model_cfg.ckpt_path)
            )

        model.load_state_dict(torch.load(model_cfg.ckpt_path, map_location="cpu"), strict=True)
        model = model.to(device).eval()

    else:
        raise NotImplementedError(f"Model {model_cfg.name} not implemented in build_model()")


    # Patch the model if needed (e.g., for headwise sparsity attention or head profiling)
    if model_cfg.get("patch_module", None) is not None:
        print("=> Patching model ...")
        model = patch_model(model, model_cfg.name.lower(), model_cfg)

    return model



# Inference
def infer_model(model, model_cfg, scene_data):
    """
    A unified interface for running inference with different F3R models on
    a single scene and obtaining predictions in a unified format.
    """
    name = model_cfg.name.lower()

    # Update args (e.g., patch dimensions) before each forward pass
    match name:
        case _ if model_cfg.get("patch_module", None) is not None:
            tmp_img = load_images(model, name, scene_data.image_files[:1])
            img_height, img_width = tmp_img.shape[-2], tmp_img.shape[-1]
            update_args_per_forward(model, name, img_height, img_width)

    # Forward model and obtain predictions in a unified format
    match name:
        case _ if "streamvggt" in name:
            return infer_streamvggt(model, scene_data)
        case _ if "vggt" in name:
            return infer_vggt(model, scene_data)
        case _ if "pi3" in name:
            return infer_pi3(model, scene_data)
        case _ if "da3" in name:
            return infer_da3(model, scene_data)
        case _:
            raise NotImplementedError(f"Model {model_cfg.name} not implemented in infer_model()")


@contextmanager
def cuda_timer():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.perf_counter()

    yield_data = Dict()
    try:
        yield yield_data
    finally:
        torch.cuda.synchronize()
        yield_data["latency"] = time.perf_counter() - t0  # seconds
        yield_data["max_mem"] = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB


def infer_vggt(model, scene_data):
    """
    Run a forward pass of VGGT on a single scene to obtain predictions in a unified format.
    """
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = next(model.parameters()).device
    
    # Load and preprocess images
    images = load_images(model, "vggt", scene_data.image_files).to(device) # [N, 3, H, W]

    # Forward model
    with cuda_timer() as stats:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            raw_output = model(images)

    # Process the raw outputs
    # pose
    extrinsics, intrinsics = pose_encoding_to_extri_intri(raw_output["pose_enc"], images.shape[-2:]) # [1, N, 3, 4], [1, N, 3, 3]
    extrinsics = extrinsics.squeeze().float().cpu().numpy()          # [N, 3, 4]
    intrinsics = intrinsics.squeeze().float().cpu().numpy()          # [N, 3, 3]
    # depth
    depth = raw_output["depth"].squeeze().float().cpu().numpy()      # [N, H, W]
    conf  = raw_output["depth_conf"].squeeze().float().cpu().numpy() # [N, H, W]
    # images
    processed_images = raw_output["images"].squeeze().permute(0, 2, 3, 1).cpu().numpy() # [N, H, W, 3]

    # Organize as a unified output format
    pred_data = Dict({
        "processed_images": processed_images, # [N, H, W, 3] float32 in [0, 1]
        "depth": depth,                       # [N, H, W]
        "conf": conf,                         # [N, H, W]
        "extrinsics": extrinsics,             # [N, 3, 4]
        "intrinsics": intrinsics,             # [N, 3, 3]
    })

    return pred_data, stats


def infer_pi3(model, scene_data):
    """
    Run a forward pass of Pi3 on a single scene to obtain predictions in a unified format.
    """
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = next(model.parameters()).device

    # Load and preprocess images
    images = load_images(model, "pi3", scene_data.image_files).to(device) # [N, 3, H, W]

    # Forward model
    with cuda_timer() as stats:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            raw_output = model(images[None])

    # Parse depth maps from the predicted local points
    # NOTE: 3d points in local camera coords: [x*z, y*z, z]
    local_points = raw_output["local_points"][0] # [N, H, W, 3]
    depth = local_points[..., -1] # [N, H, W]
    conf = raw_output["conf"][0, ..., 0] # [N, H, W]
    poses_c2w_all = raw_output["camera_poses"][0] # c2w matrices [N, 4, 4]

    # Process the raw outputs
    # pose
    # ref: https://github.com/yyfz/Pi3/blob/evaluation/utils/interfaces.py#L118
    extrinsics = se3_inverse(poses_c2w_all.cpu()).numpy()                  # [N, 4, 4]
    extrinsics = extrinsics[:, :3, :]                                      # [N, 3, 4]
    # NOTE: estimate intrinsics from local points using least-squares
    # TODO: a potentially better solution: https://github.com/yyfz/Pi3/issues/2#issuecomment-3131227289
    intrinsics = estimate_intrinsics_from_pointmaps(local_points, fix_principal_point=False) # [N, 3, 3]
    # depth
    depth = local_points[..., -1].cpu().numpy()                            # [N, H, W]
    conf = raw_output["conf"][0, ..., 0].cpu().numpy()                     # [N, H, W]
    # images
    processed_images = images.squeeze().permute(0, 2, 3, 1).cpu().numpy()  # [N, H, W, 3]
    
    # Organize as a unified output format
    pred_data = Dict({
        "processed_images": processed_images, # [N, H, W, 3] float32 in [0, 1]
        "depth": depth,                       # [N, H, W]
        "conf": conf,                         # [N, H, W]
        "extrinsics": extrinsics,             # [N, 3, 4]
        "intrinsics": intrinsics,             # [N, 3, 3]
    })

    return pred_data, stats


def infer_da3(api, scene_data):
    """
    Run a forward pass of DepthAnything3 on a single scene to obtain predictions in a unified format.
    NOTE: Taken from DepthAnything3.inference() with the export logic removed.
    """
    # imgs_cpu: [N, 3, H, W], float32, normalized
    imgs_cpu, _, _ = api._preprocess_inputs(scene_data.image_files, process_res=504, process_res_method="upper_bound_resize")
    imgs, _, _ = api._prepare_model_inputs(imgs_cpu, None, None)

    # Load images and forward model
    with cuda_timer() as stats:
        raw_output = api._run_model_forward(
            imgs, None, None, export_feat_layers=[], ref_view_strategy="first"
        )
    prediction = api._convert_to_prediction(raw_output)
    prediction = api._add_processed_images(prediction, imgs_cpu)
    # convert the processed images to float32 in [0, 1]
    processed_images = prediction.processed_images.astype(np.float32) / 255.0
    
    # Organize as a unified output format
    pred_data = Dict({
        "processed_images": processed_images,   # [N, H, W, 3] float32 in [0, 1]
        "depth": np.round(prediction.depth, 8), # [N, H, W] following `export_to_mini_npz`
        "conf": np.round(prediction.conf, 2),   # [N, H, W] following `export_to_mini_npz`
        "extrinsics": prediction.extrinsics,    # [N, 3, 4]
        "intrinsics": prediction.intrinsics     # [N, 3, 3]
    })

    return pred_data, stats


def infer_streamvggt(model, scene_data):
    """
    Run a forward pass of StreamVGGT on a single scene to obtain predictions in a unified format.
    """
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = next(model.parameters()).device
    
    # Load and preprocess images
    # NOTE: this will use the same loading function as VGGT
    images = load_images(model, "streamvggt", scene_data.image_files).to(device) # [N, 3, H, W]

    # Convert to model input format
    frames = []
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0)  # [1, N, 3, H, W]
        frames.append({"img": image})

    # Forward model
    with cuda_timer() as stats:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            raw_output = model.inference(frames)

    # Merge predicted results from each frame
    all_depth = []
    all_depth_conf = []
    all_camera_pose = []
    for res in raw_output.ress:
        all_depth.append(res["depth"])
        all_depth_conf.append(res["depth_conf"])
        all_camera_pose.append(res["camera_pose"])
    depth = torch.stack(all_depth, dim=1)                   # [1, S, H, W, 1]
    depth_conf = torch.stack(all_depth_conf, dim=1)         # [1, S, H, W]
    pose_enc = torch.stack(all_camera_pose, dim=1)          # [1, S, 9]

    # Process the raw outputs
    # pose
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:]) # [1, N, 3, 4], [1, N, 3, 3]
    extrinsics = extrinsics.squeeze().float().cpu().numpy()  # [N, 3, 4]
    intrinsics = intrinsics.squeeze().float().cpu().numpy()  # [N, 3, 3]
    # depth
    depth = depth.squeeze().float().cpu().numpy()            # [N, H, W]
    conf  = depth_conf.squeeze().float().cpu().numpy()       # [N, H, W]
    # images
    processed_images = images.squeeze().permute(0, 2, 3, 1).cpu().numpy()

    # Organize as a unified output format
    pred_data = Dict({
        "processed_images": processed_images, # [N, H, W, 3] float32 in [0, 1]
        "depth": depth,                       # [N, H, W]
        "conf": conf,                         # [N, H, W]
        "extrinsics": extrinsics,             # [N, 3, 4]
        "intrinsics": intrinsics,             # [N, 3, 3]
    })

    return pred_data, stats
