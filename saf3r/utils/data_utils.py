"""
Data-related utility functions copied from VGGT/Pi3/DA3 official codes.
"""

import os
import math
from PIL import Image

import imageio
import numpy as np
import cv2
import torch
from torchvision import transforms as TF


# Images
def load_images(model, model_name, image_files):
    match model_name.lower():
        case _ if "vggt" in model_name:
            return load_and_process_images_vggt(image_files)
        case _ if "pi3" in model_name:
            return load_and_process_images_pi3(image_files)
        case _ if "da3" in model_name:
            images, _, _ = model._preprocess_inputs(
                image_files, process_res=504, process_res_method="upper_bound_resize"
            )
            return images
        case _:
            raise NotImplementedError(f"Model {model_name} not implemented in load_images()")


def load_and_process_images_vggt(image_path_list, mode="crop"):
    """
    NOTE: Copied from `load_and_preprocess_images()` in `load_fn.py`

    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def load_and_process_images_pi3(image_path_list, PIXEL_LIMIT=255000):
    """
    Modified from pi3.utils.basic to support a list of input images instead of a directory or video.
    
    ---
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    
    # --- 1. Load images ---
    for img_path in image_path_list:
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")
    
    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    # print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = TF.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)



# Depth maps
def load_depths(gt_data):
    """
    Load GT depth maps based on dataset.

    Returns:
        gt_depths (list of np.ndarray): List of GT depth maps corresponding to the input gt_data.
            Each depth map is a 2D array of shape [H, W] with depth values in meters.
    """
    dataset_name = gt_data.dataset_name.lower()

    match dataset_name:
        case "7scenes":
            gt_depths = load_depths_7scenes(gt_data.aux.gt_depth_files)
        case "eth3d":
            gt_depths = load_depths_eth3d(gt_data.image_files)
        case "scannetpp":
            gt_depths = load_depths_scannetpp(gt_data.aux.gt_depth_files)
        case "hiroom":
            gt_depths = load_depths_hiroom(gt_data.aux.gt_depth_files, gt_data.aux.aliasing_mask_files)
        case _:
            raise NotImplementedError(f"GT depth loading not implemented for dataset: {dataset_name}")
    
    return gt_depths


def load_depths_7scenes(gt_depth_files):
    """
    Load GT depth with valid mask for 7Scenes.

    For 7Scenes, GT depth is stored as 16-bit PNG in millimeters.
    Value 65535 indicates invalid depth.

    NOTE: Copied from _load_gt_mask() in sevenscenes.py
    """
    gt_depths = []
    for f in gt_depth_files:
        # Load raw depth (16-bit PNG usually)
        gt_depth = cv2.imread(f, -1)
        
        # 65535 is invalid depth marker in 7Scenes
        gt_depth[gt_depth == 65535] = 0
        # Convert to meters
        gt_depth = gt_depth / 1000.0
  
        gt_depths.append(gt_depth)

    return gt_depths


def load_depths_eth3d(gt_image_files):
    """
    Load GT depth with valid mask for ETH3D.

    GT mask marks occluded or invalid regions that should be excluded from depth fusion and evaluation.

    NOTE: Copied from _load_gt_mask() in eth3d.py
    """
    h, w, _ = cv2.imread(gt_image_files[0]).shape

    gt_depths = []
    for img_path in gt_image_files:
        # Parse paths
        prefix, suffix = img_path.split("/images/")
        stem = suffix.rsplit(".", 1)[0]
        mask_path = f"{prefix}/masks_for_images/{stem}.png"
        depth_path = f"{prefix}/ground_truth_depth/{stem}.JPG"

        # Load GT depth
        if os.path.exists(depth_path):
            gt_depth = np.fromfile(depth_path, dtype=np.float32).reshape(h, w)
        else:
            raise ValueError(f"GT depth file not found: {depth_path}")

        # Load GT mask
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = np.asarray(gt_mask)
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Compute zero_mask
        # gt_mask == 1 means occluded/invalid region
        invalid_mask_from_gt = gt_mask == 1
        gt_depth_copy = gt_depth.copy()
        gt_depth_copy[gt_mask == 1] = 0
        invalid_mask_from_gt_depth = np.logical_or(gt_depth_copy == 0, gt_depth_copy == np.inf)

        # zero_mask: valid region that should be kept
        zero_mask = np.logical_and(
            np.logical_not(invalid_mask_from_gt),
            np.logical_not(invalid_mask_from_gt_depth)
        )
        gt_depth = gt_depth * zero_mask.astype(np.float32)

        gt_depths.append(gt_depth)


    return gt_depths


def load_depths_scannetpp(gt_depth_files):
    """
    Load GT depth with valid mask for ScanNet++.

    For ScanNet++, GT depth is stored as 16-bit PNG in millimeters.

    NOTE: Copied from _load_gt_mask() in scannetpp.py
    """
    # from DA3 constants: input resolution for ScanNet++ (after undistortion and resize)
    SCANNETPP_INPUT_H = 768
    SCANNETPP_INPUT_W = 1024

    gt_depths = []
    for f in gt_depth_files:
        gt_depth = imageio.imread(f) / 1000.0  # mm to meters

        # Resize to input resolution
        gt_depth = cv2.resize(
            gt_depth,
            (SCANNETPP_INPUT_W, SCANNETPP_INPUT_H),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        valid_mask = np.logical_and(gt_depth > 0, gt_depth != np.inf)
        gt_depth = gt_depth * valid_mask.astype(np.float32)

        gt_depths.append(gt_depth)

    return gt_depths


def load_depths_hiroom(gt_depth_files, aliasing_mask_files):
    """
    Load GT depth with valid mask for HiRoom.

    For HiRoom:
        - GT depth is stored as 16-bit PNG, scaled to 100m range
        - Aliasing mask marks regions to exclude

    NOTE: Copied from _load_gt_mask() in hiroom.py
    """
    gt_depths = []
    for depth_path, aliasing_mask_path in zip(gt_depth_files, aliasing_mask_files):
        gt_depth = cv2.imread(depth_path, -1) / 65535.0 * 100.0
        aliasing_mask = cv2.imread(aliasing_mask_path, -1) > 0

        valid_mask = np.logical_and(gt_depth > 0, gt_depth != np.inf)
        valid_mask = np.logical_and(valid_mask, np.logical_not(aliasing_mask))
        gt_depth = gt_depth * valid_mask.astype(np.float32)

        gt_depths.append(gt_depth)

    return gt_depths
