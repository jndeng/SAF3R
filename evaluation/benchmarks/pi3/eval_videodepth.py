"""
Modified from: https://github.com/yyfz/Pi3/blob/evaluation/videodepth/eval.py
"""
from addict import Dict

import numpy as np
import cv2
import torch

from .depth import depth_evaluation


def get_depth_eval_meta(dataset_name, align_method):
    # default config, to be updated based on dataset and alignment method
    eval_kwargs = {
        "max_depth": 80,
        "use_gpu": True,
        # different alignment methods for depth evaluation
        "align_with_lstsq": False,
        "align_with_lad": False,
        "align_with_lad2": False,
        "align_with_scale": False,
        "metric_scale": False,
    }

    # update dataset-specific settings
    if dataset_name.lower() == "7scenes":
        eval_kwargs["max_depth"] = 1000000.0  # no truncation (from DA3Bench constants)
    elif dataset_name.lower() == "eth3d":
        eval_kwargs["max_depth"] = 100000.0   # no truncation (from DA3Bench constants)
    elif dataset_name.lower() == "scannetpp":
        eval_kwargs["max_depth"] = 5.0        # Maximum depth for integration (meters) (from DA3Bench constants)
    elif dataset_name.lower() == "hiroom":
        eval_kwargs["max_depth"] = 10000.0    # no truncation (from DA3Bench constants)
    else:
        raise NotImplementedError(f"Depth evaluation not implemented for dataset: {dataset_name}")
    
    # update alignment method settings
    # NOTE: all false (default) is median scaling
    if align_method in eval_kwargs:
        eval_kwargs[align_method] = True

    return eval_kwargs


def evaluate_video_depth(pred_depths, gt_depths, dataset_name, align_method=""):
    """
    Evaluate depth predictions against GT for a video (image sequence).

    Args:
        pred_depths (list[np.ndarray] or np.ndarray):
            List of predicted depth maps [H_p, W_p]. Or stacked array [N, H_p, W_p].
        gt_depths (list[np.ndarray] or np.ndarray):
            List of GT depth maps [H_g, W_g]. Or stacked array [N, H_g, W_g].
        align_method (str):
            Alignment method to use for evaluation (see get_depth_eval_meta).
    
    Returns:
        metrics (dict): Averaged metrics dictionary.
    """
    if isinstance(gt_depths, np.ndarray):
        gt_depths = [g for g in gt_depths]
    if isinstance(pred_depths, np.ndarray):
        pred_depths = [p for p in pred_depths]
    assert len(pred_depths) == len(gt_depths), "Prediction and GT length mismatch"

    # Pack the pred & gt depths
    gt_stack = np.stack(gt_depths, axis=0) # [N, H, W]

    target_h, target_w = gt_stack.shape[1], gt_stack.shape[2]
    processed_preds = []
    for pred in pred_depths:
        if pred.shape[:2] != (target_h, target_w):
            resized_pred = cv2.resize(pred, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            resized_pred = pred
        processed_preds.append(resized_pred)
    pred_stack = np.stack(processed_preds, axis=0) # [N, H, W]

    # Evaluate depth with various metrics
    eval_kwargs = get_depth_eval_meta(dataset_name, align_method)
    metrics, _, _, _ = depth_evaluation(pred_stack, gt_stack, **eval_kwargs)
    
    return Dict(metrics)
