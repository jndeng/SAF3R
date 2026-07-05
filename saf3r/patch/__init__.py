from .patch_vggt import patch_model as patch_model_vggt
from .patch_vggt import update_args_per_forward as update_args_per_forward_vggt
from .patch_vggt import save_profile_results as save_profile_results_vggt
from .patch_pi3 import patch_model as patch_model_pi3
from .patch_pi3 import update_args_per_forward as update_args_per_forward_pi3
from .patch_pi3 import save_profile_results as save_profile_results_pi3
from .patch_da3 import patch_model as patch_model_da3
from .patch_da3 import update_args_per_forward as update_args_per_forward_da3
from .patch_da3 import save_profile_results as save_profile_results_da3


def patch_model(model, model_name, cfg):
    if "vggt" in model_name:
        model = patch_model_vggt(model, cfg)
    elif "da3" in model_name:
        model = patch_model_da3(model, cfg)
    elif "pi3" in model_name:
        model = patch_model_pi3(model, cfg)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model


def update_args_per_forward(model, model_name, img_h, img_w):
    if "vggt" in model_name:
        update_args_per_forward_vggt(model, img_h, img_w)
    elif "da3" in model_name:
        update_args_per_forward_da3(model, img_h, img_w)
    elif "pi3" in model_name:
        update_args_per_forward_pi3(model, img_h, img_w)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def save_profile_results(model, model_name, cfg):
    if "vggt" in model_name:
        save_profile_results_vggt(model, cfg.model.profile_save_path, cfg.model.patch_module.profile_metric)
    elif "da3" in model_name:
        save_profile_results_da3(model, cfg.model.profile_save_path, cfg.model.patch_module.profile_metric)
    elif "pi3" in model_name:
        save_profile_results_pi3(model, cfg.model.profile_save_path, cfg.model.patch_module.profile_metric)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
