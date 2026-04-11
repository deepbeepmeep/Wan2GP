import copy
import traceback
from pathlib import Path
from source.core.log import headless_logger
from source.runtime.wgp_bridge import get_wgp_runtime_module_mutable
from source.runtime.process_globals import temporary_process_globals

def apply_phase_config_patch(parsed_phase_config: dict, model_name: str, task_id: str):
    """
    Apply phase_config patches to the model definition in WGP.
    This must be called AFTER WGP is initialized (right before generation).
    """
    if not parsed_phase_config.get("_patch_config"):
        return

    try:
        wan_dir = Path(__file__).parent.parent.parent.parent / "Wan2GP"
        with temporary_process_globals(
            cwd=str(wan_dir),
            argv=["apply_phase_config_patch.py"],
            prepend_sys_path=str(wan_dir),
        ):
            wgp = get_wgp_runtime_module_mutable()

            if model_name not in wgp.models_def:
                model_def = wgp.get_model_def(model_name)
                if not model_def:
                    return
                
                if model_name not in wgp.models_def:
                    wgp.models_def[model_name] = wgp.init_model_def(model_name, model_def)

            if model_name in wgp.models_def:
                parsed_phase_config["_original_model_def"] = copy.deepcopy(wgp.models_def[model_name])
                parsed_phase_config["_model_was_patched"] = True

                patch_config = parsed_phase_config["_patch_config"]
                temp_model_def = copy.deepcopy(patch_config["model"])
                temp_settings = copy.deepcopy(patch_config)
                del temp_settings["model"]
                temp_model_def["settings"] = temp_settings

                wgp.models_def[model_name] = temp_model_def
                temp_model_def = wgp.init_model_def(model_name, temp_model_def)
                wgp.models_def[model_name] = temp_model_def

                headless_logger.info(
                    f"Patched wgp.models_def['{model_name}'] in memory",
                    task_id=task_id
                )
    except (RuntimeError, ValueError, OSError) as e:
        headless_logger.warning(f"Failed to apply phase_config patch: {e}", task_id=task_id)
        headless_logger.debug(f"Patch error traceback: {traceback.format_exc()}", task_id=task_id)


def restore_model_patches(parsed_phase_config: dict, model_name: str, task_id: str):
    """
    Restore the original model definition after phase_config patching.
    """
    if not parsed_phase_config.get("_model_was_patched"):
        return

    try:
        wan_dir_path = str(Path(__file__).parent.parent.parent.parent / "Wan2GP")
        with temporary_process_globals(prepend_sys_path=wan_dir_path):
            wgp = get_wgp_runtime_module_mutable()

            if "_original_model_def" in parsed_phase_config and model_name in wgp.models_def:
                wgp.models_def[model_name] = parsed_phase_config["_original_model_def"]
                headless_logger.info(
                    f"Restored original wgp.models_def['{model_name}']",
                    task_id=task_id
                )
    except (RuntimeError, ValueError, OSError) as e:
        headless_logger.warning(f"Failed to restore model patches: {e}", task_id=task_id)
