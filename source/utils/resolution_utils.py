"""Resolution parsing and snapping utilities."""

from source.runtime.wgp_bridge import get_model_def

__all__ = [
    "snap_resolution_to_model_grid",
    "parse_resolution",
]


def snap_resolution_to_model_grid(parsed_res: tuple[int, int], grid_size: int = 16) -> tuple[int, int]:
    """
    Snaps resolution to model grid requirements.

    Args:
        parsed_res: (width, height) tuple
        grid_size: Grid alignment size in pixels (16 for Wan, 64 for LTX-2)

    Returns:
        (width, height) tuple snapped to nearest valid values
    """
    width, height = parsed_res
    width = (width // grid_size) * grid_size
    height = (height // grid_size) * grid_size
    return width, height


def get_model_grid_size(model_name: str | None) -> int:
    """Get the VAE block size / grid alignment for a model.

    Queries WGP's model handler when available, falls back to 16 (Wan default).
    """
    if not model_name:
        return 16
    try:
        model_def = get_model_def(model_name)
        if model_def is None:
            return 16
        # Check model family handler for vae_block_size
        family = model_def.get("family")
        if family == "ltx2":
            return 64
    except (ImportError, TypeError, ValueError, KeyError, AttributeError):
        pass
    # Fallback: check model name heuristically
    if model_name and "ltx2" in model_name.lower():
        return 64
    return 16


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parses 'WIDTHxHEIGHT' string to (width, height) tuple."""
    try:
        w, h = map(int, res_str.split('x'))
        if w <= 0 or h <= 0:
            raise ValueError("Width and height must be positive.")
        return w, h
    except ValueError as e:
        raise ValueError(f"Resolution string must be in WIDTHxHEIGHT format with positive integers (e.g., '960x544'), got {res_str}. Error: {e}") from e
