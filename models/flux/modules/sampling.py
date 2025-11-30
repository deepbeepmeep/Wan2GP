import math

import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch import Tensor

# from .model import Flux2


def compress_time(t_ids: Tensor) -> Tensor:
    assert t_ids.ndim == 1
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(
        len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype
    )
    t_ids_compressed = t_remap[t_ids]
    return t_ids_compressed


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    """
    using position ids to scatter tokens into place
    """
    x_list = []
    t_coords = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape  # noqa: F841
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = compress_time(t_ids)

        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids

        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
        t_coords.append(torch.unique(t_ids, sorted=True))
    return x_list


def encode_image_refs(ae, img_ctx: list[Image.Image]):
    scale = 10

    if len(img_ctx) > 1:
        limit_pixels = 1024**2
    elif len(img_ctx) == 1:
        limit_pixels = 2024**2
    else:
        limit_pixels = None

    if not img_ctx:
        return None, None

    img_ctx_prep = default_prep(img=img_ctx, limit_pixels=limit_pixels)
    if not isinstance(img_ctx_prep, list):
        img_ctx_prep = [img_ctx_prep]

    # Encode each reference image
    encoded_refs = []
    for img in img_ctx_prep:
        encoded = ae.encode(img[None].cuda())[0]
        encoded_refs.append(encoded)

    # Create time offsets for each reference
    t_off = [scale + scale * t for t in torch.arange(0, len(encoded_refs))]
    t_off = [t.view(-1) for t in t_off]

    # Process with position IDs
    ref_tokens, ref_ids = listed_prc_img(encoded_refs, t_coord=t_off)

    # Concatenate all references along sequence dimension
    ref_tokens = torch.cat(ref_tokens, dim=0)  # (total_ref_tokens, C)
    ref_ids = torch.cat(ref_ids, dim=0)  # (total_ref_tokens, 4)

    # Add batch dimension
    ref_tokens = ref_tokens.unsqueeze(0)  # (1, total_ref_tokens, C)
    ref_ids = ref_ids.unsqueeze(0)  # (1, total_ref_tokens, 4)

    return ref_tokens.to(torch.bfloat16), ref_ids


def prc_txt(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _l, _ = x.shape  # noqa: F841

    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(1),  # dummy dimension
        "w": torch.arange(1),  # dummy dimension
        "l": torch.arange(_l),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids.to(x.device)


def batched_wrapper(fn):
    def batched_prc(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)
        return torch.stack(x), torch.stack(x_ids)

    return batched_prc


def listed_wrapper(fn):
    def listed_prc(
        x: list[Tensor],
        t_coord: list[Tensor] | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)
        return list(x), list(x_ids)

    return listed_prc


def prc_img(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _, h, w = x.shape  # noqa: F841
    x_coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(h),
        "w": torch.arange(w),
        "l": torch.arange(1),
    }
    x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids.to(x.device)


listed_prc_img = listed_wrapper(prc_img)
batched_prc_img = batched_wrapper(prc_img)
batched_prc_txt = batched_wrapper(prc_txt)


def center_crop_to_multiple_of_x(
    img: Image.Image | list[Image.Image], x: int
) -> Image.Image | list[Image.Image]:
    if isinstance(img, list):
        return [center_crop_to_multiple_of_x(_img, x) for _img in img]  # type: ignore

    w, h = img.size
    new_w = (w // x) * x
    new_h = (h // x) * x

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    resized = img.crop((left, top, right, bottom))
    return resized


def cap_pixels(img: Image.Image | list[Image.Image], k):
    if isinstance(img, list):
        return [cap_pixels(_img, k) for _img in img]
    w, h = img.size
    pixel_count = w * h

    if pixel_count <= k:
        return img

    # Scaling factor to reduce total pixels below K
    scale = math.sqrt(k / pixel_count)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def cap_min_pixels(img: Image.Image | list[Image.Image], max_ar=8, min_sidelength=64):
    if isinstance(img, list):
        return [cap_min_pixels(_img, max_ar=max_ar, min_sidelength=min_sidelength) for _img in img]
    w, h = img.size
    if w < min_sidelength or h < min_sidelength:
        raise ValueError(f"Skipping due to minimal sidelength underschritten h {h} w {w}")
    if w / h > max_ar or h / w > max_ar:
        raise ValueError(f"Skipping due to maximal ar overschritten h {h} w {w}")
    return img


def to_rgb(img: Image.Image | list[Image.Image]):
    if isinstance(img, list):
        return [
            to_rgb(
                _img,
            )
            for _img in img
        ]
    return img.convert("RGB")


def default_images_prep(
    x: Image.Image | list[Image.Image],
) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(x, list):
        return [default_images_prep(e) for e in x]  # type: ignore
    x_tensor = torchvision.transforms.ToTensor()(x)
    return 2 * x_tensor - 1


def default_prep(
    img: Image.Image | list[Image.Image], limit_pixels: int | None, ensure_multiple: int = 16
) -> torch.Tensor | list[torch.Tensor]:
    img_rgb = to_rgb(img)
    img_min = cap_min_pixels(img_rgb)  # type: ignore
    if limit_pixels is not None:
        img_cap = cap_pixels(img_min, limit_pixels)  # type: ignore
    else:
        img_cap = img_min
    img_crop = center_crop_to_multiple_of_x(img_cap, ensure_multiple)  # type: ignore
    img_tensor = default_images_prep(img_crop)
    return img_tensor


def generalized_time_snr_shift(t: Tensor, mu: float, sigma: float) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)



def concatenate_images(
    images: list[Image.Image],
) -> Image.Image:
    """
    Concatenate a list of PIL images horizontally with center alignment and white background.
    """

    # If only one image, return a copy of it
    if len(images) == 1:
        return images[0].copy()

    # Convert all images to RGB if not already
    images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

    # Calculate dimensions for horizontal concatenation
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create new image with white background
    background_color = (255, 255, 255)
    new_img = Image.new("RGB", (total_width, max_height), background_color)

    # Paste images with center alignment
    x_offset = 0
    for img in images:
        y_offset = (max_height - img.height) // 2
        new_img.paste(img, (x_offset, y_offset))
        x_offset += img.width

    return new_img
