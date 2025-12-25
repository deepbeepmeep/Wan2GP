import json
import logging
import struct
from typing import Optional, Union

import transformers
import torch
import torchvision
import torchvision.transforms.functional as F
from peft import PeftConfig, LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import ToPILImage

from .generation_utils import generate_sample, generate_sample_i2v, generate_sample_ti2i
from .magcache_utils import compute_magcache_threshold, set_magcache_params

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True


def read_safetensors_json(file_path):
    """Reads the metadata (JSON header) from a safetensors file."""
    with open(file_path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack("Q", header_size_bytes)[0]
        header_bytes = f.read(header_size)
        header_str = header_bytes.decode("utf-8")
        return json.loads(header_str)


def get_first_frame_from_image(image, vae, device, max_area, divisibility):
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError(f"unknown image type: {type(image)}")

    image = F.pil_to_tensor(pil_image).unsqueeze(0)
    k = 1.0
    image = image / 127.5 - 1.0

    with torch.no_grad():
        image = image.to(device=device, dtype=vae.dtype).transpose(0, 1).unsqueeze(0)
        enc_out = vae.encode(image)
        lat_image = (
            enc_out.latent_dist.sample()
            .squeeze(0)
            .permute(1, 2, 3, 0)
        )
        lat_image = lat_image * vae.config.scaling_factor

    return pil_image, lat_image, k


def find_nearest(available_res, real_res):
    nearest_index = torch.argmin(
        torch.tensor([abs((x[0] / x[1]) - (real_res[0] / real_res[1])) for x in available_res])
    ).item()
    return available_res[nearest_index]


class Kandinsky5Pipeline:
    VIDEO_RESOLUTIONS = {
        512: [(512, 512), (512, 768), (768, 512)],
        1024: [
            (1024, 1024),
            (640, 1408),
            (1408, 640),
            (768, 1280),
            (1280, 768),
            (896, 1152),
            (1152, 896),
        ],
    }
    IMAGE_RESOLUTIONS = {
        1024: [
            (1024, 1024),
            (640, 1408),
            (1408, 640),
            (768, 1280),
            (1280, 768),
            (896, 1152),
            (1152, 896),
        ],
    }
    VIDEO_SCALE_FACTORS = {
        512: (1.0, 2.0, 2.0),
        1024: (1.0, 3.16, 3.16),
    }
    IMAGE_SCALE_FACTORS = {
        512: (1.0, 2.0, 2.0),
        1024: (1.0, 2.0, 2.0),
    }
    VIDEO_BUCKET_AREAS = {
        512: 512 * 768,
        1024: 1024 * 1024,
    }
    IMAGE_BUCKET_AREAS = {
        512: 512 * 512,
        1024: 1024 * 1024,
    }

    def __init__(
        self,
        mode: str,
        device_map: Union[str, torch.device, dict],
        dit,
        text_embedder,
        vae,
        resolution: Optional[int] = None,
        conf=None,
    ):
        self.mode = mode
        self.dit = dit
        self.text_embedder = text_embedder
        self.vae = vae

        self.resolution = resolution if resolution is not None else conf.metrics.resolution

        self.device_map = device_map
        self.conf = conf
        self.num_steps = conf.model.num_steps
        self.guidance_weight = conf.model.guidance_weight

        self._interrupt = False
        if hasattr(self.dit, "_interrupt_check"):
            self.dit._interrupt_check = lambda: self._interrupt

        if self.mode == "i2v":
            self.max_area = 512 * 768 if self.resolution == 512 else 1024 * 1024
            self.divisibility = 16 if self.resolution == 512 else 128

        self._hf_peft_config_loaded = False
        self.peft_config = {}
        self.peft_triggers = {}
        self.peft_trigger = ""

    def _resolve_resolution_bucket(self, width, height, bucket_areas):
        if width is None or height is None:
            return None
        area = width * height
        return min(bucket_areas, key=lambda res: abs(area - bucket_areas[res]))

    def _apply_runtime_resolution(self, width, height):
        is_video = self.mode in ("t2v", "i2v")
        bucket_areas = self.VIDEO_BUCKET_AREAS if is_video else self.IMAGE_BUCKET_AREAS
        runtime_res = self._resolve_resolution_bucket(width, height, bucket_areas)
        if runtime_res is None or runtime_res == self.resolution:
            return
        self.resolution = runtime_res
        if hasattr(self.conf, "metrics"):
            self.conf.metrics.resolution = runtime_res
            scale_factor = (
                self.VIDEO_SCALE_FACTORS.get(runtime_res)
                if is_video
                else self.IMAGE_SCALE_FACTORS.get(runtime_res)
            )
            if scale_factor is not None:
                self.conf.metrics.scale_factor = list(scale_factor)
        if self.mode == "i2v":
            self.max_area = 512 * 768 if runtime_res == 512 else 1024 * 1024
            self.divisibility = 16 if runtime_res == 512 else 128

    def _disable_magcache(self):
        if getattr(self.dit, "_magcache_enabled", False):
            orig_forward = getattr(self.dit, "_magcache_orig_forward", None)
            if orig_forward is not None:
                self.dit.forward = orig_forward
            self.dit._magcache_enabled = False

    def _apply_magcache(self, num_steps, guidance_weight):
        cache = getattr(self.dit, "cache", None)
        if cache is None or getattr(cache, "cache_type", None) != "mag":
            self._disable_magcache()
            return False
        mag_ratios = getattr(cache, "def_mag_ratios", None)
        if not mag_ratios:
            self._disable_magcache()
            return False
        if not hasattr(self.dit, "_magcache_orig_forward"):
            self.dit._magcache_orig_forward = self.dit.forward
        no_cfg = abs(guidance_weight - 1.0) <= 1e-6
        speed_factor = getattr(cache, "multiplier", None)
        start_step = getattr(cache, "start_step", 0)
        magcache_K = getattr(cache, "magcache_K", 2)
        retention_ratio = getattr(cache, "magcache_retention_ratio", 0.2)
        magcache_thresh = getattr(cache, "magcache_thresh", None)
        if speed_factor is not None and speed_factor > 0:
            magcache_thresh = compute_magcache_threshold(
                mag_ratios,
                num_steps,
                speed_factor,
                start_step=start_step,
                no_cfg=no_cfg,
                magcache_K=magcache_K,
                retention_ratio=retention_ratio,
            )
            cache.magcache_thresh = magcache_thresh
        set_magcache_params(
            self.dit,
            mag_ratios,
            num_steps,
            no_cfg,
            start_step=start_step,
            magcache_thresh=magcache_thresh,
            magcache_K=magcache_K,
            retention_ratio=retention_ratio,
        )
        self.dit._magcache_enabled = True
        return True

    def _expand_prompt_t2v(self, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are a prompt beautifier that transforms short user video descriptions into rich, detailed English prompts specifically optimized for video generation models.
Here are some example descriptions from the dataset that the model was trained:
1. "In a dimly lit room with a cluttered background, papers are pinned to the wall and various objects rest on a desk. Three men stand present: one wearing a red sweater, another in a black sweater, and the third in a gray shirt. The man in the gray shirt speaks and makes hand gestures, while the other two men look forward. The camera remains stationary, focusing on the three men throughout the sequence. A gritty and realistic visual style prevails, marked by a greenish tint that contributes to a moody atmosphere. Low lighting casts shadows, enhancing the tense mood of the scene."
2. "In an office setting, a man sits at a desk wearing a gray sweater and seated in a black office chair. A wooden cabinet with framed pictures stands beside him, alongside a small plant and a lit desk lamp. Engaged in a conversation, he makes various hand gestures to emphasize his points. His hands move in different positions, indicating different ideas or points. The camera remains stationary, focusing on the man throughout. Warm lighting creates a cozy atmosphere. The man appears to be explaining something. The overall visual style is professional and polished, suitable for a business or educational context."
3. "A person works on a wooden object resembling a sunburst pattern, holding it in their left hand while using their right hand to insert a thin wire into the gaps between the wooden pieces. The background features a natural outdoor setting with greenery and a tree trunk visible. The camera stays focused on the hands and the wooden object throughout, capturing the detailed process of assembling the wooden structure. The person carefully threads the wire through the gaps, ensuring the wooden pieces are securely fastened together. The scene unfolds with a naturalistic and instructional style, emphasizing the craftsmanship and the methodical steps taken to complete the task."
IImportantly! These are just examples from a large training dataset of 200 million videos.
Rewrite Prompt: "{prompt}" to get high-quality video generation. Answer only with expanded prompt.""",
                    },
                ],
            }
        ]
        text = self.text_embedder.embedder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.text_embedder.embedder.processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.text_embedder.embedder.model.device)
        generated_ids = self.text_embedder.embedder.model.generate(
            **inputs, max_new_tokens=256
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.text_embedder.embedder.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def _expand_prompt_t2i(self, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Rewrite and enhance the original prompt with richer detail, clearer structure, and improved descriptive quality. Expand the scene, atmosphere, and context while preserving the user's intent. When adding text that should appear inside an image, place that text inside double quotes and in capital letters. Strengthen visual clarity, style, and specificity, but do not change the meaning. Output only the enhanced prompt, written in polished, vivid language suitable for high-quality image generation.
example:
Original text: white mini police car with blue stripes, with 911 and 'Police' text
Result: A miniature model car simulating the official transport of the relevant authorities. The body is white with blue stripes. The word "911" is written in large blue letters on the hood and side. Below it, "POLICE" is used in a font. The windows are transparent, and the interior has black seats. The headlights have plastic lenses, and the roof has blue and red beacons. The radiator grille has vertical slots. The wheels are black with white rims. The doors are closed, the windows have black frames. The background is uniform white.
Here 911 in double quotes because it is text on image, 'Police' -> "POLICE" because it should be in double quotes and capital letters.
Rewrite Prompt: "{prompt}". Answer only with expanded prompt.""",
                    },
                ],
            }
        ]
        text = self.text_embedder.embedder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.text_embedder.embedder.processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.text_embedder.embedder.model.device)
        generated_ids = self.text_embedder.embedder.model.generate(
            **inputs, max_new_tokens=256
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.text_embedder.embedder.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def _expand_prompt_i2i(self, prompt, image):
        width, height = image.size
        image = image.resize((width // 4, height // 4))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Rewrite and enhance the original editing instruction with richer detail, clearer structure, and improved descriptive quality. When adding text that should appear inside an image, place that text inside double quotes and in capital letters. Explain what needs to be changed and what needs to be left unchanged. Explain in details how to change camera position or tell that camera position shouldn't be changed.
example:
Original text: add text 911 and 'Police'
Result: Add the word "911" in large blue letters to the hood. Below that, add the word "POLICE." Keep the camera position unchanged, as do the background, car position, and lighting.
Rewrite Prompt: "{prompt}". Answer only with expanded prompt.""",
                    },
                    {
                        "type": "image",
                        "image": image,
                    },
                ],
            }
        ]
        text = self.text_embedder.embedder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.text_embedder.embedder.processor(
            text=[text],
            images=image,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.text_embedder.embedder.model.device)
        generated_ids = self.text_embedder.embedder.model.generate(
            **inputs, max_new_tokens=256
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.text_embedder.embedder.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def _prepare_seed(self, seed):
        if seed is not None:
            return seed
        seed = torch.randint(2**32 - 1, (1,), device="cpu")
        return int(seed.item())

    def _maybe_expand_prompt(self, caption, seed, image=None):
        if not caption:
            return caption
        transformers.set_seed(seed)
        if self.mode == "t2v":
            caption = self.peft_trigger + self._expand_prompt_t2v(caption)
        elif self.mode == "i2v":
            caption = self.text_embedder.embedder.expand_text_prompt(
                caption, image, device=self.device_map["text_embedder"]
            )
            caption = self.peft_trigger + caption
        elif self.mode == "t2i":
            caption = self._expand_prompt_t2i(caption)
        elif self.mode == "i2i":
            caption = self._expand_prompt_i2i(caption, image=image)
        return caption

    def __call__(
        self,
        text: str,
        image: Optional[Union[str, Image.Image]] = None,
        time_length: int = 5,
        width: Optional[int] = None,
        height: Optional[int] = None,
        frame_num: Optional[int] = None,
        seed: int = None,
        num_steps: int = None,
        guidance_weight: float = None,
        scheduler_scale: Optional[float] = None,
        negative_caption: str = "",
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
        callback=None,
        joint_pass: bool = False,
    ):
        num_steps = self.num_steps if num_steps is None else num_steps
        guidance_weight = self.guidance_weight if guidance_weight is None else guidance_weight
        if scheduler_scale is None:
            scheduler_scale = 10.0 if self.mode in ("t2v", "i2v") else 3.0

        seed = self._prepare_seed(seed)
        self._apply_runtime_resolution(width, height)
        magcache_enabled = self._apply_magcache(num_steps, guidance_weight)

        try:
            if self.mode in ("t2v", "i2v"):
                temporal_ratio = getattr(self.vae, "time_compression_ratio", None)
                if temporal_ratio is None:
                    temporal_ratio = getattr(getattr(self.vae, "config", None), "time_compression_ratio", None)
                if temporal_ratio is None:
                    temporal_ratio = 4
                if frame_num is None:
                    num_frames = 1 if time_length == 0 else time_length * 24 // temporal_ratio + 1
                else:
                    if frame_num <= 0:
                        raise ValueError("frame_num must be a positive integer")
                    if (frame_num - 1) % temporal_ratio != 0:
                        raise ValueError(
                            f"frame_num-1 must be a multiple of {temporal_ratio}, got {frame_num}"
                        )
                    # Convert output frame count to latent frame count.
                    num_frames = (frame_num - 1) // temporal_ratio + 1

            if self.mode == "t2v":
                if width is None or height is None:
                    raise ValueError("width and height must be provided for t2v generation")
                caption = text
                if expand_prompts:
                    caption = self._maybe_expand_prompt(caption, seed)
                shape = (1, num_frames, height // 8, width // 8, 16)
                images = generate_sample(
                    shape,
                    caption,
                    self.dit,
                    self.vae,
                    self.conf,
                    text_embedder=self.text_embedder,
                    num_steps=num_steps,
                    guidance_weight=guidance_weight,
                    scheduler_scale=scheduler_scale,
                    negative_caption=negative_caption,
                    seed=seed,
                    device=self.device_map["dit"],
                    vae_device=self.device_map["vae"],
                    text_embedder_device=self.device_map["text_embedder"],
                    progress=progress,
                    callback=callback,
                    joint_pass=joint_pass,
                    interrupt_check=lambda: self._interrupt,
                )
                if images is None:
                    return None
            elif self.mode == "i2v":
                if image is None:
                    raise ValueError("image is required for i2v generation")
                pil_image, image_lat, k = get_first_frame_from_image(
                    image, self.vae, self.device_map["vae"], self.max_area, self.divisibility
                )

                caption = text
                if expand_prompts:
                    caption = self._maybe_expand_prompt(caption, seed, image=pil_image)

                height_lat, width_lat = image_lat.shape[1:3]
                shape = (1, num_frames, height_lat, width_lat, 16)
                images = generate_sample_i2v(
                    shape,
                    caption,
                    self.dit,
                    self.vae,
                    self.conf,
                    text_embedder=self.text_embedder,
                    images=image_lat,
                    num_steps=num_steps,
                    guidance_weight=guidance_weight,
                    scheduler_scale=scheduler_scale,
                    negative_caption=negative_caption,
                    seed=seed,
                    device=self.device_map["dit"],
                    vae_device=self.device_map["vae"],
                    progress=progress,
                    callback=callback,
                    joint_pass=joint_pass,
                    interrupt_check=lambda: self._interrupt,
                )
                if images is None:
                    return None
                if k > 16:
                    h, w = images.shape[-2:]
                    images = F.resize(images[0], (int(h / k / 16), int(w / k / 16)))
            else:
                if width is None or height is None:
                    if image is None:
                        raise ValueError("width/height or image is required for image generation")
                    if isinstance(image, str):
                        image_obj = Image.open(image)
                    else:
                        image_obj = image
                    height, width = find_nearest(
                        self.IMAGE_RESOLUTIONS[self.resolution], image_obj.size[::-1]
                    )

                caption = text
                image_tensor = None
                if image is not None:
                    if isinstance(image, str):
                        image_obj = Image.open(image)
                    elif isinstance(image, Image.Image):
                        image_obj = image
                    else:
                        raise ValueError(f"unknown image type: {type(image)}")
                    image_tensor = F.pil_to_tensor(image_obj)[None]
                if expand_prompts:
                    caption = self._maybe_expand_prompt(caption, seed, image=image_obj if image is not None else None)

                shape = (1, 1, height // 8, width // 8, 16)
                images = generate_sample_ti2i(
                    shape,
                    caption,
                    self.dit,
                    self.vae,
                    self.conf,
                    text_embedder=self.text_embedder,
                    num_steps=num_steps,
                    guidance_weight=guidance_weight,
                    scheduler_scale=scheduler_scale,
                    negative_caption=negative_caption,
                    seed=seed,
                    device=self.device_map["dit"],
                    vae_device=self.device_map["vae"],
                    text_embedder_device=self.device_map["text_embedder"],
                    progress=progress,
                    image_vae=False,
                    image=image_tensor,
                    callback=callback,
                    joint_pass=joint_pass,
                    interrupt_check=lambda: self._interrupt,
                )
                if images is None:
                    return None
        finally:
            if magcache_enabled:
                self._disable_magcache()


        if self.mode in ("t2v", "i2v") and time_length != 0:
            if save_path is not None:
                if isinstance(save_path, str):
                    save_path = [save_path]
                if len(save_path) == len(images):
                    for path, video in zip(save_path, images):
                        torchvision.io.write_video(
                            path,
                            video.float().permute(1, 2, 3, 0).cpu().numpy(),
                            fps=24,
                            options={"crf": "5"},
                        )
            return images

        return_images = []
        for image in images.squeeze(2).cpu():
            return_images.append(ToPILImage()(image))
        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
            if len(save_path) == len(return_images):
                for path, out_image in zip(save_path, return_images):
                    out_image.save(path)
        return return_images

    def load_adapter(
        self,
        adapter_config: Union[PeftConfig, str],
        adapter_path: Optional[str] = None,
        adapter_name: Optional[str] = None,
        trigger: Optional[str] = None,
    ) -> None:
        if adapter_name is None:
            adapter_name = "default"
        if self._hf_peft_config_loaded and adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            try:
                with open(adapter_config, "r") as f:
                    adapter_config = json.load(f)
                adapter_config = LoraConfig(**adapter_config)
            except Exception as exc:
                raise TypeError(
                    "adapter_config should be an instance of PeftConfig or a path to a json file."
                ) from exc
        self.peft_config[adapter_name] = adapter_config

        inject_adapter_in_model(adapter_config, self.dit, adapter_name)

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        adapter_state_dict = load_file(adapter_path)
        adapter_metadata = read_safetensors_json(adapter_path)
        if trigger is not None:
            self.peft_trigger = trigger
        else:
            if "__metadata__" in adapter_metadata and "trigger" in adapter_metadata["__metadata__"]:
                self.peft_trigger = adapter_metadata["__metadata__"]["trigger"]
            else:
                self.peft_trigger = ""
        self.peft_triggers[adapter_name] = self.peft_trigger

        processed_adapter_state_dict = {}
        for key, value in adapter_state_dict.items():
            new_key = key
            for prefix in ["base_model.model.", "transformer."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    break

            new_key = new_key.replace(".default", "")
            processed_adapter_state_dict[new_key] = value

        incompatible_keys = set_peft_model_state_dict(
            self.dit, processed_adapter_state_dict, adapter_name
        )

        if incompatible_keys is not None:
            err_msg = ""
            if hasattr(incompatible_keys, "unexpected_keys") and len(incompatible_keys.unexpected_keys) > 0:
                err_msg = (
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f"{', '.join(incompatible_keys.unexpected_keys)}. "
                )

            missing_keys = getattr(incompatible_keys, "missing_keys", None)
            if missing_keys:
                lora_missing_keys = [k for k in missing_keys if "lora_" in k and adapter_name in k]
                if lora_missing_keys:
                    err_msg += (
                        f"Loading adapter weights from state_dict led to missing keys in the model: "
                        f"{', '.join(lora_missing_keys)}"
                    )

            if err_msg:
                logging.warning(err_msg)

        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: Union[list[str], str]) -> None:
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        elif isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(
                    "Following adapter(s) could not be found: "
                    f"{', '.join(missing)}. "
                    f"current loaded adapters are: {list(self.peft_config.keys())}"
                )
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        adapters_set = False

        for _, module in self.dit.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    module.disable_adapters = False
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                adapters_set = True

        if not adapters_set:
            raise ValueError(
                "Did not succeed in setting the adapter. Please make sure you are using a model that supports adapters."
            )
        self.peft_trigger = self.peft_triggers[adapter_name]

    def disable_adapters(self) -> None:
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        for _, module in self.dit.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    module.disable_adapters = True
        self.peft_trigger = ""
