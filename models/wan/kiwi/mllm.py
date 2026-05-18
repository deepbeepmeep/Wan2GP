import gc
import hashlib
import importlib.util
import inspect
import os
import sys
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from mmgp import offload

from shared.utils import files_locator as fl
from shared.utils.utils import convert_tensor_to_image


class KiwiMLLMContextEncoder:
    def __init__(
        self,
        mllm_root_folder: str = "kiwi_mllm_encoder_instruct_reference",
        qwen_weights_path: Optional[str] = None,
        any_ref: bool = True,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        offload_after_encode: bool = True,
    ):
        self.mllm_root_folder = mllm_root_folder
        self.qwen_weights_path = qwen_weights_path
        self.any_ref = bool(any_ref)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dtype = dtype
        self.offload_after_encode = offload_after_encode
        self.managed_by_mmgp = False
        self.encoder = None
        self._module = None

    @contextmanager
    def _safe_print_context(self):
        # Some upstream Kiwi runtime prints emoji to stdout; on Windows cp1252 this can raise
        # UnicodeEncodeError and abort model initialization/inference.
        import builtins
        original_print = builtins.print

        def _safe_print(*args, **kwargs):
            try:
                return original_print(*args, **kwargs)
            except UnicodeEncodeError:
                encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                safe_args = []
                for arg in args:
                    text = str(arg)
                    text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
                    safe_args.append(text)
                return original_print(*safe_args, **kwargs)

        builtins.print = _safe_print
        try:
            yield
        finally:
            builtins.print = original_print

    def _patch_transformers_cache_compat(self):
        # Kiwi's packaged Qwen2.5-VL runtime targets newer Transformers cache APIs.
        # On older local versions, CacheLayerMixin.__init__ may reject max_cache_len.
        try:
            from transformers import cache_utils
        except Exception:
            return

        mixin_cls = getattr(cache_utils, "CacheLayerMixin", None)
        if mixin_cls is None:
            return
        if getattr(mixin_cls, "_kiwi_cache_compat_patch", False):
            return

        init_fn = getattr(mixin_cls, "__init__", None)
        if init_fn is None:
            return
        try:
            sig = inspect.signature(init_fn)
        except (TypeError, ValueError):
            return
        if "max_cache_len" in sig.parameters:
            return

        original_init = init_fn

        def _compat_init(self, *args, **kwargs):
            kwargs.pop("max_cache_len", None)
            kwargs.pop("max_batch_size", None)
            kwargs.pop("sliding_window", None)
            return original_init(self)

        mixin_cls.__init__ = _compat_init
        mixin_cls._kiwi_cache_compat_patch = True

    def _runtime_module_name(self) -> str:
        # Use a stable import name so Transformers/inspect can resolve class source files.
        folder = os.path.abspath(self.mllm_root_folder).replace("\\", "/").encode("utf-8")
        folder_id = hashlib.sha1(folder).hexdigest()[:12]
        return f"kiwi_mllm_encoder_runtime_{folder_id}"

    def _resolve_encoder_dir(self) -> Optional[str]:
        root = fl.locate_folder(self.mllm_root_folder, error_if_none=False)
        if root is None:
            return None
        if os.path.isfile(os.path.join(root, "config.json")):
            return root
        encoder_dir = os.path.join(root, "mllm_encoder")
        if os.path.isdir(encoder_dir) and os.path.isfile(os.path.join(encoder_dir, "config.json")):
            return encoder_dir
        return None

    def available(self) -> bool:
        encoder_dir = self._resolve_encoder_dir()
        if encoder_dir is None:
            return False
        required = (
            "config.json",
            "diffusion_pytorch_model.safetensors",
        )
        if not all(os.path.isfile(os.path.join(encoder_dir, filename)) for filename in required):
            return False
        return self._resolve_qwen_weights_path() is not None

    def _load_runtime_module(self, module_file: str):
        module_name = self._runtime_module_name()
        loaded = sys.modules.get(module_name)
        if loaded is not None and getattr(loaded, "__file__", None) == module_file:
            return loaded
        if loaded is not None and getattr(loaded, "__file__", None) != module_file:
            del sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import Kiwi MLLM module from {module_file}")
        module = importlib.util.module_from_spec(spec)
        # Register before execution so dynamic class inspection can resolve the module file.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    def _ensure_encoder(self):
        if self.encoder is not None:
            return
        self._patch_transformers_cache_compat()
        encoder_dir = self._resolve_encoder_dir()
        if encoder_dir is None:
            raise FileNotFoundError(
                f"Kiwi MLLM folder '{self.mllm_root_folder}' is missing under checkpoints."
            )
        module_file = os.path.join(os.path.dirname(__file__), "mllm_encoder.py")
        if not os.path.isfile(module_file):
            raise FileNotFoundError(f"Missing Kiwi MLLM runtime file: {module_file}")
        with self._safe_print_context():
            self._module = self._load_runtime_module(module_file)
            mllm_cls = getattr(self._module, "MLLMEncoder", None)
            if mllm_cls is None:
                raise RuntimeError("Kiwi mllm_encoder.py does not expose MLLMEncoder.")
            self.encoder = mllm_cls.from_pretrained(encoder_dir, torch_dtype=self.dtype, low_cpu_mem_usage=False, local_files_only=True, any_ref=self.any_ref)
            # Flat layout support: processor/tokenizer files can live directly in the variant folder.
            self.encoder._processor_path = encoder_dir
        self.encoder.eval().requires_grad_(False)
        target_device = torch.device("cpu") if self.managed_by_mmgp else self.device
        self.encoder.to(device=target_device, dtype=self.dtype)

    def _ensure_on_device(self):
        if self.encoder is None:
            return
        try:
            current_device = next(self.encoder.parameters()).device
        except StopIteration:
            current_device = self.device
        if current_device != self.device:
            self.encoder.to(device=self.device, dtype=self.dtype)

    def _resolve_qwen_weights_path(self) -> Optional[str]:
        path = self.qwen_weights_path
        if isinstance(path, str) and len(path) > 0:
            if os.path.isfile(path):
                return path
            resolved = fl.locate_file(path, error_if_none=False)
            if resolved is not None:
                return resolved
            resolved = fl.locate_file(os.path.basename(path), error_if_none=False)
            if resolved is not None:
                return resolved
        encoder_dir = self._resolve_encoder_dir()
        if encoder_dir is None:
            return None
        for filename in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
            candidate = os.path.join(encoder_dir, filename)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _load_qwen_with_mmgp(self, target_device: torch.device):
        if self.encoder is None:
            raise RuntimeError("Kiwi MLLM encoder is not initialized.")
        if getattr(self.encoder, "qwen_model", None) is not None:
            return
        if self._module is None:
            raise RuntimeError("Kiwi runtime module is missing.")

        from transformers import AutoProcessor

        qwen_model_cls = getattr(self._module, "Qwen2_5_VLForConditionalGeneration", None)
        if qwen_model_cls is None:
            raise RuntimeError("Kiwi runtime module does not expose Qwen2_5_VLForConditionalGeneration.")

        qwen_path = self.encoder._resolve_qwen_path()
        processor_path = self.encoder._resolve_processor_path()
        qwen_config_file = os.path.join(qwen_path, "qwen_config.json")
        if not os.path.isfile(qwen_config_file) and processor_path:
            qwen_config_file = os.path.join(processor_path, "qwen_config.json")
        if not os.path.isfile(qwen_config_file):
            extra = f" or {processor_path}" if processor_path else ""
            raise FileNotFoundError(f"qwen_config.json is missing under {qwen_path}{extra}.")

        qwen_weights_path = self._resolve_qwen_weights_path()
        if qwen_weights_path is None:
            raise FileNotFoundError("Merged Kiwi Qwen weights file is missing.")

        preprocess_qwen_sd = {
            "model.language_model": "language_model",
            "model.visual": "visual",
            "model": "language_model",
        }
        qwen_model = offload.fast_load_transformers_model(
            qwen_weights_path,
            modelClass=qwen_model_cls,
            writable_tensors=False,
            defaultConfigPath=qwen_config_file,
            forcedConfigPath=qwen_config_file,
            preprocess_sd=preprocess_qwen_sd,
        )
        qwen_model.eval().requires_grad_(False)
        qwen_model.to(device=target_device, dtype=self.dtype)
        self.encoder.qwen_model = qwen_model
        self.encoder.qwen_model.model.num_image_queries = self.encoder.num_image_queries
        self.encoder.qwen_model.num_image_queries = self.encoder.num_image_queries
        self.encoder.qwen_model.model.num_video_queries = self.encoder.num_video_queries
        self.encoder.qwen_model.num_video_queries = self.encoder.num_video_queries
        if getattr(self.encoder, "any_ref", True):
            self.encoder.qwen_model.model.num_ref_queries = self.encoder.num_ref_queries

        if getattr(self.encoder, "processor", None) is None:
            processor_base = processor_path or qwen_path
            for fname in ("tokenizer_config.json", "tokenizer.json", "preprocessor_config.json"):
                if not os.path.isfile(os.path.join(processor_base, fname)):
                    raise FileNotFoundError(f"Processor asset {fname} is missing under {processor_base}.")
            self.encoder.processor = AutoProcessor.from_pretrained(processor_base)

    def prepare_for_mmgp(self):
        self.managed_by_mmgp = True
        self._ensure_encoder()
        self.encoder.to(device=torch.device("cpu"), dtype=self.dtype)
        self._load_qwen_with_mmgp(target_device=torch.device("cpu"))
        qwen_model = getattr(self.encoder, "qwen_model", None)
        if qwen_model is not None:
            qwen_model.eval().requires_grad_(False)

    def _release_qwen(self):
        if self.encoder is None:
            return
        qwen_model = getattr(self.encoder, "qwen_model", None)
        if qwen_model is not None:
            try:
                qwen_model.to("cpu")
            except Exception:
                pass
            self.encoder.qwen_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_src_video_frames(self, input_frames: torch.Tensor, max_frames: int = 16) -> List[Image.Image]:
        frame_count = int(input_frames.shape[1])
        if frame_count <= max_frames:
            frame_ids = range(frame_count)
        else:
            frame_ids = np.linspace(0, frame_count - 1, max_frames, dtype=np.int64).tolist()
        return [convert_tensor_to_image(input_frames[:, frame_no].detach().to(device="cpu", dtype=torch.float32)) for frame_no in frame_ids]

    def _prepare_ref_image(self, input_ref_images, use_ref_image: bool):
        if not use_ref_image or input_ref_images is None:
            return None
        ref_image = input_ref_images[0] if isinstance(input_ref_images, (list, tuple)) else input_ref_images
        if torch.is_tensor(ref_image):
            ref_image = convert_tensor_to_image(ref_image)
        return ref_image

    @torch.no_grad()
    def encode_from_inputs(
        self,
        prompt: str,
        input_frames: torch.Tensor,
        input_ref_images=None,
        use_ref_image: bool = False,
        max_frames: int = 16,
    ) -> torch.Tensor:
        src_video_frames = self._prepare_src_video_frames(input_frames, max_frames=max_frames)
        ref_image = self._prepare_ref_image(input_ref_images, use_ref_image=use_ref_image)
        return self.encode(prompt, src_video_frames, ref_image=ref_image)

    @torch.no_grad()
    def encode(
        self,
        prompt: str,
        src_video_frames: List[Image.Image],
        ref_image: Optional[Image.Image] = None,
    ) -> torch.Tensor:
        if src_video_frames is None or len(src_video_frames) == 0:
            raise ValueError("Kiwi MLLM requires at least one source frame.")
        self._ensure_encoder()
        if getattr(self.encoder, "qwen_model", None) is None:
            target_device = torch.device("cpu") if self.managed_by_mmgp else self.device
            self._load_qwen_with_mmgp(target_device=target_device)
        if not self.managed_by_mmgp:
            self._ensure_on_device()
        if len(src_video_frames) == 1 and ref_image is None:
            with self._safe_print_context():
                context = self.encoder(prompt, src_image=src_video_frames)
        else:
            mllm_kwargs = {"src_video": src_video_frames}
            if ref_image is not None:
                mllm_kwargs["ref_image"] = [ref_image]
            with self._safe_print_context():
                context = self.encoder(prompt, **mllm_kwargs)
        context = context.to(device=self.device, dtype=self.dtype)
        if self.offload_after_encode and not self.managed_by_mmgp:
            self._release_qwen()
        return context
