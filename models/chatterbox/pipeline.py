from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from shared.utils import files_locator as fl 
import torch

from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES


class ChatterboxPipeline:
    """
    Thin wrapper around Chatterbox's multilingual TTS to fit WanGP's model API expectations.
    """

    def __init__(
        self,
        ckpt_root: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_root = Path(ckpt_root) if ckpt_root is not None else None
        self.model = self._load_model()
        self.model.ve._model_dtype = torch.float32
        self.model.s3gen._model_dtype = torch.float32
        self.model.t3._model_dtype = torch.float32
        self.model.conds._model_dtype = torch.float32

        self.sr = getattr(self.model, "sr", 44100)
        self._interrupt = False

    @property
    def supported_languages(self):
        return SUPPORTED_LANGUAGES

    def _load_model(self):
        return ChatterboxMultilingualTTS.from_local(self.ckpt_root, device=self.device)
        # return ChatterboxMultilingualTTS.from_pretrained(device=self.device)

    def prepare_reference(self, audio_prompt_path: Optional[str], exaggeration: float) -> None:
        if not audio_prompt_path:
            return
        if not os.path.isfile(audio_prompt_path):
            raise FileNotFoundError(f"Audio prompt file '{audio_prompt_path}' not found.")
        self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)

    def generate(
        self,
        input_prompt: str,
        model_mode: Optional[str],
        audio_guide: Optional[str],
        *,
        exaggeration: float,
        pace: float,
        temperature: float,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        **bkwargs
    ) -> torch.Tensor:
        text = input_prompt
        if not text or not text.strip():
            raise ValueError("Prompt text cannot be empty for Chatterbox generation.")

        language_id = model_mode
        cfg_weight = pace
        if language_id:
            language_id = language_id.lower()
            if language_id not in SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"Unsupported language '{language_id}'. "
                    f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}"
                )


        self.prepare_reference(audio_guide, exaggeration)
        wav = self.model.generate(
            text=text,
            language_id=language_id,
            audio_prompt_path=audio_guide,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
        return {"x": wav, "audio_sampling_rate": self.sr } 

    def release(self) -> None:
        if hasattr(self.model, "to"):
            self.model.to("cpu")
        self.model = None
        torch.cuda.empty_cache()
