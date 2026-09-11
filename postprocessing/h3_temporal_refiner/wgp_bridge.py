from __future__ import annotations

import os
from dataclasses import dataclass

from postprocessing.spatial_upsamplers import (PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_MEDIA_FLOW,
    PARAMETER_UI_POSTPROCESSING, POSTPROCESSING_CATEGORY_REFINER, SimpleScaleSuffixMixin,
    UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_TYPE_POSTPROCESSING, UPSAMPLER_TYPE_VAE)


NAME = "H3 Temporal Refiner"
PIXEL_METHOD = "h3temporal"
VAE_METHOD = "h3temporalvae"
MODEL_TYPE = "minimax_h3_ref2va_pruned"


@dataclass
class TemporalSession:
    handler: object
    strength: float
    maximum: int

    def refine(self, video, *, pipeline, latents, **kwargs):
        return self.handler.refine(video, pipeline=pipeline, latents=latents, strength=self.strength, maximum=self.maximum, **kwargs)


class H3TemporalRefinerBridge(SimpleScaleSuffixMixin):
    def __init__(self, server_config, files_locator):
        self.server_config = server_config
        self.files_locator = files_locator
        self.model = None
        self.offloadobj = None

    @classmethod
    def query_upsampler_def(cls):
        contexts = (PARAMETER_UI_POSTPROCESSING, PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_MEDIA_FLOW)
        parameters = [
            {"name": "spatial_upsampler_param", "setting": "strength", "type": "integer", "component": "slider",
             "ui": contexts, "required": False, "default": 50, "minimum": 0, "maximum": 100, "step": 1,
             "label": "Repair Strength", "label_long": "Repair Strength (%) - higher repairs more but changes more; 0 disables",
             "description": "Start at 50. Higher values regenerate more of the motion and take longer."},
            {"name": "spatial_upsampler_param2", "setting": "maximum", "type": "integer", "component": "slider",
             "ui": contexts, "required": False, "default": 4, "minimum": 1, "maximum": 4, "step": 1,
             "label": "Maximum Stretch", "label_long": "Maximum Stretch - higher costs more time and VRAM; 1 disables",
             "description": "Temporarily slow detected fast action by up to this factor, then restore the original timing."},
            {"name": "spatial_upsampler_prompt", "setting": "prompt", "type": "string", "component": "textbox",
             "ui": (PARAMETER_UI_LATE_POSTPROCESSING,), "required": False, "default": "", "label": "Refiner Prompt",
             "description": "Describe the source subjects, scene and action. The source prompt is reused when available."},
        ]
        return {"name": NAME, "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING, UPSAMPLER_TYPE_VAE),
                "media": ("video",), "profile": UPSAMPLER_PROFILE_VIDEO, "config_key": "h3_temporal_refiner", "pos": 45,
                "methods": [(NAME + " (Pixel)", PIXEL_METHOD)], "vae_methods": [(NAME + " (VAE)", VAE_METHOD)],
                "postprocessing_category": POSTPROCESSING_CATEGORY_REFINER, "progress_label": NAME,
                "source_audio_conditioning": True, "default_spatial_upsampling": PIXEL_METHOD,
                "description": "Repair smeared fast motion. Keeps duration, resolution and soundtrack; may change appearance. Experimental, especially at low resolution.",
                "method_descriptions": {
                    PIXEL_METHOD: "For existing videos or non-H3 generators (18+ frames). Loads H3 to repair smeared fast motion; keeps timing and soundtrack. May change appearance.",
                    VAE_METHOD: "For H3 generation. Reuses H3 latents to locate fast motion and repairs it before output. Keeps timing and soundtrack; may change appearance."},
                "method_parameters": {PIXEL_METHOD: parameters, VAE_METHOD: [dict(p, ui=(PARAMETER_UI_POSTPROCESSING,)) for p in parameters[:2]]}}

    def enabled(self):
        return True

    @classmethod
    def default_config(cls):
        return {"dyrope": False}

    @classmethod
    def normalize_config_section(cls, config):
        return {"dyrope": bool(config.get("dyrope", False))}

    def create_config_ui(self, gr, config, *, lock_config=False):
        control = gr.Checkbox(value=config["dyrope"], label="DyRoPE temporal repair (experimental)",
                             info="Use original timing during repair. May reduce changes in motion speed but add flicker. Applies to both input types.",
                             interactive=not lock_config)
        return [("dyrope", control)]

    def supports_model_vae_method(self, method, model_type, model_def, image_mode):
        return method == VAE_METHOD and image_mode == 0 and model_def.get("h3_temporal_refiner", False) and not model_def.get("audio_only", False)

    def validate_upsampling(self, value, image_mode):
        return "H3 Temporal Refiner requires video input" if image_mode else ""

    def prepare_vae_upsampler(self, value, *, spatial_upsampler_param=None, spatial_upsampler_param2=None, spatial_upsampler_parameters=None, **kwargs):
        parameters = spatial_upsampler_parameters or {}
        spatial_upsampler_param = parameters.get("spatial_upsampler_param", spatial_upsampler_param)
        spatial_upsampler_param2 = parameters.get("spatial_upsampler_param2", spatial_upsampler_param2)
        return TemporalSession(self, float(50 if spatial_upsampler_param is None else spatial_upsampler_param) / 100,
                               int(4 if spatial_upsampler_param2 is None else spatial_upsampler_param2))

    def download(self, process_files, **kwargs):
        import wgp
        from models.minimax_h3.minimax_h3_handler import family_handler

        definition = wgp.get_model_def(MODEL_TYPE).copy()
        definition.update(definition["system_configs"]["gguf_q4_k_m"])
        for download in family_handler.query_model_files(lambda filename: [os.path.basename(filename)], MODEL_TYPE, definition):
            process_files(**download)
        filename = wgp.get_model_filename(MODEL_TYPE, wgp.transformer_quantization, wgp.transformer_dtype_policy, model_def=definition)
        wgp.download_models(filename, MODEL_TYPE, 0, -1, model_def=definition)
        urls = wgp.get_model_recursive_prop(MODEL_TYPE, "text_encoder_URLs", return_list=True, model_def=definition)
        encoder = wgp.get_model_filename(MODEL_TYPE, wgp.text_encoder_quantization, wgp.transformer_dtype_policy, URLs=urls)
        wgp.download_models(encoder, MODEL_TYPE, 2, -1, force_path=definition["text_encoder_folder"], model_def=definition)
        return True

    def load_upsampler(self, value, *, process_files, **kwargs):
        self.download(process_files)

    def _load(self):
        if self.model is not None:
            return self.model
        import wgp
        from shared.utils import offload_registry

        self.model, self.offloadobj = wgp.load_models(MODEL_TYPE, override_profile=wgp.get_default_profile("video"),
                                                     output_type="video", config_id="gguf_q4_k_m", track_as_main=False,
                                                     disable_pinning=True)
        offload_registry.register_offloadobj(NAME, self.offloadobj, self.release_vram)
        return self.model

    def refine(self, sample, *, pipeline=None, **kwargs):
        from .repair import repair
        from postprocessing.spatial_upsamplers import read_config_section

        private = pipeline is None or pipeline.transformer.pdd_num_steps is not None or pipeline.fixed_prompt is not None
        if private:
            if pipeline is not None:
                import wgp
                source_pipeline = pipeline
                kwargs["abort_callback"] = lambda: source_pipeline._interrupt
                wgp.get_loaded_model_context().offloadobj.unload_all()
                self.download(wgp.process_files_def)
            pipeline = self._load()
            pipeline._interrupt = False
        try:
            return repair(sample, pipeline=pipeline, dyrope=read_config_section(self.server_config, self)["dyrope"], **kwargs)
        finally:
            if private:
                self.offloadobj.unload_all()

    def upscale(self, sample, value, *, prompt="", seed=0, fps=24, frame_offset=0, strength=50, maximum=4,
                audio_waveform=None, audio_sample_rate=32000, source_audio_path=None, vae_tile_size=None,
                abort_callback=None, progress_callback=None, **kwargs):
        if source_audio_path is not None:
            from shared.utils.audio_video import slice_audio_window
            audio_waveform, audio_sample_rate = slice_audio_window(source_audio_path, frame_offset, sample.shape[1], fps)
        result = self.refine(sample, strength=float(strength) / 100, maximum=int(maximum), prompt=prompt, seed=seed,
                             fps=fps, audio_waveform=audio_waveform, audio_sample_rate=audio_sample_rate,
                             vae_tile_size=vae_tile_size, abort_callback=abort_callback, progress_callback=progress_callback)
        return result, None

    def release_vram(self):
        from shared.utils import offload_registry

        if self.offloadobj is not None:
            offload_registry.unregister_offloadobj(NAME, self.offloadobj)
            self.offloadobj.release()
        self.model = self.offloadobj = None
