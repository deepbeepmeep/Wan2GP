import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import copy
import json
from shared.deepy.engine import get_or_create_assistant_session
from shared.gradio import assistant_chat, gradio_queue_focus_patch
from shared.gradio.hierarchy_selector import HierarchySelector
from shared import notifications
from shared.utils import prompt_parser
from shared.gradio.model_selector_toolbar import unload_models_from_ram
from shared.utils.video_codecs import SDR_VIDEO_CODEC_CHOICES, VIDEO_CONTAINER_CHOICES, validate_video_output_settings
from shared.deepy.config import (
    DEEPY_ALLOW_READ_FILE_SYSTEM_DEFAULT,
    DEEPY_ALLOW_READ_FILE_SYSTEM_KEY,
    DEEPY_FILE_SYSTEM_ACCESS_DISABLED,
    DEEPY_FILE_SYSTEM_ACCESS_READ,
    DEEPY_FILE_SYSTEM_ACCESS_READ_WRITE,
    DEEPY_FILE_SYSTEM_PATHS_DEFAULT,
    DEEPY_FILE_SYSTEM_PATHS_KEY,
    DEEPY_READ_EVERYWHERE_DEFAULT,
    DEEPY_READ_EVERYWHERE_KEY,
    DEEPY_COMPACTION_TYPE_DEFAULT,
    DEEPY_COMPACTION_TYPE_DISCARD,
    DEEPY_COMPACTION_TYPE_KEY,
    DEEPY_COMPACTION_TYPE_SUMMARIZE,
    DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS,
    DEEPY_CONTEXT_TOKENS_MIN,
    DEEPY_CONTEXT_TOKENS_DEFAULT,
    DEEPY_CONTEXT_TOKENS_KEY,
    DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT,
    DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY,
    DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY,
    DEEPY_ENABLED_KEY,
    DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY,
    DEEPY_PRIME_GUIDANCE_DEFAULT,
    DEEPY_PRIME_MCP_SERVERS_KEY,
    DEEPY_TYPE_DEFAULT,
    DEEPY_TYPE_DISABLED,
    DEEPY_TYPE_KEY,
    DEEPY_TYPE_PRIME,
    DEEPY_TYPE_ZERO,
    DEEPY_KV_CACHE_QUANTIZATION_DEFAULT,
    DEEPY_KV_CACHE_QUANTIZATION_AUTO,
    DEEPY_KV_CACHE_QUANTIZATION_KEY,
    DEEPY_VRAM_MODE_KEY,
    DEEPY_VRAM_MODE_ALWAYS_LOADED,
    DEEPY_VRAM_MODE_UNLOAD,
    DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST,
    deepy_available,
    deepy_mode_from_config,
    format_deepy_context_tokens_label,
    deepy_requirement_message,
    normalize_deepy_context_tokens,
    normalize_deepy_file_system_access,
    normalize_deepy_file_system_paths,
    parse_deepy_file_system_paths,
    normalize_deepy_read_everywhere,
    normalize_deepy_compaction_type,
    normalize_deepy_custom_system_prompt,
    normalize_deepy_enabled,
    normalize_deepy_kv_cache_quantization,
    normalize_deepy_mcp_auto_discover_paths,
    normalize_deepy_prime_guidance,
    normalize_deepy_prime_mcp_servers,
    normalize_deepy_vram_mode,
    normalize_deepy_type,
    set_deepy_runtime_config,
    split_deepy_mode,
    validate_deepy_compaction_config,
    validate_deepy_version_config,
)
from shared.prompt_enhancer.config import (
    PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO,
    PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT,
    PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY,
    normalize_prompt_enhancer_speculative_decoding,
    validate_prompt_enhancer_speculative_decoding,
)
from shared.remote_llm.config import (
    DEEPY_ENGINE_CHOICES,
    ENGINE_CLAUDE,
    ENGINE_CODEX,
    ENGINE_OPENCODE,
    ENGINE_QWEN35_4B,
    LLM_CONFIG_KEY,
    is_remote_engine,
    local_enhancer_id,
    normalize_llm_config,
    privacy_warning,
    resolve_role_engine,
    validate_llm_config,
)
from shared.remote_llm.claude_config import bind_claude_config_ui, claude_profile_from_values, create_claude_config_ui
from shared.remote_llm.codex_config import bind_codex_config_ui, codex_profile_from_values, create_codex_config_ui
from shared.remote_llm.opencode_config import bind_opencode_config_ui, create_opencode_config_ui, opencode_profile_from_values
from postprocessing import audio_processors as audio_processor_api
from postprocessing import temporal_upsamplers as temporal_upsampler_api
from postprocessing import spatial_upsamplers as upsampler_api
from shared.utils.wgp_config_migration import (
    PROMPT_ENHANCER_CHOICES,
    enabled_choice_value,
    get_prompt_enhancer_default_mode,
    migrate_extension_defaults,
)


QWEN35_PROMPT_ENHANCER_IDS = (3, 4)
QWEN38_PROMPT_ENHANCER_ID = 5
QWEN35_QUANTIZATION_CHOICES = [("Quanto Int8 (recommended, better quality)", "quanto_int8"), ("GGUF Q4 (less VRAM/RAM & faster if kernels are installed, but worse quality)", "gguf")]
QWEN38_QUANTIZATION_CHOICES = [("GGUF Q4 (default, higher quality, requires at least 24 GB of VRAM)", "gguf"), ("GGUF Q2 (lower VRAM/RAM,  requires at least 16 GB of VRAM, lower quality)", "gguf_q2")]


def prompt_enhancer_quantization_ui_state(enhancer_enabled, quantization):
    enhancer_enabled = int(enhancer_enabled)
    if enhancer_enabled == QWEN38_PROMPT_ENHANCER_ID:
        value = quantization if quantization in ("gguf", "gguf_q2") else "gguf"
        return QWEN38_QUANTIZATION_CHOICES, value, True
    value = "gguf" if quantization in ("gguf", "gguf_q2") else "quanto_int8"
    return QWEN35_QUANTIZATION_CHOICES, value, enhancer_enabled in QWEN35_PROMPT_ENHANCER_IDS


class ConfigTabPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Configuration Tab"
        self.version = "1.1.6"
        self.description = "Lets you adjust all your performance and UI options for WAN2GP"

    def setup_ui(self):
        self.request_global("args")
        self.request_global("server_config")
        self.request_global("server_config_filename")
        self.request_global("attention_mode")
        self.request_global("compile")
        self.request_global("default_profile_video")
        self.request_global("default_profile_image")
        self.request_global("default_profile_audio")
        self.request_global("vae_config")
        self.request_global("boost")
        self.request_global("enable_int8_kernels")
        self.request_global("preload_model_policy")
        self.request_global("transformer_quantization")
        self.request_global("transformer_dtype_policy")
        self.request_global("transformer_types")
        self.request_global("text_encoder_quantization")
        self.request_global("attention_modes_installed")
        self.request_global("attention_modes_supported")
        self.request_global("displayed_model_types")
        self.request_global("memory_profile_choices")
        self.request_global("attention_modes_choices")
        self.request_global("save_path")
        self.request_global("image_save_path")
        self.request_global("audio_save_path")
        self.request_global("quit_application")
        self.request_global("release_model")
        self.request_global("release_flashvsr_vram")
        self.request_global("release_pid_vram")
        self.request_global("release_extension_offloadobjs")
        self.request_global("app")
        self.request_global("fl")
        self.request_global("is_generation_in_progress")
        self.request_global("generate_header")
        self.request_global("generate_dropdown_model_list")
        self.request_global("create_models_selector_hierarchy")
        self.request_global("get_unique_id")
        self.request_global("reset_prompt_enhancer")
        self.request_global("reset_prompt_enhancer_if_requested")
        self.request_global("release_deepy_vram")
        self.request_global("any_GPU_process_running")
        self.request_global("apply_int8_kernel_setting")

        self.request_component("model_description")
        self.request_component("header")
        self.request_component("model_family")
        self.request_component("model_base_type_choice")
        self.request_component("model_choice")
        self.request_component("refresh_form_trigger")      
        self.request_component("state")
        self.request_component("resolution")
        self.request_component("assistant_launcher_host")
        self.request_component("assistant_panel")

        self.add_tab(
            tab_id="configuration",
            label="Configuration",
            component_constructor=self.create_config_ui,
        )

    def create_config_ui(self):
        migrate_extension_defaults(self.server_config, self.server_config_filename)
        set_deepy_runtime_config(self.server_config, self.server_config_filename)
        prompt_enhancer_default_mode = get_prompt_enhancer_default_mode()
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("General"):
                    self.transformer_types_choices = HierarchySelector(
                        hierarchy=self.create_models_selector_hierarchy(self.displayed_model_types),
                        value=self.transformer_types,
                        height=0,
                        label="Selectable Generative Models (leave empty for all)",
                        display_mode="breadcrumb",
                        sort_hierarchy=False,
                        search_empty_label="No matching models",
                        interactive=not self.args.lock_config,
                    )
                    self.model_hierarchy_type_choice = gr.Dropdown(
                        choices=[
                            ("Two Levels: Model Family > Models & Finetunes", 0),
                            ("Three Levels: Model Family > Models > Finetunes", 1),
                        ],
                        value=self.server_config.get("model_hierarchy_type", 1),
                        label="Models Hierarchy In User Interface",
                        interactive=not self.args.lock_config
                    )
                    self.fit_canvas_choice = gr.Dropdown(
                        choices=[
                            ("Dimensions are Pixel Budget (preserves aspect ratio, may exceed dimensions)", 0),
                            ("Dimensions are Max Width/Height (preserves aspect ratio, fits within box)", 1),
                            ("Dimensions are Exact Output (crops input to fit exact dimensions)", 2),
                        ],
                        value=self.server_config.get("fit_canvas", 0),
                        label="Input Image/Video Sizing Behavior",
                        interactive=not self.args.lock_config
                    )

                    self.attention_choice = gr.Dropdown(
                        choices=self.attention_modes_choices,
                        value=self.attention_mode, label="Attention Type", interactive=not self.args.lock_config
                    )
                    self.preload_model_policy_choice = gr.CheckboxGroup(
                        [("Preload Model on App Launch","P"), ("Preload Model on Switch", "S"), ("Unload Model when Queue is Done", "U")],
                        value=self.preload_model_policy, label="Model Loading/Unloading Policy"
                    )
                    self.clear_file_list_choice = gr.Dropdown(
                        choices=[("None", 0), ("Keep last video", 1), ("Keep last 5 videos", 5), ("Keep last 10", 10), ("Keep last 20", 20), ("Keep last 30", 30)],
                        value=self.server_config.get("clear_file_list", 5), label="Keep Previous Generations in Gallery"
                    )
                    self.multi_prompts_gen_type_choice = gr.Dropdown(
                        choices=prompt_parser.get_multi_prompts_gen_choices("Video"),
                        value=prompt_parser.normalize_multi_prompts_mode(self.server_config.get("multi_prompts_gen_type", prompt_parser.DEFAULT_MULTI_PROMPTS_MODE), default=prompt_parser.DEFAULT_MULTI_PROMPTS_MODE),
                        label="How to Process each Line of the Text Prompt (First Time Model SDefault)",
                    )
                    self.display_stats_choice = gr.Dropdown(
                        choices=[("Disabled", 0), ("Enabled", 1)],
                        value=self.server_config.get("display_stats", 0), label="Display real-time RAM/VRAM stats (requires restart)"
                    )
                    self.max_frames_multiplier_choice = gr.Dropdown(
                        choices=[("Default", 1), ("x2", 2), ("x3", 3), ("x4", 4), ("x5", 5), ("x6", 6), ("x7", 7)],
                        value=self.server_config.get("max_frames_multiplier", 1), label="Max Frames / Duration Multiplier (requires restart)"
                    )
                    with gr.Row():
                        self.keep_resolution_on_model_switch_choice = gr.Dropdown(
                            choices=[("Yes", True), ("No", False)],
                            value=self.server_config.get("keep_resolution_on_model_switch", True),
                            label="Try to Keep Resolution when Switching Model",
                        )
                        self.enable_4k_resolutions_choice = gr.Dropdown(
                            choices=[("Off", 0), ("On", 1)],
                            value=self.server_config.get("enable_4k_resolutions", 0),
                            label="3K/4K+ Resolutions are available for all models"
                        )
                    default_paths = self.fl.default_checkpoints_paths
                    checkpoints_paths_text = "\n".join(self.server_config.get("checkpoints_paths", default_paths))
                    self.checkpoints_paths_choice = gr.Textbox(
                        label="Model Checkpoint Folders (One Path per Line. First is Default Download Path)",
                        value=checkpoints_paths_text,
                        lines=3,
                        interactive=not self.args.lock_config
                    )
                    self.loras_root_choice = gr.Textbox(
                        label="Loras Root Folder",
                        value=self.server_config.get("loras_root", "loras"),
                        interactive=not self.args.lock_config
                    )
                    self.save_queue_if_crash_choice = gr.Dropdown(
                        choices=[("Disabled", 0), ("Overwrite Last Error Queue", 1), ("Create a New Error Queue File", 2)],
                        value=self.server_config.get("save_queue_if_crash", 1),
                        label="Save Queue if Crash during Generation",
                        interactive=not self.args.lock_config
                    )
                    self.UI_theme_choice = gr.Dropdown(
                        choices=[("Blue Sky (Default)", "default"), ("Classic Gradio", "gradio")],
                        value=self.server_config.get("UI_theme", "default"), label="UI Theme (requires restart)"
                    )
                    self.queue_color_scheme_choice = gr.Dropdown(
                        choices=[
                            ("Pastel (Unique color for each item)", "pastel"),
                            ("Alternating Grey Shades", "alternating_grey"),
                        ],
                        value=self.server_config.get("queue_color_scheme", "pastel"),
                        label="Queue Color Scheme"
                    )
                    self.process_queues_when_browser_unfocused_choice = gr.Dropdown(
                        choices=[("Yes", 1), ("No", 0)],
                        value=self.server_config.get(gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY, 1),
                        label="Process Queues when Browser is not in focus (may drain more energy)",
                        interactive=not self.args.lock_config
                    )

                with gr.Tab("Performance"):
                    self.quantization_choice = gr.Dropdown(choices=[("Scaled Int8 (recommended)", "int8"), ("Scaled Fp8", "fp8"), ("16-bit (no quantization)", "bf16")], value=self.transformer_quantization, label="Transformer Model Quantization (if available otherwise get the closest available)")
                    self.transformer_dtype_policy_choice = gr.Dropdown(choices=[("Auto (Best for Hardware)", ""), ("FP16", "fp16"), ("BF16", "bf16")], value=self.transformer_dtype_policy, label="Transformer Data Type (if available)")
                    self.mixed_precision_choice = gr.Dropdown(choices=[("16-bit only (less VRAM)", "0"), ("Mixed 16/32-bit (better quality)", "1")], value=self.server_config.get("mixed_precision", "0"), label="Transformer Engine Precision")
                    self.text_encoder_quantization_choice = gr.Dropdown(choices=[("16-bit (more RAM, better quality)", "bf16"), ("8-bit (less RAM, slightly lower quality)", "int8")], value=self.text_encoder_quantization, label="Text Encoder Quantization")
                    self.lm_decoder_engine_choice = gr.Dropdown(
                        choices=[
                            ("Auto", ""),
                            ("PyTorch: slow, compatible", "legacy"),
                            ("Cuda Graph: up to x6 faster, whole LM will be loaded in VRAM", "cg"),
                            ("vllm: up to x10 faster, whole LM will be loaded in VRAM, requires Triton & Flash Attention 2", "vllm"),
                        ],
                        value=self.server_config.get("lm_decoder_engine", ""),
                        label="Language Models Decoder Engine (when available for a model)",
                    )
                    self.VAE_precision_choice = gr.Dropdown(choices=[("16-bit (faster, less VRAM)", "16"), ("32-bit (slower, better for sliding window)", "32")], value=self.server_config.get("vae_precision", "16"), label="VAE Encoding/Decoding Precision")
                    self.compile_choice = gr.Dropdown(choices=[("On (up to 20% faster, requires Triton)", "transformer"), ("Off", "")], value=self.compile, label="Compile Transformer Model (slight speed again, but first generation is slower and potential compatibility issues with some GPUs/Models)", interactive=not self.args.lock_config)
                    vae_config_value = self.vae_config if self.vae_config in [0, 1, 2, 3] else 0
                    self.vae_config_choice = gr.Dropdown(
                        choices=[
                            ("Auto", 0),
                            ("Spatial / Temporal Tiling Preset for 16GB+", 1),
                            ("Spatial / Temporal Tiling Preset for 8GB+", 2),
                            ("Spatial / Temporal Tiling Preset for 6GB+", 3),
                        ],
                        value=vae_config_value,
                        label="VAE Tiling (higher presets use less VRAM and may increase artifacts like banding)",
                    )
                    self.boost_choice = gr.Dropdown(choices=[("ON", 1), ("OFF", 2)], value=self.boost, label="Boost (~10% speedup for ~1GB VRAM)")
                    self.enable_int8_kernels_choice = gr.Dropdown(choices=[("Disabled", 0), ("Enabled if Triton available", 1)], value=self.server_config.get("enable_int8_kernels", 1), label="Int8 Kernels (Experimental, 10% faster with INT8 quantized checkpoints, requires Triton)")
                    self.video_profile_choice = gr.Dropdown(
                        choices=self.memory_profile_choices,
                        value=self.default_profile_video,
                        label="Default Memory Profile (Video)",
                    )
                    self.image_profile_choice = gr.Dropdown(
                        choices=self.memory_profile_choices,
                        value=self.default_profile_image,
                        label="Default Memory Profile (Image)",
                    )
                    self.audio_profile_choice = gr.Dropdown(
                        choices=self.memory_profile_choices,
                        value=self.default_profile_audio,
                        label="Default Memory Profile (Audio)",
                    )
                    self.preload_in_VRAM_choice = gr.Slider(0, 40000, value=self.server_config.get("preload_in_VRAM", 0), step=100, label="VRAM (MB) for Preloaded Models (0=profile default)")
                    self.max_reserved_loras_choice = gr.Slider(
                        -1,
                        10000,
                        value=self.server_config.get("max_reserved_loras", -1),
                        step=1,
                        label="Max Amount of Loras (in MB) to be Pinned To Reserved Memory (set it to 0-500MB if Out of Memory when starting Gen, -1= No limit)"
                    )
                    self.release_RAM_btn = gr.Button("Force Unload Models from RAM")

                with gr.Tab("Extensions"):
                    gr.Markdown("**Audio Postprocessors**")
                    self.audio_processor_config_bindings = audio_processor_api.create_config_ui(gr, self.server_config, lock_config=self.args.lock_config)
                    self.audio_processor_config_components = audio_processor_api.config_components(self.audio_processor_config_bindings)
                    gr.Markdown("**Temporal Postprocessors**")
                    self.temporal_upsampler_config_bindings = temporal_upsampler_api.create_config_ui(gr, self.server_config, lock_config=self.args.lock_config)
                    self.temporal_upsampler_config_components = temporal_upsampler_api.config_components(self.temporal_upsampler_config_bindings)
                    gr.Markdown("**Spatial Upsamplers / Visual Refiners**")
                    self.upsampler_config_bindings = upsampler_api.create_config_ui(gr, self.server_config, lock_config=self.args.lock_config)
                    self.upsampler_config_components = upsampler_api.config_components(self.upsampler_config_bindings)
                    gr.Markdown("**Video Preprocessors**")
                    with gr.Group():
                        self.matanyone_version_choice = gr.Dropdown(
                            choices=[("MatAnyone v1 (original, default)", "v1"), ("MatAnyone v2", "v2"), ("SAM3 (no Alpha / Grey level support but better Temporal Stability & Auto Mask Selection by Keyword)", "sam3")],
                            value=self.server_config.get("matanyone_version", "v1"),
                            label="Mask Generator Engine",
                            interactive=not self.args.lock_config
                        )

                    with gr.Group():
                        self.depth_anything_v2_variant_choice = gr.Dropdown(choices=[("Depth Anything 2 Large (more precise, slower)", "vitl"), ("Depth Anything 2 Big (less precise, faster)", "vitb"), ("Depth Anything 3 Metric Large (better temporal stability ?)", "da3_metric_large")], value=self.server_config.get("depth_anything_v2_variant", "vitl"), label="Depth Anything Preprocessor")


                with gr.Tab("Prompt Enhancer / Deepy"):
                    llm_config = normalize_llm_config(self.server_config)
                    if llm_config["deepy"] not in {value for _label, value in DEEPY_ENGINE_CHOICES}:
                        llm_config["deepy"] = ENGINE_QWEN35_4B
                    llm_config_view = {**self.server_config, LLM_CONFIG_KEY: llm_config}
                    active_llm_engines = {resolve_role_engine(llm_config_view, "deepy")}
                    deepy_remote_default = is_remote_engine(resolve_role_engine(llm_config_view, "deepy"))
                    with gr.Group(elem_classes=["wangp-transparent-group"]):
                        self.deepy_llm_engine_choice = gr.Dropdown(choices=DEEPY_ENGINE_CHOICES, value=llm_config["deepy"], label="Prompt Enhancer / Deepy LLM Engine")
                        self.remote_llm_warning_md = gr.Markdown(value=privacy_warning(self.server_config))
                        self.remote_llm_auth_md = gr.Markdown("Authentication is managed by each external engine. WanGP does not request or store passwords, API keys, access tokens, or refresh tokens.", visible=deepy_remote_default)
                        self.codex_config_ui = create_codex_config_ui(gr, llm_config["profiles"][ENGINE_CODEX], visible=ENGINE_CODEX in active_llm_engines, lock_config=self.args.lock_config)
                        self.claude_config_ui = create_claude_config_ui(gr, llm_config["profiles"][ENGINE_CLAUDE], visible=ENGINE_CLAUDE in active_llm_engines, lock_config=self.args.lock_config)
                        self.opencode_config_ui = create_opencode_config_ui(gr, llm_config["profiles"][ENGINE_OPENCODE], visible=ENGINE_OPENCODE in active_llm_engines, lock_config=self.args.lock_config)
                    with gr.Group():
                        enhancer_enabled_value = local_enhancer_id(llm_config["deepy"], enabled_choice_value(self.server_config.get("enhancer_enabled", prompt_enhancer_default_mode), PROMPT_ENHANCER_CHOICES, prompt_enhancer_default_mode))
                        self.enhancer_enabled_choice = gr.Dropdown(
                            choices=PROMPT_ENHANCER_CHOICES,
                            value=enhancer_enabled_value,
                            label="Local model used to power Prompt Enhancer / Deepy",
                            visible=False,
                        )
                        enhancer_quantization_choices, enhancer_quantization_value, enhancer_quantization_visible = prompt_enhancer_quantization_ui_state(enhancer_enabled_value, self.server_config.get("prompt_enhancer_quantization", "quanto_int8"))
                        with gr.Row():
                            self.enhancer_quantization_choice = gr.Dropdown(
                                choices=enhancer_quantization_choices,
                                value=enhancer_quantization_value,
                                label="Qwen LLM Quantization",
                                visible=enhancer_quantization_visible and not deepy_remote_default,
                            )
                            self.enhancer_speculative_decoding_choice = gr.Dropdown(
                                choices=[("Auto", PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO), ("Yes", 1), ("No", 0)],
                                value=normalize_prompt_enhancer_speculative_decoding(self.server_config.get(PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT)),
                                label="Speculative Decoding (x2 faster, but needs 100-400MB extra VRAM)",
                                interactive=not self.args.lock_config,
                                visible=not deepy_remote_default,
                            )
                        self.enhancer_mode_choice = gr.Dropdown(choices=[("On-Demand Button Only", 1),("Automatic on Generation", 0)], value=self.server_config.get("enhancer_mode", 1), label="Prompt Enhancer Usage")
                    with gr.Row():
                        self.prompt_enhancer_temperature_choice = gr.Slider(
                            0.1,
                            1.5,
                            value=self.server_config.get("prompt_enhancer_temperature", 0.6),
                            step=0.01,
                            label="Sampling Temperature (High = More Creativity)",
                            interactive=not self.args.lock_config,
                            visible=not deepy_remote_default,
                        )
                        self.prompt_enhancer_top_p_choice = gr.Slider(
                            0.1,
                            1.0,
                            value=self.server_config.get("prompt_enhancer_top_p", 0.9),
                            step=0.01,
                            label="Sampling Top-p (High = More Variety)",
                            interactive=not self.args.lock_config,
                            visible=not deepy_remote_default,
                        )
                    self.prompt_enhancer_randomize_seed_choice = gr.Checkbox(
                        value=self.server_config.get("prompt_enhancer_randomize_seed", True),
                        label="Randomize Prompt Enhancer Seed",
                        interactive=not self.args.lock_config,
                        visible=not deepy_remote_default,
                    )
                    deepy_type_default = deepy_mode_from_config(self.server_config.get(DEEPY_ENABLED_KEY, 0), self.server_config.get(DEEPY_TYPE_KEY, DEEPY_TYPE_DEFAULT))
                    self.deepy_type_choice = gr.Dropdown(
                        choices=[("Disabled", DEEPY_TYPE_DISABLED), ("Deepy Zero", DEEPY_TYPE_ZERO), ("Deepy Prime", DEEPY_TYPE_PRIME)],
                        value=deepy_type_default,
                        label="Deepy",
                        info="Deepy Zero is lightweight and local-only. Deepy Prime plans advanced workflows and is required for every external LLM because it exposes WanGP's MCP tools. With a local LLM, Prime requires Qwen3.8 VL 27B, Summarize compaction, and at least 32,000 context tokens.",
                        elem_id="deepy_type_choice",
                    )
                    self.deepy_type_value = gr.HTML(value=deepy_type_default, elem_id="deepy_type_value")
                    self.deepy_vram_mode_choice = gr.Dropdown(
                        choices=[
                            ("Unload from VRAM as soon as possible", DEEPY_VRAM_MODE_UNLOAD),
                            ("Unload from VRAM if VRAM requested by another WanGP component", DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST),
                            ("Always loaded in VRAM", DEEPY_VRAM_MODE_ALWAYS_LOADED),
                        ],
                        value=normalize_deepy_vram_mode(self.server_config.get(DEEPY_VRAM_MODE_KEY, DEEPY_VRAM_MODE_UNLOAD)),
                        label="Deepy VRAM Loading Mode (the longer Deepy stays in VRAM, the faster Deepy is)",
                        visible=not deepy_remote_default,
                    )
                    deepy_context_tokens_default = normalize_deepy_context_tokens(self.server_config.get(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT))
                    if deepy_type_default == DEEPY_TYPE_PRIME:
                        deepy_context_tokens_default = max(DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS, deepy_context_tokens_default)
                    deepy_kv_cache_quantization_default = normalize_deepy_kv_cache_quantization(self.server_config.get(DEEPY_KV_CACHE_QUANTIZATION_KEY, DEEPY_KV_CACHE_QUANTIZATION_DEFAULT))
                    deepy_compaction_type_default = normalize_deepy_compaction_type(self.server_config.get(DEEPY_COMPACTION_TYPE_KEY, DEEPY_COMPACTION_TYPE_DEFAULT))
                    if deepy_type_default == DEEPY_TYPE_PRIME:
                        deepy_compaction_type_default = DEEPY_COMPACTION_TYPE_SUMMARIZE
                    with gr.Row():
                        with gr.Column(scale=2):
                            self.deepy_context_tokens_choice = gr.Slider(
                                minimum=DEEPY_CONTEXT_TOKENS_MIN,
                                maximum=256000,
                                value=deepy_context_tokens_default,
                                step=512,
                                label=format_deepy_context_tokens_label(self.server_config.get("enhancer_enabled", 0), deepy_context_tokens_default, deepy_kv_cache_quantization_default),
                                visible=not deepy_remote_default,
                            )
                        with gr.Column(scale=1):
                            self.deepy_kv_cache_quantization_choice = gr.Dropdown(
                                choices=[("Auto", DEEPY_KV_CACHE_QUANTIZATION_AUTO), ("Disabled (BF16)", ""), ("INT8 (about half the KV-cache VRAM)", "int8")],
                                value=deepy_kv_cache_quantization_default,
                                label="KV Cache Quantization",
                                visible=not deepy_remote_default,
                            )
                    self.deepy_compaction_type_choice = gr.Dropdown(
                        choices=[("Discard Oldest Entries", DEEPY_COMPACTION_TYPE_DISCARD), ("Summarize", DEEPY_COMPACTION_TYPE_SUMMARIZE)],
                        value=deepy_compaction_type_default,
                        label="Compaction Type When Cache is Full",
                        info="Summarize starts at the lower of 85% usage and 4,096 tokens before the KV-cache limit, and requires at least 32,000 context tokens.",
                        visible=not deepy_remote_default,
                    )
                    prime_guidance_value = normalize_deepy_prime_guidance(self.server_config.get(DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY, DEEPY_PRIME_GUIDANCE_DEFAULT))
                    deepy_file_system_access = normalize_deepy_file_system_access(self.server_config.get(DEEPY_ALLOW_READ_FILE_SYSTEM_KEY, DEEPY_ALLOW_READ_FILE_SYSTEM_DEFAULT))
                    self.deepy_allow_read_file_system_choice = gr.Dropdown(
                        choices=[("Disabled", DEEPY_FILE_SYSTEM_ACCESS_DISABLED), ("Read Outputs + Selected Folders", DEEPY_FILE_SYSTEM_ACCESS_READ), ("Read / Write Outputs + Selected Folders", DEEPY_FILE_SYSTEM_ACCESS_READ_WRITE)],
                        value=deepy_file_system_access,
                        label="Deepy Filesystem Access",
                        info="Output folders are always the default scope. Add one extra folder per line below.",
                    )
                    self.deepy_file_system_paths_choice = gr.Textbox(
                        value="\n".join(normalize_deepy_file_system_paths(self.server_config.get(DEEPY_FILE_SYSTEM_PATHS_KEY, DEEPY_FILE_SYSTEM_PATHS_DEFAULT))),
                        lines=4,
                        label="Additional Filesystem Folders",
                        info='One folder per line, optionally followed by an optional alias. Quote paths containing spaces, for example: "D:\\My Media" projects.',
                        visible=deepy_file_system_access != DEEPY_FILE_SYSTEM_ACCESS_DISABLED,
                    )
                    self.deepy_read_everywhere_choice = gr.Checkbox(
                        value=normalize_deepy_read_everywhere(self.server_config.get(DEEPY_READ_EVERYWHERE_KEY, DEEPY_READ_EVERYWHERE_DEFAULT)),
                        label="Read Everywhere (Warning!)",
                        info="Allows Deepy to read any server file. Writing always remains limited to output and selected folders.",
                        visible=deepy_file_system_access != DEEPY_FILE_SYSTEM_ACCESS_DISABLED,
                    )
                    self.deepy_allow_read_file_system_choice.change(fn=lambda value: (gr.update(visible=value != DEEPY_FILE_SYSTEM_ACCESS_DISABLED), gr.update(visible=value != DEEPY_FILE_SYSTEM_ACCESS_DISABLED)), inputs=[self.deepy_allow_read_file_system_choice], outputs=[self.deepy_file_system_paths_choice, self.deepy_read_everywhere_choice], show_progress="hidden")
                    with gr.Tabs(selected="prime_guidance" if deepy_type_default == DEEPY_TYPE_PRIME else "zero_prompt"):
                        with gr.Tab("Deepy Zero Prompt", id="zero_prompt"):
                            self.deepy_zero_custom_system_prompt_choice = gr.Textbox(
                                value=normalize_deepy_custom_system_prompt(self.server_config.get(DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY, "")),
                                lines=7,
                                label="Deepy Zero Custom System Prompt",
                                info="Added after the built-in Deepy Zero system prompt on the next user interaction.",
                            )
                        with gr.Tab("Deepy Prime Guidance", id="prime_guidance"):
                            self.deepy_prime_custom_system_prompt_choice = gr.Textbox(
                                value=prime_guidance_value,
                                lines=7,
                                label="Deepy Prime User Guidance",
                                info="Treated as the user's standing preferences and appended to Deepy Prime's trusted system instructions.",
                            )
                            self.deepy_prime_mcp_servers_choice = gr.Textbox(
                                value=json.dumps(self.server_config.get(DEEPY_PRIME_MCP_SERVERS_KEY, {}), indent=2),
                                lines=8,
                                label="External MCP Servers (JSON)",
                                info='Optional servers keyed by name. Use {"transport":"stdio","command":"...","args":[]} or {"transport":"streamable-http","url":"..."}.',
                            )
                            self.deepy_mcp_auto_discover_paths_choice = gr.Checkbox(
                                value=normalize_deepy_mcp_auto_discover_paths(self.server_config.get(DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY, DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT)),
                                label="Allow Searching for Changed MCP Executable Paths",
                                info="Disabled by default. If a versioned stdio executable disappears, search only sibling version folders under the same runtime root for the newest exact filename.",
                            )
                    self.deepy_requirement_md = gr.Markdown(value=deepy_requirement_message(self.server_config))

                with gr.Tab("Outputs"):
                    self.video_container_choice = gr.Dropdown(choices=VIDEO_CONTAINER_CHOICES, value=self.server_config.get("video_container", "mp4"), label="Video Container")
                    self.video_output_codec_choice = gr.Dropdown(choices=SDR_VIDEO_CODEC_CHOICES, value=self.server_config.get("video_output_codec", "libx264_8"), label="SDR Video Codec")
                    self.hdr_video_crf_choice = gr.Dropdown(
                        choices=[
                            ("Low (x265 CRF 14)", 14),
                            ("Medium (x265 CRF 8)", 8),
                            ("High (x265 CRF 4)", 4),
                        ],
                        value=self.server_config.get("hdr_video_crf", 8),
                        label="HDR Video Codec",
                    )
                    self.image_output_codec_choice = gr.Dropdown(choices=[("JPEG Q85", 'jpeg_85'), ("WEBP Q85", 'webp_85'), ("JPEG Q95", 'jpeg_95'), ("WEBP Q95", 'webp_95'), ("WEBP Lossless", 'webp_lossless'), ("PNG Lossless", 'png')], value=self.server_config.get("image_output_codec", "jpeg_95"), label="Image Codec")
                    self.audio_output_codec_choice = gr.Dropdown(
                        choices=[
                            ("AAC 128 kbps", "aac_128"),
                            ("AAC 192 kbps", "aac_192"),
                            ("AAC 256 kbps (High Quality, Recommended)", "aac_256"),
                            ("AAC 320 kbps (Very High Quality)", "aac_320"),
                            ("ALAC Lossless (preview/playback compatibility may be limited)", "alac"),
                        ],
                        value=self.server_config.get("audio_output_codec", "aac_128"),
                        visible=True,
                        label="Audio Codec to use for MP4/MOV/MKV container",
                    )
                    audio_standalone_default = self.server_config.get("audio_stand_alone_output_codec", "wav")
                    if audio_standalone_default == "mp3":
                        audio_standalone_default = "mp3_192"
                    self.audio_stand_alone_output_codec_choice = gr.Dropdown(
                        choices=[
                            ("WAV (Lossless)", "wav"),
                            ("MP3 128 kbps", "mp3_128"),
                            ("MP3 192 kbps", "mp3_192"),
                            ("MP3 320 kbps", "mp3_320"),
                        ],
                        value=audio_standalone_default,
                        visible=True,
                        label="Audio Codec to use for standalone audio files",
                    )
                    self.metadata_choice = gr.Dropdown(
                        choices=[("Export JSON files", "json"), ("Embed metadata in file (Exif/tag)", "metadata"), ("None", "none")],
                        value=self.server_config.get("metadata_type", "metadata"), label="Metadata Handling"
                    )
                    self.keep_intermediate_sliding_windows_choice = gr.Dropdown(
                        choices=[("Yes", 1), ("No", 0)],
                        value=self.server_config.get("keep_intermediate_sliding_windows", 1),
                        label="Keep Intermediate Sliding Windows"
                    )
                    self.embed_source_images_choice = gr.Checkbox(
                        value=self.server_config.get("embed_source_images", False),
                        label="Embed Source Images",
                        info="Saves i2v source images inside MP4/MOV/MKV files"
                    )
                    self.video_save_path_choice = gr.Textbox(label="Video Output Folder (requires restart)", value=self.save_path)
                    self.image_save_path_choice = gr.Textbox(label="Image Output Folder (requires restart)", value=self.image_save_path)
                    self.audio_save_path_choice = gr.Textbox(label="Audio Output Folder (requires restart)", value=self.audio_save_path)

                with gr.Tab("Notifications"):
                    self.notification_sound_enabled_choice = gr.Dropdown(choices=[("On", 1), ("Off", 0)], value=self.server_config.get("notification_sound_enabled", 0), label="Notification Sound")
                    self.notification_sound_volume_choice = gr.Slider(0, 100, value=self.server_config.get("notification_sound_volume", 50), step=5, label="Notification Volume")
                    self.notification_config_ui = notifications.create_config_ui(gr, self.server_config)
                    self.notification_apprise_urls_choice, self.notification_secure_storage_choice, self.notification_on_generation_choice, self.notification_on_queue_complete_choice, self.notification_on_queue_interrupted_choice = self.notification_config_ui.save_components

            self.msg = gr.Markdown()
            with gr.Row():
                self.apply_btn = gr.Button("Save Settings")

        def update_deepy_requirement(enhancer_enabled_choice, deepy_type_choice, deepy_context_tokens_choice, deepy_compaction_type_choice, deepy_llm_engine_choice):
            runtime_config = dict(self.server_config)
            deepy_enabled_choice, deepy_type_choice = split_deepy_mode(deepy_type_choice)
            runtime_config["enhancer_enabled"] = enhancer_enabled_choice
            runtime_config[DEEPY_ENABLED_KEY] = deepy_enabled_choice
            runtime_config[DEEPY_TYPE_KEY] = deepy_type_choice
            runtime_config[DEEPY_CONTEXT_TOKENS_KEY] = deepy_context_tokens_choice
            runtime_config[DEEPY_COMPACTION_TYPE_KEY] = deepy_compaction_type_choice
            runtime_config[LLM_CONFIG_KEY] = {**normalize_llm_config(runtime_config), "deepy": deepy_llm_engine_choice}
            return deepy_requirement_message(runtime_config)

        deepy_requirement_inputs = [self.enhancer_enabled_choice, self.deepy_type_choice, self.deepy_context_tokens_choice, self.deepy_compaction_type_choice, self.deepy_llm_engine_choice]
        for requirement_input in (self.enhancer_enabled_choice, self.deepy_context_tokens_choice, self.deepy_compaction_type_choice, self.deepy_llm_engine_choice):
            requirement_input.input(fn=update_deepy_requirement, inputs=deepy_requirement_inputs, outputs=[self.deepy_requirement_md], show_progress="hidden")

        def enforce_deepy_prime_requirements(deepy_type_choice, deepy_context_tokens_choice, deepy_compaction_type_choice, enhancer_enabled_choice, deepy_kv_cache_quantization_choice, deepy_llm_engine_choice):
            runtime_config = dict(self.server_config)
            deepy_enabled_choice, deepy_type_choice = split_deepy_mode(deepy_type_choice)
            runtime_config["enhancer_enabled"] = enhancer_enabled_choice
            runtime_config[DEEPY_ENABLED_KEY] = deepy_enabled_choice
            runtime_config[DEEPY_TYPE_KEY] = deepy_type_choice
            if normalize_deepy_type(deepy_type_choice) == DEEPY_TYPE_PRIME:
                deepy_context_tokens_choice = max(DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS, normalize_deepy_context_tokens(deepy_context_tokens_choice))
                deepy_compaction_type_choice = DEEPY_COMPACTION_TYPE_SUMMARIZE
            runtime_config[DEEPY_CONTEXT_TOKENS_KEY] = deepy_context_tokens_choice
            runtime_config[DEEPY_COMPACTION_TYPE_KEY] = deepy_compaction_type_choice
            runtime_config[LLM_CONFIG_KEY] = {**normalize_llm_config(runtime_config), "deepy": deepy_llm_engine_choice}
            context_label = format_deepy_context_tokens_label(enhancer_enabled_choice, deepy_context_tokens_choice, deepy_kv_cache_quantization_choice)
            return gr.update(value=deepy_context_tokens_choice, label=context_label), gr.update(value=deepy_compaction_type_choice), deepy_requirement_message(runtime_config), deepy_mode_from_config(deepy_enabled_choice, deepy_type_choice)

        self.deepy_type_choice.input(fn=enforce_deepy_prime_requirements, inputs=[self.deepy_type_choice, self.deepy_context_tokens_choice, self.deepy_compaction_type_choice, self.enhancer_enabled_choice, self.deepy_kv_cache_quantization_choice, self.deepy_llm_engine_choice], outputs=[self.deepy_context_tokens_choice, self.deepy_compaction_type_choice, self.deepy_requirement_md, self.deepy_type_value], show_progress="hidden")

        def update_deepy_context_label(enhancer_enabled_choice, deepy_context_tokens_choice, deepy_kv_cache_quantization_choice):
            return gr.update(label=format_deepy_context_tokens_label(enhancer_enabled_choice, deepy_context_tokens_choice, deepy_kv_cache_quantization_choice))

        deepy_context_label_inputs = [self.enhancer_enabled_choice, self.deepy_context_tokens_choice, self.deepy_kv_cache_quantization_choice]
        self.enhancer_enabled_choice.input(fn=update_deepy_context_label, inputs=deepy_context_label_inputs, outputs=[self.deepy_context_tokens_choice], show_progress="hidden")
        self.deepy_context_tokens_choice.input(fn=update_deepy_context_label, inputs=deepy_context_label_inputs, outputs=[self.deepy_context_tokens_choice], show_progress="hidden")
        self.deepy_kv_cache_quantization_choice.input(fn=update_deepy_context_label, inputs=deepy_context_label_inputs, outputs=[self.deepy_context_tokens_choice], show_progress="hidden")

        def update_speculative_decoding_choice(enhancer_enabled_choice, speculative_decoding_choice):
            return gr.update(value=normalize_prompt_enhancer_speculative_decoding(speculative_decoding_choice), interactive=not self.args.lock_config)

        def update_enhancer_quantization_choice(enhancer_enabled_choice, enhancer_quantization_choice):
            choices, value, visible = prompt_enhancer_quantization_ui_state(enhancer_enabled_choice, enhancer_quantization_choice)
            return gr.update(choices=choices, value=value, visible=visible)

        self.enhancer_enabled_choice.input(fn=update_enhancer_quantization_choice, inputs=[self.enhancer_enabled_choice, self.enhancer_quantization_choice], outputs=[self.enhancer_quantization_choice], show_progress="hidden")
        self.enhancer_enabled_choice.input(fn=update_speculative_decoding_choice, inputs=[self.enhancer_enabled_choice, self.enhancer_speculative_decoding_choice], outputs=[self.enhancer_speculative_decoding_choice], show_progress="hidden")

        def update_remote_engine_ui(deepy_engine, enhancer_quantization):
            view = normalize_llm_config(self.server_config)
            view["deepy"] = deepy_engine
            runtime_config = {**self.server_config, LLM_CONFIG_KEY: view}
            deepy_remote = is_remote_engine(resolve_role_engine(runtime_config, "deepy"))
            active = {resolve_role_engine(runtime_config, "deepy")}
            quantization_choices, quantization_value, quantization_visible = prompt_enhancer_quantization_ui_state(local_enhancer_id(deepy_engine), enhancer_quantization)
            return (
                privacy_warning(runtime_config),
                gr.update(visible=deepy_remote),
                gr.update(visible=ENGINE_CODEX in active), gr.update(visible=ENGINE_CLAUDE in active), gr.update(visible=ENGINE_OPENCODE in active),
                gr.update(value=local_enhancer_id(deepy_engine), visible=False), gr.update(choices=quantization_choices, value=quantization_value, visible=quantization_visible and not deepy_remote), gr.update(visible=not deepy_remote),
                gr.update(visible=not deepy_remote), gr.update(visible=not deepy_remote), gr.update(visible=not deepy_remote),
                gr.update(visible=not deepy_remote), gr.update(visible=not deepy_remote), gr.update(visible=not deepy_remote), gr.update(visible=not deepy_remote),
            )

        remote_engine_state_inputs = [self.deepy_llm_engine_choice, self.enhancer_quantization_choice]
        remote_engine_outputs = [
            self.remote_llm_warning_md, self.remote_llm_auth_md, self.codex_config_ui.group, self.claude_config_ui.group, self.opencode_config_ui.group,
            self.enhancer_enabled_choice, self.enhancer_quantization_choice, self.enhancer_speculative_decoding_choice,
            self.prompt_enhancer_temperature_choice, self.prompt_enhancer_top_p_choice, self.prompt_enhancer_randomize_seed_choice,
            self.deepy_vram_mode_choice, self.deepy_context_tokens_choice, self.deepy_kv_cache_quantization_choice, self.deepy_compaction_type_choice,
        ]
        self.deepy_llm_engine_choice.input(fn=update_remote_engine_ui, inputs=remote_engine_state_inputs, outputs=remote_engine_outputs, show_progress="hidden")

        bind_codex_config_ui(gr, self.codex_config_ui, self.server_config, self.server_config_filename)
        bind_claude_config_ui(gr, self.claude_config_ui, self.server_config, self.server_config_filename)
        bind_opencode_config_ui(gr, self.opencode_config_ui, self.server_config, self.server_config_filename)
        self.process_queues_when_browser_unfocused_choice.change(
            fn=None,
            inputs=[self.process_queues_when_browser_unfocused_choice],
            js="""
                (enabled) => {
                    if (window.__gradioFocusQueuePatch) {
                        window.__gradioFocusQueuePatch.enableBackgroundScheduler = Number(enabled) !== 0;
                    }
                }
            """,
            queue=False,
            show_progress="hidden",
        )

        inputs = [
            self.state,
            self.transformer_types_choices, self.model_hierarchy_type_choice, self.fit_canvas_choice,
            self.attention_choice, self.preload_model_policy_choice, self.clear_file_list_choice, self.multi_prompts_gen_type_choice, self.keep_intermediate_sliding_windows_choice,
            self.display_stats_choice, self.max_frames_multiplier_choice, self.keep_resolution_on_model_switch_choice, self.enable_4k_resolutions_choice, self.checkpoints_paths_choice, self.loras_root_choice, self.save_queue_if_crash_choice,
            self.UI_theme_choice, self.queue_color_scheme_choice, self.process_queues_when_browser_unfocused_choice,
            self.quantization_choice, self.transformer_dtype_policy_choice, self.mixed_precision_choice,
            self.text_encoder_quantization_choice, self.lm_decoder_engine_choice, self.VAE_precision_choice, self.compile_choice,
            self.depth_anything_v2_variant_choice,
            self.vae_config_choice, self.boost_choice, self.enable_int8_kernels_choice,
            self.video_profile_choice, self.image_profile_choice, self.audio_profile_choice,
            self.preload_in_VRAM_choice, self.max_reserved_loras_choice,
            self.deepy_llm_engine_choice,
            *self.codex_config_ui.save_components,
            *self.claude_config_ui.save_components,
            *self.opencode_config_ui.save_components,
            self.enhancer_enabled_choice, self.enhancer_quantization_choice, self.enhancer_speculative_decoding_choice, self.enhancer_mode_choice,
            self.prompt_enhancer_temperature_choice, self.prompt_enhancer_top_p_choice, self.prompt_enhancer_randomize_seed_choice,
            self.matanyone_version_choice,
            self.deepy_type_choice, self.deepy_vram_mode_choice, self.deepy_allow_read_file_system_choice, self.deepy_file_system_paths_choice, self.deepy_read_everywhere_choice,
            self.deepy_context_tokens_choice, self.deepy_kv_cache_quantization_choice, self.deepy_compaction_type_choice, self.deepy_zero_custom_system_prompt_choice, self.deepy_prime_custom_system_prompt_choice, self.deepy_prime_mcp_servers_choice, self.deepy_mcp_auto_discover_paths_choice,
            self.video_container_choice, self.video_output_codec_choice, self.hdr_video_crf_choice, self.image_output_codec_choice, self.audio_output_codec_choice, self.audio_stand_alone_output_codec_choice,
            self.metadata_choice, self.embed_source_images_choice,
            self.video_save_path_choice, self.image_save_path_choice, self.audio_save_path_choice,
            self.notification_sound_enabled_choice, self.notification_sound_volume_choice, self.notification_apprise_urls_choice, self.notification_secure_storage_choice, self.notification_on_generation_choice, self.notification_on_queue_complete_choice, self.notification_on_queue_interrupted_choice,
            *self.audio_processor_config_components,
            *self.temporal_upsampler_config_components,
            *self.upsampler_config_components,
            self.resolution
        ]

        self.apply_btn.click(
            fn=self._save_changes,
            inputs=inputs,
            outputs=[
                self.msg,
                self.model_description,
                self.header,
                self.model_family,
                self.model_base_type_choice,
                self.model_choice,
                self.refresh_form_trigger,
                self.assistant_launcher_host,
                self.assistant_panel
            ]
        )

        def release_ram_and_notify(state):
            unload_models_from_ram(
                state,
                server_config=self.server_config,
                any_GPU_process_running=self.any_GPU_process_running,
                release_deepy_vram=self.release_deepy_vram,
                reset_prompt_enhancer=self.reset_prompt_enhancer,
                reset_prompt_enhancer_if_requested=self.reset_prompt_enhancer_if_requested,
                release_extensions=self.release_extension_offloadobjs,
                release_model=self.release_model,
            )

        self.release_RAM_btn.click(fn=release_ram_and_notify, inputs=[self.state])
        return [self.release_RAM_btn]

    def _save_changes(self, state, *args):
        gen_in_progress = self.is_generation_in_progress()
        # return "<div style='color:red; text-align:center;'>Unable to change config when a generation is in progress.</div>", *[gr.update()]*5

        if self.args.lock_config:
            return "<div style='color:red; text-align:center;'>Configuration is locked by command-line arguments.</div>", *[gr.update()]*8

        old_server_config = copy.deepcopy(self.server_config)
        audio_processor_component_count = len(getattr(self, "audio_processor_config_components", []))
        temporal_upsampler_component_count = len(getattr(self, "temporal_upsampler_config_components", []))
        upsampler_component_count = len(getattr(self, "upsampler_config_components", []))
        extension_component_count = audio_processor_component_count + temporal_upsampler_component_count + upsampler_component_count
        if extension_component_count:
            extension_config_values = args[-extension_component_count - 1:-1]
            audio_processor_config_values = extension_config_values[:audio_processor_component_count]
            temporal_upsampler_config_values = extension_config_values[audio_processor_component_count:audio_processor_component_count + temporal_upsampler_component_count]
            upsampler_config_values = extension_config_values[audio_processor_component_count + temporal_upsampler_component_count:]
            fixed_args = args[:-extension_component_count - 1] + (args[-1],)
        else:
            audio_processor_config_values = []
            temporal_upsampler_config_values = []
            upsampler_config_values = []
            fixed_args = args

        (
            transformer_types_choices, model_hierarchy_type_choice, fit_canvas_choice,
            attention_choice, preload_model_policy_choice, clear_file_list_choice, multi_prompts_gen_type_choice, keep_intermediate_sliding_windows_choice,
            display_stats_choice, max_frames_multiplier_choice, keep_resolution_on_model_switch_choice, enable_4k_resolutions_choice, checkpoints_paths_choice, loras_root_choice, save_queue_if_crash_choice,
            UI_theme_choice, queue_color_scheme_choice, process_queues_when_browser_unfocused_choice,
            quantization_choice, transformer_dtype_policy_choice, mixed_precision_choice,
            text_encoder_quantization_choice, lm_decoder_engine_choice, VAE_precision_choice, compile_choice,
            depth_anything_v2_variant_choice,
            vae_config_choice, boost_choice, enable_int8_kernels_choice,
            video_profile_choice, image_profile_choice, audio_profile_choice,
            preload_in_VRAM_choice, max_reserved_loras_choice,
            deepy_llm_engine_choice,
            codex_executable_choice, codex_model_choice, codex_reasoning_effort_choice,
            claude_executable_choice, claude_model_choice, claude_reasoning_effort_choice,
            opencode_executable_choice, opencode_base_url_choice, opencode_provider_choice, opencode_model_choice, opencode_reasoning_effort_choice, opencode_config_choice,
            enhancer_enabled_choice, enhancer_quantization_choice, enhancer_speculative_decoding_choice, enhancer_mode_choice,
            prompt_enhancer_temperature_choice, prompt_enhancer_top_p_choice, prompt_enhancer_randomize_seed_choice,
            matanyone_version_choice,
            deepy_type_choice, deepy_vram_mode_choice, deepy_allow_read_file_system_choice, deepy_file_system_paths_choice, deepy_read_everywhere_choice,
            deepy_context_tokens_choice, deepy_kv_cache_quantization_choice, deepy_compaction_type_choice, deepy_zero_custom_system_prompt_choice, deepy_prime_custom_system_prompt_choice, deepy_prime_mcp_servers_choice, deepy_mcp_auto_discover_paths_choice,
            video_container_choice, video_output_codec_choice, hdr_video_crf_choice, image_output_codec_choice, audio_output_codec_choice, audio_stand_alone_output_codec_choice,
            metadata_choice, embed_source_images_choice,
            save_path_choice, image_save_path_choice, audio_save_path_choice,
            notification_sound_enabled_choice, notification_sound_volume_choice, notification_apprise_urls_choice, notification_secure_storage_choice, notification_on_generation_choice, notification_on_queue_complete_choice, notification_on_queue_interrupted_choice,
            last_resolution_choice
        ) = fixed_args

        llm_config_choice = {
            "deepy": deepy_llm_engine_choice,
            "profiles": {
                ENGINE_CODEX: codex_profile_from_values(codex_executable_choice, codex_model_choice, codex_reasoning_effort_choice, normalize_llm_config(old_server_config)["profiles"][ENGINE_CODEX]["model_catalog"]),
                ENGINE_CLAUDE: claude_profile_from_values(claude_executable_choice, claude_model_choice, claude_reasoning_effort_choice, normalize_llm_config(old_server_config)["profiles"][ENGINE_CLAUDE]["model_catalog"]),
                ENGINE_OPENCODE: opencode_profile_from_values(opencode_executable_choice, opencode_base_url_choice, opencode_provider_choice, opencode_model_choice, opencode_reasoning_effort_choice, opencode_config_choice, normalize_llm_config(old_server_config)["profiles"][ENGINE_OPENCODE]["model_catalog"]),
            },
        }

        deepy_enabled_choice, deepy_type_choice = split_deepy_mode(deepy_type_choice)
        validation_config = {**old_server_config, LLM_CONFIG_KEY: llm_config_choice}
        try:
            llm_config_choice = validate_llm_config(validation_config, deepy_enabled=deepy_enabled_choice, deepy_type=deepy_type_choice)
        except ValueError as exc:
            gr.Info(f"Configuration was not saved: {exc}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {exc}</div>", *[gr.update()]*8

        deepy_remote = is_remote_engine(deepy_llm_engine_choice)
        if deepy_remote:
            # Local-only controls are hidden for external engines. Preserve their
            # saved values so a stale hidden component cannot block or alter a
            # remote-engine configuration.
            enhancer_enabled_choice = old_server_config.get("enhancer_enabled", enhancer_enabled_choice)
            enhancer_quantization_choice = old_server_config.get("prompt_enhancer_quantization", enhancer_quantization_choice)
            enhancer_speculative_decoding_choice = old_server_config.get(PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, enhancer_speculative_decoding_choice)
        else:
            enhancer_enabled_choice = local_enhancer_id(deepy_llm_engine_choice, enhancer_enabled_choice)

        if not deepy_remote and int(enhancer_enabled_choice) == QWEN38_PROMPT_ENHANCER_ID and enhancer_quantization_choice not in ("gguf", "gguf_q2"):
            error = "Qwen3.8-27B is available only as GGUF. Select GGUF Q2 or Q4 as the Qwen LLM quantization."
            gr.Info(f"Configuration was not saved: {error}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {error}</div>", *[gr.update()]*8
        if not deepy_remote and int(enhancer_enabled_choice) in QWEN35_PROMPT_ENHANCER_IDS and enhancer_quantization_choice not in ("quanto_int8", "gguf"):
            error = "Qwen3.5 is available as Quanto Int8 or GGUF Q4."
            gr.Info(f"Configuration was not saved: {error}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {error}</div>", *[gr.update()]*8

        if not deepy_remote:
            try:
                enhancer_speculative_decoding_choice = validate_prompt_enhancer_speculative_decoding(enhancer_enabled_choice, enhancer_speculative_decoding_choice)
            except ValueError as exc:
                gr.Info(f"Configuration was not saved: {exc}")
                return f"<div style='color:red; text-align:center;'>Configuration was not saved: {exc}</div>", *[gr.update()]*8

        try:
            if not deepy_remote:
                deepy_compaction_type_choice = validate_deepy_compaction_config(deepy_compaction_type_choice, deepy_context_tokens_choice)
                deepy_type_choice, deepy_compaction_type_choice, deepy_context_tokens_choice = validate_deepy_version_config(deepy_type_choice, deepy_compaction_type_choice, deepy_context_tokens_choice, enhancer_enabled_choice)
            deepy_prime_mcp_servers_choice = normalize_deepy_prime_mcp_servers(deepy_prime_mcp_servers_choice)
            deepy_file_system_paths_choice = normalize_deepy_file_system_paths(deepy_file_system_paths_choice)
            parse_deepy_file_system_paths(deepy_file_system_paths_choice)
        except ValueError as exc:
            gr.Info(f"Configuration was not saved: {exc}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {exc}</div>", *[gr.update()]*8

        if len(checkpoints_paths_choice.strip()) == 0:
            checkpoints_paths = self.fl.default_checkpoints_paths
        else:
            checkpoints_paths = [path.strip() for path in checkpoints_paths_choice.replace("\r", "").split("\n") if len(path.strip()) > 0]

        video_output_error = validate_video_output_settings(video_output_codec_choice, video_container_choice, audio_output_codec_choice)
        if video_output_error is not None:
            gr.Info(f"Configuration was not saved: {video_output_error}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {video_output_error}</div>", *[gr.update()]*8

        self.fl.set_checkpoints_paths(checkpoints_paths)

        audio_processor_config_update = audio_processor_api.collect_config_update(self.audio_processor_config_bindings, audio_processor_config_values)
        for message in audio_processor_api.validate_config_update_messages(self.audio_processor_config_bindings, audio_processor_config_update):
            gr.Info(message)
        temporal_upsampler_config_update = temporal_upsampler_api.collect_config_update(self.temporal_upsampler_config_bindings, temporal_upsampler_config_values)
        for message in temporal_upsampler_api.validate_config_update_messages(self.temporal_upsampler_config_bindings, temporal_upsampler_config_update):
            gr.Info(message)
        upsampler_config_update = upsampler_api.collect_config_update(self.upsampler_config_bindings, upsampler_config_values)
        for message in upsampler_api.validate_config_update_messages(self.upsampler_config_bindings, upsampler_config_update):
            gr.Info(message)

        try:
            notification_config_update = notifications.prepare_config_update(old_server_config, notification_apprise_urls_choice, notification_secure_storage_choice, notification_on_generation_choice, notification_on_queue_complete_choice, notification_on_queue_interrupted_choice)
        except notifications.SecureStorageError as exc:
            gr.Info(f"Configuration was not saved: {exc}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {exc}</div>", *[gr.update()]*8

        new_server_config = copy.deepcopy(old_server_config)
        new_server_config.update({
            "attention_mode": attention_choice, "transformer_types": transformer_types_choices,
            "text_encoder_quantization": text_encoder_quantization_choice, "save_path": save_path_choice,
            "image_save_path": image_save_path_choice, "audio_save_path": audio_save_path_choice,
            "lm_decoder_engine": lm_decoder_engine_choice,
            "compile": compile_choice, "profile": video_profile_choice,
            "video_profile": video_profile_choice, "image_profile": image_profile_choice, "audio_profile": audio_profile_choice,
            "vae_config": vae_config_choice, "vae_precision": VAE_precision_choice,
            "mixed_precision": mixed_precision_choice, "metadata_type": metadata_choice,
            "transformer_quantization": quantization_choice, "transformer_dtype_policy": transformer_dtype_policy_choice,
            "boost": boost_choice, "enable_int8_kernels": enable_int8_kernels_choice, "clear_file_list": clear_file_list_choice,
            "multi_prompts_gen_type": prompt_parser.normalize_multi_prompts_mode(multi_prompts_gen_type_choice, default=prompt_parser.DEFAULT_MULTI_PROMPTS_MODE),
            "keep_intermediate_sliding_windows": keep_intermediate_sliding_windows_choice,
            "preload_model_policy": preload_model_policy_choice, "UI_theme": UI_theme_choice,
            "fit_canvas": fit_canvas_choice, "enhancer_enabled": enhancer_enabled_choice,
            LLM_CONFIG_KEY: llm_config_choice,
            "prompt_enhancer_quantization": enhancer_quantization_choice,
            PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY: enhancer_speculative_decoding_choice,
            "enhancer_mode": enhancer_mode_choice,
            "matanyone_version": matanyone_version_choice,
            "prompt_enhancer_temperature": prompt_enhancer_temperature_choice,
            "prompt_enhancer_top_p": prompt_enhancer_top_p_choice,
            "prompt_enhancer_randomize_seed": prompt_enhancer_randomize_seed_choice,
            DEEPY_ENABLED_KEY: normalize_deepy_enabled(deepy_enabled_choice),
            DEEPY_TYPE_KEY: normalize_deepy_type(deepy_type_choice),
            DEEPY_VRAM_MODE_KEY: normalize_deepy_vram_mode(deepy_vram_mode_choice),
            DEEPY_ALLOW_READ_FILE_SYSTEM_KEY: normalize_deepy_file_system_access(deepy_allow_read_file_system_choice),
            DEEPY_FILE_SYSTEM_PATHS_KEY: deepy_file_system_paths_choice,
            DEEPY_READ_EVERYWHERE_KEY: normalize_deepy_read_everywhere(deepy_read_everywhere_choice),
            DEEPY_CONTEXT_TOKENS_KEY: normalize_deepy_context_tokens(deepy_context_tokens_choice),
            DEEPY_KV_CACHE_QUANTIZATION_KEY: normalize_deepy_kv_cache_quantization(deepy_kv_cache_quantization_choice),
            DEEPY_COMPACTION_TYPE_KEY: deepy_compaction_type_choice,
            DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY: normalize_deepy_custom_system_prompt(deepy_zero_custom_system_prompt_choice),
            DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY: normalize_deepy_prime_guidance(deepy_prime_custom_system_prompt_choice),
            DEEPY_PRIME_MCP_SERVERS_KEY: deepy_prime_mcp_servers_choice,
            DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY: normalize_deepy_mcp_auto_discover_paths(deepy_mcp_auto_discover_paths_choice),
            "preload_in_VRAM": preload_in_VRAM_choice, "depth_anything_v2_variant": depth_anything_v2_variant_choice,
            "notification_sound_enabled": notification_sound_enabled_choice,
            "notification_sound_volume": notification_sound_volume_choice,
            **notification_config_update,
            "max_frames_multiplier": max_frames_multiplier_choice, "display_stats": display_stats_choice,
            "keep_resolution_on_model_switch": keep_resolution_on_model_switch_choice,
            "enable_4k_resolutions": enable_4k_resolutions_choice,
            "max_reserved_loras": max_reserved_loras_choice,
            "video_output_codec": video_output_codec_choice, "hdr_video_crf": hdr_video_crf_choice,
            "image_output_codec": image_output_codec_choice,
            "audio_output_codec": audio_output_codec_choice,
            "audio_stand_alone_output_codec": audio_stand_alone_output_codec_choice,
            "model_hierarchy_type": model_hierarchy_type_choice,
            "checkpoints_paths": checkpoints_paths,
            "loras_root": loras_root_choice,
            "save_queue_if_crash": save_queue_if_crash_choice,
            "queue_color_scheme": queue_color_scheme_choice,
            gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY: process_queues_when_browser_unfocused_choice,
            "embed_source_images": embed_source_images_choice,
            "video_container": video_container_choice,
            "last_model_type": state["model_type"],
            "last_model_per_family": state["last_model_per_family"],
            "last_model_per_type": state["last_model_per_type"],
            "last_advanced_choice": state["advanced"], "last_resolution_choice": last_resolution_choice,
            "last_resolution_per_group": state["last_resolution_per_group"],
        })
        audio_processor_api.apply_config_update(new_server_config, self.audio_processor_config_bindings, audio_processor_config_update)
        temporal_upsampler_api.apply_config_update(new_server_config, self.temporal_upsampler_config_bindings, temporal_upsampler_config_update)
        upsampler_api.apply_config_update(new_server_config, self.upsampler_config_bindings, upsampler_config_update)

        if self.args.lock_config:
            if "attention_mode" in old_server_config: new_server_config["attention_mode"] = old_server_config["attention_mode"]
            if "compile" in old_server_config: new_server_config["compile"] = old_server_config["compile"]

        for key in ("depth_anything_v3_process_res", "depth_anything_v3_chunk_size", "depth_anything_v3_chunk_overlap"):
            new_server_config.pop(key, None)

        with open(self.server_config_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(new_server_config, indent=4))
        try:
            notifications.cleanup_config_update(old_server_config, new_server_config)
        except notifications.SecureStorageError as exc:
            gr.Warning(f"Configuration was saved, but the previous credential-store entry could not be removed: {exc}")
        
        changes = [k for k, v in new_server_config.items() if v != old_server_config.get(k)]

        no_reload_keys = [
            "attention_mode", "vae_config", "boost", "enable_int8_kernels", "save_path", "image_save_path", "audio_save_path",
            "metadata_type", "clear_file_list", "multi_prompts_gen_type", "keep_intermediate_sliding_windows", "fit_canvas", "depth_anything_v2_variant",
            "notification_sound_enabled", "notification_sound_volume", *notifications.CONFIG_KEYS, "audio_processors", "temporal_upsamplers", "spatial_upsamplers", "matanyone_version",
            "prompt_enhancer_temperature", "prompt_enhancer_top_p", "prompt_enhancer_randomize_seed", "prompt_enhancer_quantization", PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, "enhancer_mode",
            DEEPY_ENABLED_KEY, DEEPY_TYPE_KEY, DEEPY_VRAM_MODE_KEY, DEEPY_ALLOW_READ_FILE_SYSTEM_KEY, DEEPY_FILE_SYSTEM_PATHS_KEY, DEEPY_READ_EVERYWHERE_KEY, DEEPY_CONTEXT_TOKENS_KEY, DEEPY_KV_CACHE_QUANTIZATION_KEY, DEEPY_COMPACTION_TYPE_KEY, DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY, DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY, DEEPY_PRIME_MCP_SERVERS_KEY, DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY,
            LLM_CONFIG_KEY,
            "max_frames_multiplier", "display_stats", "keep_resolution_on_model_switch", "enable_4k_resolutions", "max_reserved_loras", "video_output_codec", "hdr_video_crf", "video_container",
            "embed_source_images", "image_output_codec", "audio_output_codec", "audio_stand_alone_output_codec", "checkpoints_paths", "loras_root", "save_queue_if_crash",
            "model_hierarchy_type", "UI_theme", "queue_color_scheme", gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY
        ]

        needs_reload = not all(change in no_reload_keys for change in changes)

        self.set_global("server_config", new_server_config)
        self.set_global("three_levels_hierarchy", new_server_config["model_hierarchy_type"] == 1)
        self.set_global("attention_mode", new_server_config["attention_mode"])
        self.set_global("default_profile", new_server_config["profile"])
        self.set_global("default_profile_video", new_server_config["video_profile"])
        self.set_global("default_profile_image", new_server_config["image_profile"])
        self.set_global("default_profile_audio", new_server_config["audio_profile"])
        self.set_global("compile", new_server_config["compile"])
        self.set_global("text_encoder_quantization", new_server_config["text_encoder_quantization"])
        self.set_global("lm_decoder_engine", new_server_config["lm_decoder_engine"])
        self.set_global("vae_config", new_server_config["vae_config"])
        self.set_global("boost", new_server_config["boost"])
        self.set_global("enable_int8_kernels", new_server_config["enable_int8_kernels"])
        self.set_global("save_path", new_server_config["save_path"])
        self.set_global("image_save_path", new_server_config["image_save_path"])
        self.set_global("audio_save_path", new_server_config["audio_save_path"])
        self.set_global("preload_model_policy", new_server_config["preload_model_policy"])
        self.set_global("transformer_quantization", new_server_config["transformer_quantization"])
        self.set_global("transformer_dtype_policy", new_server_config["transformer_dtype_policy"])
        self.set_global("transformer_types", new_server_config["transformer_types"])
        set_deepy_runtime_config(new_server_config, self.server_config_filename)
        if needs_reload: self.set_global("reload_needed", True)
        self.server_config.update(new_server_config)

        enhancer_runtime_changed = LLM_CONFIG_KEY in changes or "enhancer_enabled" in changes or "prompt_enhancer_quantization" in changes or "lm_decoder_engine" in changes or DEEPY_KV_CACHE_QUANTIZATION_KEY in changes or DEEPY_ENABLED_KEY in changes or DEEPY_VRAM_MODE_KEY in changes
        deepy_type_changed = DEEPY_TYPE_KEY in changes
        deepy_prime_mcp_servers_changed = DEEPY_PRIME_MCP_SERVERS_KEY in changes or DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY in changes
        deepy_file_access_changed = any(key in changes for key in (DEEPY_ALLOW_READ_FILE_SYSTEM_KEY, DEEPY_FILE_SYSTEM_PATHS_KEY, DEEPY_READ_EVERYWHERE_KEY, "save_path", "image_save_path", "audio_save_path"))
        speculative_decoding_changed = PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY in changes
        enhancer_profile_changed = "profile" in changes or "video_profile" in changes
        if enhancer_runtime_changed:
            get_or_create_assistant_session(state).force_loading_status_once = True
            self.release_deepy_vram(state, clear_session_state=True, discard_runtime_snapshot=True)
            self.reset_prompt_enhancer()
            self.reset_prompt_enhancer_if_requested()
        elif speculative_decoding_changed:
            get_or_create_assistant_session(state).force_loading_status_once = True
            self.release_deepy_vram(state, clear_session_state=False, discard_runtime_snapshot=True)
            self.reset_prompt_enhancer()
            self.reset_prompt_enhancer_if_requested()
        elif enhancer_profile_changed:
            self.release_deepy_vram(state, clear_session_state=False, discard_runtime_snapshot=True)
            self.reset_prompt_enhancer()
            self.reset_prompt_enhancer_if_requested()
        elif deepy_type_changed or deepy_prime_mcp_servers_changed or deepy_file_access_changed:
            self.release_deepy_vram(state, clear_session_state=True, discard_runtime_snapshot=True)
        audio_processor_api.release_changed_config_processors(old_server_config, new_server_config, changes)
        temporal_upsampler_api.release_changed_config_temporal_upsamplers(old_server_config, new_server_config, changes)
        upsampler_api.release_changed_config_upsamplers(old_server_config, new_server_config, changes)
        if "enable_int8_kernels" in changes:
            self.apply_int8_kernel_setting(new_server_config["enable_int8_kernels"], True)

        model_type = state["model_type"]
        
        model_family_update, model_base_type_update, model_choice_update = self.generate_dropdown_model_list(model_type)
        description_update, header_update = self.generate_header(model_type, compile=new_server_config["compile"], attention_mode=new_server_config["attention_mode"])

        if gen_in_progress:
            msg = "<div style='color:green; text-align:center;'>The new configuration has been succesfully applied. Some of the Settings will be only effective when you will start another Generation</div>"
        else:
            msg = "<div style='color:green; text-align:center;'>The new configuration has been succesfully applied.</div>"

        deepy_visible = deepy_available(new_server_config)
        launcher_update = gr.update(value=assistant_chat.render_launcher_html() if deepy_visible else "", visible=deepy_visible)
        panel_update = gr.update(visible=deepy_visible)

        return (
            msg,
            description_update,
            header_update,
            model_family_update,
            model_base_type_update,
            model_choice_update,
            self.get_unique_id(),
            launcher_update,
            panel_update,
        )
