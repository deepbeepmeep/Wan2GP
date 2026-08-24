# Spatial Upsampler / Visual Refiner Plugin API

Spatial upsamplers and visual refiners (Lanczos, FlashVSR, PiD, H3 Face Refiner,
Chain-of-Zoom, VAE upscalers, ...) are
registered in `postprocessing/spatial_upsamplers.py`. Each upsampler declares itself and its
capabilities through a handler object; the registry owns dropdown state, value
parsing, validation helpers, config nesting, downloads and dispatch.

## Upsampler types

- `postprocessing`: works on decoded frames. Interchangeable: WanGP can call any of
  them through the same `upscale()` interface, both at generation time and in
  **late post processing** (Post Processing tab on existing media). Image-only
  handlers (`"media": ("image",)`) are offered for image outputs and in late
  post processing.
- `vae`: plugged directly into a model pipeline (e.g. Wan VAE 2x, PiD VAE). They
  expose capabilities through the same API. Handlers declare whether a selected
  VAE upsampler requires a main model reload (Wan VAE) or an external runtime
  session passed to the model pipeline (PiD-style upsamplers).

Decoded-media handlers also declare `postprocessing_category`:

- `upsampler`: increases spatial resolution and declares one or more scale
  multipliers.
- `refiner`: improves decoded media without necessarily resizing it. A refiner
  may omit `multipliers`; WanGP then hides the Scale control and serializes the
  bare method id.

## Handler contract

```python
class MyUpsampler:
    def query_upsampler_def(self):
        return {
            "name": "MyUpsampler",
            "upsampler_types": ("postprocessing",),        # and/or "vae"
            "media": ("video", "image"),
            "profile": "video",                            # memory profile: video, image, or audio
            "config_key": "myup",                          # wgp_config["spatial_upsamplers"] subkey
            "pos": 100,                                     # default dropdown order for this handler's methods
            "method_pos": {"myup": 100},                    # optional per-method order; independent of multiplier
            "methods": [("MyUpsampler", "myup")],          # method dropdown entries
            "vae_methods": [],                             # VAE entries (manual integration)
            "multipliers": {"myup": (2.0, 4.0)},           # supported multipliers per method
            "default_spatial_upsampling": "myup2",
            "postprocessing_category": "upsampler",       # "upsampler" or "refiner"
            "description": "Upscale while restoring detail.", # processor-owned help/discovery text
            "media_descriptions": {"video": "Uses overlapping windows."}, # optional media-specific help
            "method_descriptions": {"myup": "..."},       # optional per-method descriptions
            "method_parameters": {"myup": [{
                "name": "spatial_upsampler_prompt",       # common prefix for UI parameters
                "setting": "prompt",                      # upscale() keyword
                "type": "string",
                "component": "textbox",
                "ui": ("late_postprocessing",),           # where to display the control
                "required": False,
                "default": "",
                "label": "Refiner Prompt",
                "description": "Optional refinement instructions.",
            }]},
            "source_audio_conditioning": False,            # request source audio in late postprocessing
        }

    def is_upsampling(self, value): ...                    # does this handler own the value?
    def split_value(self, value): ...                      # -> (method, scale) or None
    def build_value(self, method, scale): ...              # -> value or None
    def validate_upsampling(self, value, image_mode): ...  # -> "" or error text
    # postprocessing type only:
    def upscale(self, sample, value, *, seed, ..., abort_callback, progress_callback): ...
    def download(self, process_files, send_cmd=None, status_text=None, spatial_upsampling=None): ...
    def load_upsampler(self, value, **kwargs): ...            # optional pre-dispatch load hook
    def supports_loaded_model(self, value, context, **kwargs): ... # optional core-model borrowing
    def release_private_runtime(self): ...                    # optional before borrowing core model
    def release_vram(self): ...
    def enabled(self): ...                                 # optional UI gating
    # VAE type only:
    def supports_model_vae_method(self, method, model_type, model_def, image_mode): ...
    def prepare_vae_upsampler(self, value, *, send_cmd, process_files, init_pipe, profile, attention_mode=None): ...
    def model_load_upsampling_value(self, value, model_type, model_def, image_mode): ...
    def loaded_model_vae_upsampling_value(self, model): ...
    def model_load_kwargs_for_vae_upsampling(self, value, model_type, model_def, image_mode): ...
    # optional Configuration tab integration:
    def default_config(self): ...                          # -> dict
    def legacy_config(self, server_config): ...            # -> old top-level values, if any
    def legacy_config_keys(self): ...                      # -> keys deleted after migration
    def normalize_config_section(self, section): ...       # -> normalized dict
    def create_config_ui(self, gr, section, *, lock_config=False): ...
    def validate_config_section(self, section): ...        # -> "" or message/list
    def config_requires_release(self, old, new, changed_keys): ...
```

`SimpleScaleSuffixMixin` provides `is_upsampling`/`split_value`/`build_value` for the
common `<method><multiplier>` value encoding (e.g. `lanczos2`, `coz4`).
Handler-exposed method ids in `methods`, `vae_methods`, `multipliers`,
`method_pos`, `model_def["vae_upsamplers"]`, and
`model_def["excluded_spatial_upsamplers"]` must be multiplier-free. The multiplier
only appears in serialized `spatial_upsampling` values returned by `build_value()`
or stored as `default_spatial_upsampling`.

Dropdown entries are sorted by method position, then by method label. A handler
can define a default `pos` and override individual methods with `method_pos`.
Position is independent of multiplier; expanded choices such as `myup2` and
`myup4` share the `myup` method position.

### Borrowing the loaded generation model

A postprocessor that can run through an already loaded generation pipeline may
implement `supports_loaded_model(value, context)`. The context contains the
core-owned pipeline, MMGP offload object, model type/family/definition, profile,
and selected config. If the method returns true, the registry skips
`load_upsampler()`, passes the context to `upscale(...,
loaded_model_context=context)`, and leaves ownership of the model and offload
object with WanGP. `release_private_runtime()` is called first so the handler
does not retain a duplicate private model. If compatibility returns false, the
normal private-runtime path is used; this is also the fallback when no core
model is loaded.

Borrowed runtimes are always released from RAM after the call, even when shared
spatial-upscaler persistence is enabled. This detaches borrowed model references
before the core MMGP owner can be released or replaced.

Borrowing is an explicit capability contract, not an architecture guess. A
pipeline should advertise the protocol its handler expects (for example H3 uses
`refinement_api = "masked_video_sigma_v1"`). A borrowing handler that changes
temporary model state, such as active LoRAs or caches, must restore that state
before returning.

Discovery consumers infer a required `multiplier` parameter only when the method
declares `multipliers`. `description` or `method_descriptions` supplies both
Deepy's process description and the dynamic help next to WanGP's selector.
`media_descriptions` can add guidance specific to `image` or `video`. The UI
compiles help only from methods available for the current media and, during
generation, the current model; late postprocessing follows the selected gallery
item.

Each `method_parameters` entry is a list of dictionaries with at least `name`,
`type`, `description`, and `required`. It may also define `default`, `enum`,
`minimum`, `maximum`, and a runtime keyword override named `setting`. UI-exposed
parameter names must use the shared `spatial_upsampler_` prefix. Built-in
parameters must also declare their default as a separate top-level entry in
`models/_settings.json`; do not group method parameters inside one settings
dictionary. `ui` selects
one or both UI contexts: `postprocessing` means the normal generation-time Post
Processing section, while `late_postprocessing` means the Post Processing tab
for an existing gallery item. A parameter can still be inferred and passed by
WanGP when it is absent from a UI context; H3, for example, receives generation
prompt/reference data without showing redundant controls during generation.

Supported generic UI components are `textbox`, `number`, `slider`, `dropdown`,
`checkbox`, and `images`. Image parameters are rendered by
`AdvancedMediaGallery`; set `multiple` to `True` for an ordered list or `False`
for one image. Deepy receives only the call-relevant fields (`name`, `type`,
`description`, `required`, `default`, `enum`, `minimum`, `maximum`, and
`media_type`), so UI/runtime fields such as `component`, `ui`, `label`, `step`,
and `setting` do not consume assistant context. Parameters with `media_type:
"image"` are resolved from media ids to paths. Each runtime value remains a flat
queue setting under its generic parameter `name`; dispatch maps it to the
`upscale()` keyword named by `setting`.

Registration is owned by `postprocessing/spatial_upsamplers.py`. Add the handler class path
to `spatial_upsampler_handlers`:

```python
spatial_upsampler_handlers = [
    "postprocessing.my_upsampler.wgp_bridge.MyUpsampler",
]
```

`wgp.py` only calls `upsampler_api.register_spatial_upsamplers(server_config, fl)`;
it should not import or keep one explicit variable per spatial upsampler.

Enabled plugins can expose upsamplers without editing core code by adding
`spatial_upsampler_handlers` to `plugin_info.json`. Entries that start with `.`
are relative to the plugin package root; entries without a leading `.` are
absolute import paths:

```json
{
  "name": "My Plugin",
  "spatial_upsampler_handlers": [
    ".upsampler.MyUpsampler",
    "./nested/other_upsampler.py:OtherUpsampler",
    "postprocessing.my_upsampler.wgp_bridge.MySharedUpsampler"
  ]
}
```

Only enabled/loaded plugins are considered. The plugin manager reuses its cached
`plugin_info.json` metadata when registering those handlers, so the file is not
parsed a second time for upsampler discovery.

Upsampler settings are stored under `wgp_config["spatial_upsamplers"][config_key]`.
Handlers can read old top-level keys during migration with `legacy_config()`, but
those keys are deleted after the nested section is written.

SeedVR2 stores `window_size` under `spatial_upsamplers.seedvr2`. `0` selects the
GPU-based automatic limit, `-1` disables windowing, and positive values are
finite frame limits. SeedVR2 aligns finite limits down to its required `4n+1`
input shape and crossfades three overlapping output frames.

The LTX video handler exposes LTX 2.3 and LTX 2.5 as decoded-video x2 methods.
It reuses the existing checkpoint declarations and LTX family loader under
private runtime model types and always creates its MMGP offload object with
memory profile 5. Each source window is VAE encoded and appended as a
downscale-2 reference for the matching official Pixel Spatial Upscaler IC-LoRA;
the x2 target starts from noise, follows the official eight-step distilled sigma
schedule, and is VAE decoded. LTX 2.3 uses the Dev checkpoint with Distilled
1.1 at 0.5 and the x2 IC-LoRA at 1.0; LTX 2.5 uses its distilled checkpoint with
the official 2.5 x2 IC-LoRA at 1.0. Inputs longer than the configured window use
stride-aligned windows (81 frames with a 17-frame overlap by default);
overlapping windows address the same deterministic global noise and time
coordinates. Both versions are shown in Post Processing and Late Postprocessing,
and Media Flow keeps explicit processes for both versions. The Configuration
plugin exposes only the shared LTX window size and overlap controls under
`spatial_upsamplers.ltx2`. Both values follow
the VAE's `8n+1` frame cadence; window size ranges from 9 to 481 frames, with 81
as the default. The values are read at the start of every native or Media Flow
upscale. Audio remains under the existing WGP and Media Flow preservation paths;
the LTX spatial upsampler itself only returns video frames.

Model persistence is a registry-wide setting stored at
`wgp_config["spatial_upsamplers"]["persistence"]`; handlers must not expose a
separate persistence control in their own config section. The registry retains
at most one spatial upsampler handler. When dispatch changes handlers, it fully
releases the previous handler before loading the new one. A handler remains
responsible for releasing incompatible variants that share that handler, such as
PiD version, backbone, profile, dtype, or checkpoint-set changes.

Models declare external VAE upsampler support with method ids under
`model_def["vae_upsamplers"]`, for example:

```python
{
    "vae_upsamplers": {
        "flux_vae_pid": [1, 2],
        "flux2_vae_pid": {"image_modes": [1]},
    },
    "excluded_spatial_upsamplers": ["flux_pid"]
}
```

`excluded_spatial_upsamplers` hides post-processing methods that should not be
offered for that model, for example when a model supports the corresponding VAE
upsampler path and should not show both choices in the generation UI.

Optional attributes honored by `wgp.py`: `batch_image_inputs` (process image batches
in one call instead of per-frame). `uses_image_profile` is still accepted for older
handlers, but new handlers should declare `"profile"`.

## Extension offload object registry

Every extension that creates its own mmgp offload object (`offload.profile(...)`)
must register it in `shared/utils/offload_registry.py`:

```python
offload_registry.register_offloadobj("MyUpsampler", offloadobj, release_fn)
...
offload_registry.unregister_offloadobj("MyUpsampler", offloadobj)  # in release_fn
```

This lets WanGP track every extension offload object and release all extension
resources centrally: the toolbar "Unload Models" tool (and the Configuration plugin
release button) calls `offload_registry.release_all()`.
