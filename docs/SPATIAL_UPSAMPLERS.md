# Spatial Upsampler Plugin API

Spatial upsamplers (Lanczos, FlashVSR, PiD, Chain-of-Zoom, VAE upscalers, ...) are
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
        }

    def is_upsampling(self, value): ...                    # does this handler own the value?
    def split_value(self, value): ...                      # -> (method, scale) or None
    def build_value(self, method, scale): ...              # -> value or None
    def validate_upsampling(self, value, image_mode): ...  # -> "" or error text
    # postprocessing type only:
    def upscale(self, sample, value, *, seed, ..., abort_callback, progress_callback): ...
    def download(self, process_files, send_cmd=None, status_text=None, spatial_upsampling=None): ...
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
