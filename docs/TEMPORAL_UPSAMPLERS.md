# Temporal Upsampler Plugin API

Temporal upsamplers (RIFE, frame interpolation plugins, ...) are registered in
`postprocessing/temporal_upsamplers.py`. Each upsampler declares its methods and
supported multipliers through a handler object; the registry owns dropdown state,
value parsing, validation helpers, config nesting, downloads, and dispatch.

Temporal upsamplers are used on decoded video frames during generation and in
late post processing. They are video-only; late post processing validation rejects
temporal upsampling for still images.

## Handler Contract

```python
class MyTemporalUpsampler:
    def query_temporal_upsampler_def(self):
        return {
            "name": "MyTemporalUpsampler",
            "config_key": "mytemporal",                       # wgp_config["temporal_upsamplers"] subkey
            "pos": 100,                                       # default dropdown order for this handler's methods
            "method_pos": {"mytemporal": 100},                # optional per-method order
            "methods": [("MyTemporalUpsampler", "mytemporal")],
            "multipliers": {"mytemporal": (2.0, 4.0)},        # supported multipliers per method
            "default_temporal_upsampling": "mytemporal2",
        }

    def is_upsampling(self, value): ...                       # does this handler own the value?
    def split_value(self, value): ...                         # -> (method, scale) or None
    def build_value(self, method, scale): ...                 # -> value or None
    def validate_upsampling(self, value, *, source_is_image=False): ...  # -> "" or error text
    def temporal_upsample(self, value, sample, previous_last_frame, fps, *, processing_device="cuda", abort_callback=None, progress_callback=None, **kwargs): ...
    def download(self, process_files, send_cmd=None, status_text=None, temporal_upsampling=None): ...
    def query_download_def(self, *, enabled_only=True): ...    # -> one process_files definition
    def query_download_defs(self, *, enabled_only=True): ...   # -> list of process_files definitions
    def load_upsampler(self, value, **kwargs): ...             # optional pre-dispatch load hook
    def persistent_models(self): ...                          # optional, True keeps model in RAM and unloads VRAM only
    def release_vram(self): ...                               # optional Configuration-tab release hook
    def enabled(self): ...                                    # optional UI gating
    # optional Configuration tab integration:
    def default_config(self): ...                             # -> dict
    def normalize_config_section(self, section): ...          # -> normalized dict
    def create_config_ui(self, gr, section, *, lock_config=False): ...
    def validate_config_section(self, section): ...           # -> "" or message/list
    def config_requires_release(self, old, new, changed_keys): ...
```

`SimpleScaleSuffixMixin` provides `is_upsampling` / `split_value` / `build_value`
for the common `<method><multiplier>` value encoding, for example `rife2` or
`mytemporal4`.

Handler-exposed method ids in `methods`, `multipliers`, and `method_pos` must be
multiplier-free. The multiplier only appears in serialized
`temporal_upsampling` values returned by `build_value()` or stored as
`default_temporal_upsampling`.

Dropdown entries are sorted by method position, then by method label. A handler
can define a default `pos` and override individual methods with `method_pos`.
Position is independent of multiplier; expanded choices such as `mytemporal2`
and `mytemporal4` share the `mytemporal` method position.

## Runtime Contract

`temporal_upsample(...)` receives the current decoded sample, an optional
`previous_last_frame` for chunk continuity, and the current `fps`. It must return:

```python
return sample, previous_last_frame, fps
```

When a handler inserts frames, it must update `fps` to match the new temporal
rate. For example, x2 interpolation should return `fps * 2`.

`previous_last_frame` is the continuity frame saved from the prior chunk. If the
handler consumes it, it should return the last frame of the postprocessed chunk so
the next chunk can continue smoothly.

Do not assume a different frame layout than the one passed by WanGP. Convert
dtypes or layouts only when the handler implementation requires it, and keep the
returned sample layout compatible with the surrounding WanGP postprocessing flow.

`postprocessing/temporal_upsamplers.py` wraps dispatch in shared attention state,
unloads the main generation offload object before running the temporal upsampler,
and releases registered extension resources after use. If `persistent_models()`
returns `True`, WanGP unloads the handler VRAM through the offload registry but
keeps persistent CPU/RAM state available for the next call.

## Registration

Registration is owned by `postprocessing/temporal_upsamplers.py`. Add the handler
class path to `temporal_upsampler_handlers`:

```python
temporal_upsampler_handlers = [
    "postprocessing.my_temporal.temporal_upsampler.MyTemporalUpsampler",
]
```

`wgp.py` only calls
`temporal_upsampler_api.register_temporal_upsamplers(server_config, fl)`; it
should not import or keep one explicit variable per temporal upsampler.

Enabled plugins can expose temporal upsamplers without editing core code by
adding `temporal_upsampler_handlers` to `plugin_info.json`. Entries that start
with `.` are relative to the plugin package root; entries without a leading `.`
are absolute import paths:

```json
{
  "name": "My Plugin",
  "temporal_upsampler_handlers": [
    ".temporal.MyTemporalUpsampler",
    "./nested/other_temporal.py:OtherTemporalUpsampler",
    "postprocessing.my_temporal.temporal_upsampler.MySharedTemporalUpsampler"
  ]
}
```

Only enabled/loaded plugins are considered. The plugin manager reuses cached
`plugin_info.json` metadata when registering those handlers.

## Configuration

Temporal upsampler settings are stored under
`wgp_config["temporal_upsamplers"][config_key]`.

Handlers with configuration should implement `default_config()` and
`normalize_config_section(...)`. If they expose Configuration-tab controls, return
`[(field_name, gradio_component), ...]` from `create_config_ui(...)`; WanGP uses
that binding to collect, normalize, validate, and save the nested section.

If a config change requires releasing loaded runtime state, implement either
`release_vram()` or `config_requires_release(old, new, changed_keys)`.

## Built-In Example

RIFE is implemented by `postprocessing/rife/temporal_upsampler.py`:

- method id: `rife`
- serialized values: `rife2`, `rife4`
- config key: `rife`
- config section: `wgp_config["temporal_upsamplers"]["rife"]`

## Extension Offload Object Registry

Every extension that creates its own mmgp offload object (`offload.profile(...)`)
must register it in `shared/utils/offload_registry.py`:

```python
offload_registry.register_offloadobj("MyTemporalUpsampler", offloadobj, release_fn)
...
offload_registry.unregister_offloadobj("MyTemporalUpsampler", offloadobj)  # in release_fn
```

This lets WanGP track every extension offload object and release all extension
resources centrally: the toolbar "Unload Models" tool and Configuration plugin
release button call `offload_registry.release_all()`.
