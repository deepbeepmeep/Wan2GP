"""Convert diffusers-port Krea 2 LoRA key naming to WanGP's native naming.

The diffusers port of Krea 2 renames every module (transformer_blocks/attn.to_q
style); LoRAs trained against it (a sizable share of Civitai's Krea 2 catalog)
fail to load in WanGP with "unexpected module keys"
(https://github.com/deepbeepmeep/Wan2GP/issues/1994). The mapping is 1:1 —
both implementations use unfused attention and the same SwiGLU layout — so a
pure key rename suffices; tensors are untouched.

Runs inside mmgp's preprocess_sd hook, i.e. BEFORE mmgp strips the
"transformer."/"diffusion_model." prefix — rules therefore keep any prefix
intact and are anchored on interior segments. Suffix-agnostic: the same rules
cover .lora_A/.lora_B/.lora_down/.lora_up/.alpha/.dora_scale keys.

Torch-free on purpose so it can be unit-tested standalone.
"""

# Markers that only ever appear in the diffusers-port naming.
_DIFFUSERS_MARKERS = ("transformer_blocks.", "text_fusion.", ".attn.to_q.")

# Ordered rules; each is (search, replace) on the full key string.
# ".attn.to_out.0." must precede the other attn renames only for clarity —
# none of the patterns overlap, but keep the compound one first anyway.
_RENAME_RULES = (
    (".attn.to_out.0.", ".attn.wo."),
    (".attn.to_q.", ".attn.wq."),
    (".attn.to_k.", ".attn.wk."),
    (".attn.to_v.", ".attn.wv."),
    (".attn.to_gate.", ".attn.gate."),
    (".ff.", ".mlp."),                      # SwiGLU field names (up/gate/down) already match
    ("text_fusion.", "txtfusion."),
    ("transformer_blocks.", "blocks."),     # safe: the "transformer." *prefix* ends with "." before "transformer_blocks"
    ("time_embed.linear_1.", "tmlp.0."),    # nn.Sequential(Linear, GELU, Linear)
    ("time_embed.linear_2.", "tmlp.2."),
    ("time_mod_proj.", "tproj.1."),         # nn.Sequential(GELU, Linear)
    ("txt_in.linear_1.", "txtmlp.1."),      # nn.Sequential(RMSNorm, Linear, GELU, Linear)
    ("txt_in.linear_2.", "txtmlp.3."),
    ("final_layer.linear.", "last.linear."),
    ("img_in.", "first."),
)


def is_diffusers_lora(state_dict):
    for key in state_dict:
        for marker in _DIFFUSERS_MARKERS:
            if marker in key:
                return True
    return False


def convert_diffusers_lora(state_dict):
    """Return a converted copy when the dict uses diffusers-port naming, else
    the dict unchanged. Idempotent: converted output carries no markers."""
    if not is_diffusers_lora(state_dict):
        return state_dict
    print("Converting Krea 2 Lora from Diffusers naming to native naming")
    converted = {}
    for key, value in state_dict.items():
        for search, replace in _RENAME_RULES:
            if search in key:
                key = key.replace(search, replace)
        converted[key] = value
    return converted
