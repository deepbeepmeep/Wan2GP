import torch
from shared.attention import pay_attention


def _normalize_attention_engine(engine):
    if engine is None or engine == "auto":
        return None, None
    if engine == "flash_attention_2":
        return "flash", None
    if engine == "flash_attention_3":
        return "flash", 3
    return engine, None


def _expand_attention_mask(attention_mask, q_len, num_heads):
    if attention_mask is None:
        return None
    mask = attention_mask
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    if mask.dim() == 2:
        mask = mask[:, None, None, :]
        return mask.expand(-1, q_len, num_heads, -1)
    if mask.dim() == 3:
        if mask.shape[1] == q_len:
            mask = mask[:, :, None, :]
            return mask.expand(-1, q_len, num_heads, -1)
        if mask.shape[1] == num_heads:
            mask = mask[:, None, :, :]
            return mask.expand(-1, q_len, num_heads, -1)
    if mask.dim() == 4:
        if mask.shape[1] == num_heads and mask.shape[2] == q_len:
            return mask.transpose(1, 2)
        if mask.shape[2] == num_heads and mask.shape[1] == q_len:
            return mask
    return mask


def apply_attention(q, k, v, attention_mask=None, force_attention=None, flash_version=None):
    q_len = q.shape[1]
    num_heads = q.shape[2]
    mask = _expand_attention_mask(attention_mask, q_len, num_heads)
    qkv_list = [q, k, v]
    return pay_attention(
        qkv_list,
        attention_mask=mask,
        force_attention=force_attention,
        version=flash_version,
    )


class SelfAttentionEngine:
    def __init__(self, engine="auto"):
        self.force_attention, self.flash_version = _normalize_attention_engine(engine)

    def __call__(self, q, k, v, attn_mask=None):
        return apply_attention(
            q,
            k,
            v,
            attention_mask=attn_mask,
            force_attention=self.force_attention,
            flash_version=self.flash_version,
        )

    def get_attention(self):
        return self.__call__

