"""Native VideoDeltaNet attention for MiniMax H3.

The implementation follows OpenVDN's released hybrid attention: exact softmax over
nearby VAE-aligned frames plus a bidirectional delta-rule state for distant frames.
Only the learned VDN branch lives here; H3's QKV projections remain shared.
"""

import torch
import torch.nn.functional as F
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


_SLOW_NOTICE_PRINTED = False
_TRITON_NOTICE_PRINTED = False
_SAGE_NOTICE_PRINTED = False
_LINEAR_HEAD_CHUNK = 14


def _notify_slow():
    global _SLOW_NOTICE_PRINTED
    if not _SLOW_NOTICE_PRINTED:
        print("[MiniMax H3 VDN] Triton kernels unavailable; using the slower PyTorch implementation.")
        _SLOW_NOTICE_PRINTED = True


def _notify_triton():
    global _TRITON_NOTICE_PRINTED
    if not _TRITON_NOTICE_PRINTED:
        print("[MiniMax H3 VDN] Triton temporal-convolution kernels are active.")
        _TRITON_NOTICE_PRINTED = True


def _notify_sage():
    global _SAGE_NOTICE_PRINTED
    if not _SAGE_NOTICE_PRINTED:
        print("[MiniMax H3 VDN] SageAttention 2 grouped-window kernels are active.")
        _SAGE_NOTICE_PRINTED = True


if triton is not None:
    @triton.jit
    def _temporal_conv_kernel(src, taps, dst, frames, spatial, channels: tl.constexpr,
                              dim: tl.constexpr, block_t: tl.constexpr, normalize: tl.constexpr):
        frame_block, pixel, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        rows = frame_block * block_t + tl.arange(0, block_t)
        channel = head * dim + tl.arange(0, dim)
        valid = rows < frames
        acc = tl.zeros((block_t, dim), tl.float32)
        for tap in tl.static_range(5):
            source_frame = rows + tap - 2
            mask = valid[:, None] & (source_frame[:, None] >= 0) & (source_frame[:, None] < frames)
            value = tl.load(src + (source_frame[:, None] * spatial + pixel) * channels + channel[None, :], mask=mask, other=0.0)
            coeff = tl.load(taps + channel * 5 + tap)
            acc += value * coeff[None, :]
        value = acc * tl.sigmoid(acc)
        if normalize:
            value *= tl.rsqrt(tl.maximum(tl.sum(value * value, axis=1), 1e-12))[:, None]
        tl.store(dst + (rows[:, None] * spatial + pixel) * channels + channel[None, :], value, mask=valid[:, None])


def _triton_temporal_conv(x, weight, heads, head_dim, l2norm):
    frames, spatial, channels = x.shape
    if triton is None or not x.is_cuda or head_dim < 16 or head_dim & (head_dim - 1):
        return None
    out = torch.empty_like(x)
    _temporal_conv_kernel[((frames + 15) // 16, spatial, heads)](x.contiguous(), weight.contiguous(), out, frames, spatial,
                                                                  channels=channels, dim=head_dim, block_t=16,
                                                                  normalize=l2norm, num_warps=4, num_stages=2)
    _notify_triton()
    return out.reshape(-1, heads, head_dim)


def _activate(x, normalize):
    x = F.silu(x)
    return F.normalize(x, dim=-1, eps=1e-6).to(x.dtype) if normalize else x


class OutputGate(nn.Module):
    def __init__(self, hidden, heads, head_dim=None, bottleneck=None, dtype=None, device=None):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.down = nn.Linear(hidden, bottleneck, bias=False, dtype=dtype, device=device) if bottleneck else None
        self.up = nn.Linear(bottleneck or hidden, heads * (head_dim or 1), dtype=dtype, device=device)

    def forward(self, x):
        x = self.down(x) if self.down is not None else x
        return torch.sigmoid(self.up(x)).view(-1, self.heads, self.head_dim or 1)


class FrameKDAAlpha(nn.Module):
    def __init__(self, hidden, heads, head_dim, dtype=None, device=None):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.down = nn.Linear(hidden, head_dim, bias=False, dtype=dtype, device=device)
        self.up = nn.Linear(head_dim, heads * head_dim, bias=False, dtype=dtype, device=device)
        self.A_log = nn.Parameter(torch.empty(heads, dtype=torch.float32, device=device))
        self.dt_bias = nn.Parameter(torch.empty(heads * head_dim, dtype=torch.float32, device=device))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.down.weight.float()), self.up.weight.float())
            delta = delta.view(-1, self.heads, self.head_dim) + self.dt_bias.float().view(1, self.heads, self.head_dim)
            return torch.exp(-torch.exp(self.A_log.float())[None, :, None] * F.softplus(delta))


class LinearAttentionSepConv(nn.Module):
    KERNEL = 5

    def __init__(self, channels, dtype=None, device=None):
        super().__init__()
        for name in ("k", "v"):
            setattr(self, f"{name}_sp", nn.Conv2d(channels, channels, 5, padding=2, groups=channels,
                                                   bias=False, dtype=dtype, device=device))
            setattr(self, f"{name}_tm", nn.Conv1d(channels, channels, 5, padding=2, groups=channels,
                                                   bias=False, dtype=dtype, device=device))

    def apply(self, name, tokens, frames, frame_size, use_triton):
        heads, head_dim = tokens.shape[-2:]
        height, width = frame_size
        channels = heads * head_dim
        volume = tokens.reshape(frames, height, width, channels).permute(0, 3, 1, 2)
        volume = F.conv2d(volume, getattr(self, f"{name}_sp").weight, padding=2, groups=channels)
        x = volume.permute(0, 2, 3, 1).reshape(frames, height * width, channels)
        weight = getattr(self, f"{name}_tm").weight.squeeze(1).to(x.dtype)
        if use_triton:
            result = _triton_temporal_conv(x, weight, heads, head_dim, name == "k")
            if result is not None:
                return result
        _notify_slow()
        padded = F.pad(x, (0, 0, 0, 0, 2, 2))
        result = sum(padded[tap:tap + frames] * weight[:, tap].view(1, 1, -1) for tap in range(5))
        return _activate(result.reshape(-1, heads, head_dim), name == "k")


def _frame_statistics(key, value, beta):
    with torch.autocast(device_type=key.device.type, enabled=False):
        key16 = key.contiguous()
        key32 = key16.float()
        scaled = (key32 * beta.unsqueeze(-1).float()).contiguous()
        a = scaled.transpose(-1, -2) @ key32
        a = 0.5 * (a + a.transpose(-1, -2))
        b = ((value * beta.unsqueeze(-1).to(value.dtype)).contiguous().transpose(-1, -2) @ key16).float()
        return a, b


def _factor(alpha, a, b):
    eye = torch.eye(a.shape[-1], device=a.device, dtype=torch.float32).expand_as(a)
    chol = torch.linalg.cholesky(a.float() + eye)
    inv_l = torch.linalg.solve_triangular(chol, eye, upper=False)
    del chol, eye
    inv = inv_l.transpose(-1, -2) @ inv_l
    del inv_l
    injection = b.float() @ inv
    inv.mul_(alpha.unsqueeze(-1))
    return inv, injection


def _scan(alpha, statistics_handoff, initial=None):
    a, b = statistics_handoff
    statistics_handoff.clear()
    transition, injection = _factor(alpha, a, b)
    a = b = None
    frames = transition.shape[0]
    initial = torch.zeros_like(injection[0]) if initial is None else initial.to(injection.dtype)
    prefix, suffix = torch.empty_like(injection), torch.empty_like(injection)
    state = initial
    for frame in range(frames):
        torch.baddbmm(injection[frame], state, transition[frame], out=prefix[frame])
        state = prefix[frame]
    state = initial
    for frame in range(frames - 1, -1, -1):
        torch.baddbmm(injection[frame], state, transition[frame], out=suffix[frame])
        state = suffix[frame]
    return prefix, suffix


def _gather(prefix, suffix, alpha, bounds, initial=None):
    frames = len(bounds)
    device = prefix.device
    before = torch.tensor([lo - 1 for lo, _ in bounds], device=device)
    after = torch.tensor([hi + 1 for _, hi in bounds], device=device)
    has_before, has_after = before >= 0, after < frames
    left, right = prefix[before.clamp_min(0)], suffix[after.clamp_max(frames - 1)]
    if initial is not None:
        initial = initial.to(left.dtype)
        left = torch.where(has_before[:, None, None, None], left, initial)
        right = torch.where(has_after[:, None, None, None], right, initial)
    log_prefix = torch.cat((torch.zeros_like(alpha[:1]), torch.log(alpha.clamp_min(1e-12)).cumsum(0)))
    rows = torch.arange(frames, device=device)
    left *= torch.exp(log_prefix[rows + 1] - log_prefix[(before + 1).clamp_min(0)]).unsqueeze(2)
    right *= torch.exp(log_prefix[after.clamp_max(frames)] - log_prefix[rows]).unsqueeze(2)
    if initial is not None:
        return left + right
    return left * has_before[:, None, None, None] + right * has_after[:, None, None, None]


def _window_plan(layout, video_start, frames, per_frame, bounds, device):
    key = (video_start, frames, per_frame, tuple(bounds), str(device))
    cache = getattr(layout, "_vdn_window_plans", None)
    if cache is None:
        cache = layout._vdn_window_plans = {}
    plan = cache.get(key)
    if plan is not None:
        return plan
    global_idx = torch.arange(video_start, device=device)
    anchors = (0, frames - 1)
    dense_rows = torch.cat((global_idx, *(torch.arange(video_start + frame * per_frame,
                                                       video_start + (frame + 1) * per_frame,
                                                       device=device) for frame in anchors)))
    groups = []
    for frame, frame_bounds in enumerate(bounds):
        if frame in anchors:
            continue
        if groups and groups[-1][1] == frame_bounds:
            groups[-1][0].append(frame)
        else:
            groups.append(([frame], frame_bounds))
    packed_groups = []
    for group_frames, (lo, hi) in groups:
        lo, hi = max(lo, 0), min(hi, frames - 1)
        rows = torch.arange(video_start + group_frames[0] * per_frame,
                            video_start + (group_frames[-1] + 1) * per_frame, device=device)
        pieces = [global_idx, torch.arange(video_start + lo * per_frame,
                                            video_start + (hi + 1) * per_frame, device=device)]
        pieces.extend(torch.arange(video_start + anchor * per_frame,
                                   video_start + (anchor + 1) * per_frame, device=device)
                      for anchor in anchors if not lo <= anchor <= hi)
        packed_groups.append((rows, torch.cat(pieces)))
    plan = dense_rows, packed_groups
    cache[key] = plan
    return plan


class VDNLinearBranch(nn.Module):
    def __init__(self, hidden, heads, head_dim, dtype=None, device=None):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.alpha = FrameKDAAlpha(hidden, heads, head_dim, dtype, device)
        self.beta_proj = nn.Linear(hidden, heads, bias=False, dtype=dtype, device=device)
        self.norm = nn.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.output_gate = OutputGate(hidden, heads, head_dim, head_dim, dtype, device)
        self.short_conv = LinearAttentionSepConv(heads * head_dim, dtype, device)

    def _features(self, raw_handoff, frames, frame_size, use_triton):
        raw_q, raw_k, raw_v = raw_handoff
        raw_handoff.clear()
        q = _activate(raw_q, True)
        raw_q = None
        k = self.short_conv.apply("k", raw_k, frames, frame_size, use_triton)
        raw_k = None
        v = self.short_conv.apply("v", raw_v, frames, frame_size, use_triton)
        return q, k, v

    def _text_state(self, text_x, raw_handoff):
        if text_x.numel() == 0:
            raw_handoff.clear()
            return None
        raw_key, raw_value = raw_handoff
        raw_handoff.clear()
        key, value = _activate(raw_key, True), _activate(raw_value, False)
        raw_key = raw_value = None
        beta = torch.sigmoid(self.beta_proj(text_x)).transpose(0, 1).unsqueeze(0)
        key = key.permute(1, 0, 2).unsqueeze(0)
        value = value.permute(1, 0, 2).unsqueeze(0)
        a, b = _frame_statistics(key, value, beta)
        _, injection = _factor(torch.ones(1, self.heads, self.head_dim, device=a.device), a, b)
        return injection[0] * 0.5

    def forward(self, x, raw_handoff, frames, tokens_per_frame, frame_size, bounds, text_x, text_raw_handoff, use_triton):
        if frames <= 2:
            raw_handoff.clear()
            text_raw_handoff.clear()
            return x.new_zeros(x.shape[0], self.heads * self.head_dim)
        inner = slice(tokens_per_frame, (frames - 1) * tokens_per_frame)
        raw_inner = [tensor[inner] for tensor in raw_handoff]
        raw_handoff.clear()
        result = self._forward_inner(x[inner], raw_inner, frames - 2, tokens_per_frame,
                                     frame_size, [(lo - 1, hi - 1) for lo, hi in bounds[1:-1]],
                                     text_x, text_raw_handoff, use_triton)
        output = result.new_zeros(x.shape[0], result.shape[-1])
        output[inner] = result
        return output

    def _forward_inner(self, x, raw_handoff, frames, tokens_per_frame, frame_size, bounds, text_x, text_raw_handoff, use_triton):
        initial = self._text_state(text_x, text_raw_handoff)
        q, k, v = self._features(raw_handoff, frames, frame_size, use_triton)
        shape = (frames, tokens_per_frame, self.heads, self.head_dim)
        qf = q.view(shape).permute(0, 2, 1, 3)
        kf = k.view(shape).permute(0, 2, 1, 3)
        vf = v.view(shape).permute(0, 2, 1, 3)
        beta = torch.sigmoid(self.beta_proj(x)).view(frames, tokens_per_frame, self.heads).permute(0, 2, 1)
        alpha = self.alpha(x.view(frames, tokens_per_frame, -1).mean(1, dtype=torch.float32))
        readout = torch.empty((frames, self.heads, tokens_per_frame, self.head_dim), dtype=x.dtype, device=x.device)
        for first_head in range(0, self.heads, _LINEAR_HEAD_CHUNK):
            heads = slice(first_head, min(first_head + _LINEAR_HEAD_CHUNK, self.heads))
            a, b = _frame_statistics(kf[:, heads], vf[:, heads], beta[:, heads])
            statistics_handoff = [a, b]
            a = b = None
            prefix, suffix = _scan(alpha[:, heads], statistics_handoff, None if initial is None else initial[heads])
            state = _gather(prefix, suffix, alpha[:, heads], bounds, None if initial is None else initial[heads]).to(x.dtype)
            del prefix, suffix
            readout[:, heads] = torch.matmul(qf[:, heads], state.transpose(-1, -2))
            del state
        del qf, kf, vf, q, k, v, beta, alpha, initial
        readout = readout.permute(0, 2, 1, 3).reshape(-1, self.heads, self.head_dim)
        readout = self.norm(readout)
        readout.mul_(self.output_gate(x))
        return readout.reshape(x.shape[0], -1)


class VDNHybridAttention(nn.Module):
    def __init__(self, hidden, heads, head_dim, dtype=None, device=None):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.linear_attention = VDNLinearBranch(hidden, heads, head_dim, dtype, device)
        self.softmax_gate = OutputGate(hidden, heads, dtype=dtype, device=device)
        self.to_out_linear = nn.Linear(heads * head_dim, hidden, bias=False, dtype=dtype, device=device)
        self.layout = None
        self.use_triton = False

    def begin_forward(self, layout, latent_t, latent_h, latent_w, patch_size):
        self.layout = layout
        self.frames = latent_t
        self.frame_size = (latent_h // patch_size[1], latent_w // patch_size[2])
        self.tokens_per_frame = self.frame_size[0] * self.frame_size[1]
        self.video_start = layout.sequence_length - self.frames * self.tokens_per_frame
        self.text_indices = layout.text_indices
        self.bounds = [((frame // 5 - 1) * 5, (frame // 5 + 2) * 5 - 1) for frame in range(self.frames)]
        self.use_triton = triton is not None and torch.cuda.is_available()
        if not self.use_triton:
            _notify_slow()

    def forward(self, x_handoff, raw_qkv, softmax_qkv, original_out):
        x = x_handoff.pop()
        video = slice(self.video_start, x.shape[0])
        local = self._window_softmax(softmax_qkv)
        local.mul_(self.softmax_gate(x))
        output = original_out(local.reshape(x.shape[0], -1))
        del local
        text_idx = self.text_indices.to(x.device)
        query, key, value = raw_qkv
        raw_qkv.clear()
        video_raw = [query[video], key[video], value[video]]
        text_raw = [key[text_idx], value[text_idx]]
        query = key = value = None
        branch = self.linear_attention(x[video], video_raw, self.frames,
                                       self.tokens_per_frame, self.frame_size, self.bounds, x[text_idx],
                                       text_raw, self.use_triton)
        output[video].add_(self.to_out_linear(branch))
        return output

    def _window_softmax(self, qkv_handoff):
        from shared.attention import pay_attention, sage2_supported

        query, key, value = qkv_handoff
        qkv_handoff.clear()
        video_start, frames, per_frame = self.video_start, self.frames, self.tokens_per_frame
        output = torch.empty_like(query)
        force_attention = "sage2" if query.is_cuda and sage2_supported else "sdpa"
        dense_rows, groups = _window_plan(self.layout, video_start, frames, per_frame, self.bounds, query.device)
        if force_attention == "sage2":
            _notify_sage()
        output[:, dense_rows] = pay_attention([query[:, dense_rows], key, value], force_attention=force_attention, recycle_q=True)
        for rows, indices in groups:
            output[:, rows] = pay_attention([query[:, rows], key[:, indices], value[:, indices]], force_attention=force_attention, recycle_q=True)
        return output


__all__ = ["VDNHybridAttention"]
