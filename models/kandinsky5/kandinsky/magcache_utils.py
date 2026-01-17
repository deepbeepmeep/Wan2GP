# This is an adaptation of Magcache from https://github.com/Zehong-Ma/MagCache/
import numpy as np
import torch
from types import MethodType


def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


def _prepare_mag_ratios(mag_ratios, num_steps):
    if mag_ratios is None:
        return None
    mag_ratios = np.array([1.0] * 2 + list(mag_ratios))
    if len(mag_ratios) != num_steps * 2:
        mag_ratio_con = nearest_interp(mag_ratios[0::2], num_steps)
        mag_ratio_ucon = nearest_interp(mag_ratios[1::2], num_steps)
        mag_ratios = np.concatenate(
            [mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1
        ).reshape(-1)
    return mag_ratios


def compute_magcache_threshold(
    mag_ratios,
    num_steps,
    speed_factor,
    start_step=0,
    no_cfg=False,
    magcache_K=2,
    retention_ratio=0.2,
):
    if mag_ratios is None or num_steps <= 0 or speed_factor is None or speed_factor <= 0:
        return None

    mag_ratios = _prepare_mag_ratios(mag_ratios, num_steps)
    if mag_ratios is None:
        return None

    total_calls = num_steps if no_cfg else num_steps * 2
    target_calls = max(1, int(total_calls / max(speed_factor, 1e-6)))
    retention_cnt = int(num_steps * 2 * retention_ratio)

    best_threshold = 0.01
    best_diff = float("inf")
    best_signed_diff = 0

    threshold = 0.01
    while threshold <= 0.6:
        nb_calls = 0
        accumulated_err = [0.0, 0.0]
        accumulated_steps = [0, 0]
        accumulated_ratio = [1.0, 1.0]

        for i in range(total_calls):
            cnt = i * 2 if no_cfg else i
            step_no = cnt // 2
            skip_forward = False

            if cnt >= retention_cnt and (start_step is None or step_no > start_step):
                stream = cnt % 2
                cur_mag_ratio = mag_ratios[cnt]
                accumulated_ratio[stream] *= cur_mag_ratio
                accumulated_steps[stream] += 1
                cur_skip_err = np.abs(1 - accumulated_ratio[stream])
                accumulated_err[stream] += cur_skip_err

                if accumulated_err[stream] < threshold and accumulated_steps[stream] <= magcache_K:
                    skip_forward = True
                else:
                    accumulated_err[stream] = 0.0
                    accumulated_steps[stream] = 0
                    accumulated_ratio[stream] = 1.0

            if not skip_forward:
                nb_calls += 1

        signed_diff = target_calls - nb_calls
        diff = abs(signed_diff)
        if diff < best_diff:
            best_threshold = threshold
            best_diff = diff
            best_signed_diff = signed_diff
        elif diff > best_diff:
            break
        threshold += 0.01

    nb_calls = target_calls - best_signed_diff
    achieved_speed = total_calls / max(1, nb_calls)
    print(
        f"Mag Cache, best threshold found:{best_threshold:0.2f} "
        f"with gain x{achieved_speed:0.2f} for a target of x{speed_factor}"
    )
    return best_threshold


def set_magcache_params(
    dit,
    mag_ratios,
    num_steps,
    no_cfg,
    start_step=None,
    magcache_thresh=None,
    magcache_K=None,
    retention_ratio=None,
):
    print('using Magcache')
    dit.forward = MethodType(magcache_forward, dit)
    dit.cnt = 0
    dit.num_steps = num_steps * 2
    dit.magcache_thresh = 0.12 if magcache_thresh is None else magcache_thresh
    dit.K = 2 if magcache_K is None else magcache_K
    dit.accumulated_err = [0.0, 0.0]
    dit.accumulated_steps = [0, 0]
    dit.accumulated_ratio = [1.0, 1.0]
    dit.consecutive_skips = [0, 0]
    dit.magcache_start_step = 0 if start_step is None else int(start_step)
    dit.retention_ratio = 0.2 if retention_ratio is None else retention_ratio
    dit.magcache_retention_cnt = int(dit.num_steps * dit.retention_ratio)
    dit.residual_cache = [None, None]
    dit.mag_ratios = _prepare_mag_ratios(mag_ratios, num_steps)
    dit.no_cfg = no_cfg


def _magcache_should_skip(dit, cnt):
    stream = cnt % 2
    skip_forward = False
    residual_visual_embed = None

    step_no = cnt // 2
    if cnt >= dit.magcache_retention_cnt and step_no > dit.magcache_start_step:
        cur_mag_ratio = dit.mag_ratios[cnt]
        dit.accumulated_ratio[stream] *= cur_mag_ratio
        dit.accumulated_steps[stream] += 1
        cur_skip_err = np.abs(1 - dit.accumulated_ratio[stream])
        dit.accumulated_err[stream] += cur_skip_err

        if dit.accumulated_err[stream] < dit.magcache_thresh and dit.accumulated_steps[stream] <= dit.K:
            if getattr(dit, "consecutive_skips", [0, 0])[stream] < dit.K:
                skip_forward = True
                residual_visual_embed = dit.residual_cache[stream]
            else:
                dit.accumulated_err[stream] = 0.0
                dit.accumulated_steps[stream] = 0
                dit.accumulated_ratio[stream] = 1.0
        else:
            dit.accumulated_err[stream] = 0.0
            dit.accumulated_steps[stream] = 0
            dit.accumulated_ratio[stream] = 1.0

    return skip_forward, residual_visual_embed, stream


def _magcache_forward_single(
    self,
    x,
    text_embed,
    pooled_text_embed,
    time,
    visual_rope_pos,
    text_rope_pos,
    scale_factor=(1.0, 1.0, 1.0),
    sparse_params=None,
    attention_mask=None
):
    text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
        text_embed, time, pooled_text_embed, x, text_rope_pos)
    x = None
    pooled_text_embed = None

    for text_transformer_block in self.text_transformer_blocks:
        text_embed = text_transformer_block(text_embed, time_embed, text_rope, attention_mask)
        if self._check_interrupt():
            return None
    text_rope = None

    visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
        visual_embed, visual_rope_pos, scale_factor, sparse_params)
    visual_rope_pos = None

    skip_forward, residual_visual_embed, stream = _magcache_should_skip(self, self.cnt)

    if skip_forward and residual_visual_embed is None:
        skip_forward = False

    if skip_forward:
        cache = getattr(self, "cache", None)
        if cache is not None and hasattr(cache, "skipped_steps") and stream == 0:
            cache.skipped_steps += 1
        visual_embed = visual_embed + residual_visual_embed
        if hasattr(self, "consecutive_skips"):
            self.consecutive_skips[stream] += 1
    else:
        if hasattr(self, "consecutive_skips"):
            self.consecutive_skips[stream] = 0
        ori_visual_embed = visual_embed.clone()
        for visual_transformer_block in self.visual_transformer_blocks:
            visual_embed = visual_transformer_block(visual_embed, text_embed, time_embed,
                                                    visual_rope, sparse_params, attention_mask)
            if self._check_interrupt():
                return None
        torch.sub(visual_embed, ori_visual_embed, out=ori_visual_embed)
        residual_visual_embed = ori_visual_embed

    self.residual_cache[stream] = residual_visual_embed
    visual_rope = None

    x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)

    if self.no_cfg:
        self.cnt += 2
    else:
        self.cnt += 1

    if self.cnt >= self.num_steps: 
        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
        if hasattr(self, "consecutive_skips"):
            self.consecutive_skips = [0, 0]
    return x


def _magcache_forward_joint(
    self,
    x_list,
    text_embed_list,
    pooled_text_embed_list,
    time_list,
    visual_rope_pos_list,
    text_rope_pos_list,
    scale_factor_list,
    sparse_params_list,
    attention_mask_list,
):
    count = len(x_list)
    text_embed_list = self._normalize_list(text_embed_list, count)
    pooled_text_embed_list = self._normalize_list(pooled_text_embed_list, count)
    time_list = self._normalize_list(time_list, count)
    visual_rope_pos_list = self._normalize_list(visual_rope_pos_list, count)
    text_rope_pos_list = self._normalize_list(text_rope_pos_list, count)
    scale_factor_list = self._normalize_list(scale_factor_list, count)
    sparse_params_list = self._normalize_list(sparse_params_list, count)
    attention_mask_list = self._normalize_list(attention_mask_list, count)

    text_embed_out = [None] * count
    time_embed_out = [None] * count
    text_rope_out = [None] * count
    visual_embed_out = [None] * count

    for idx in range(count):
        text_embed_out[idx], time_embed_out[idx], text_rope_out[idx], visual_embed_out[idx] = (
            self.before_text_transformer_blocks(
                text_embed_list[idx],
                time_list[idx],
                pooled_text_embed_list[idx],
                x_list[idx],
                text_rope_pos_list[idx],
            )
        )
        x_list[idx] = None
        pooled_text_embed_list[idx] = None

    for text_transformer_block in self.text_transformer_blocks:
        for idx in range(count):
            text_embed_out[idx] = text_transformer_block(
                text_embed_out[idx], time_embed_out[idx], text_rope_out[idx], attention_mask_list[idx]
            )
            if self._check_interrupt():
                return None
    for idx in range(count):
        text_rope_out[idx] = None

    visual_shape_list = [None] * count
    to_fractal_list = [None] * count
    visual_rope_out = [None] * count

    for idx in range(count):
        visual_embed_out[idx], visual_shape_list[idx], to_fractal_list[idx], visual_rope_out[idx] = (
            self.before_visual_transformer_blocks(
                visual_embed_out[idx],
                visual_rope_pos_list[idx],
                scale_factor_list[idx],
                sparse_params_list[idx],
            )
        )
        visual_rope_pos_list[idx] = None

    stream_ids = [0] * count
    skip_forward = [False] * count
    residual_visual_embed = [None] * count
    ori_visual_embed = [None] * count

    cnt = self.cnt
    for idx in range(count):
        stream_ids[idx] = cnt % 2
        skip_forward[idx], residual_visual_embed[idx], _ = _magcache_should_skip(self, cnt)
        if skip_forward[idx] and residual_visual_embed[idx] is None:
            skip_forward[idx] = False
        if not skip_forward[idx] and hasattr(self, "consecutive_skips"):
            self.consecutive_skips[stream_ids[idx]] = 0
        if skip_forward[idx]:
            if idx == 0:
                cache = getattr(self, "cache", None)
                if cache is not None and hasattr(cache, "skipped_steps"):
                    cache.skipped_steps += 1
            visual_embed_out[idx] = visual_embed_out[idx] + residual_visual_embed[idx]
            if hasattr(self, "consecutive_skips"):
                self.consecutive_skips[stream_ids[idx]] += 1
        else:
            ori_visual_embed[idx] = visual_embed_out[idx].clone()
        cnt += 2 if self.no_cfg else 1

    for visual_transformer_block in self.visual_transformer_blocks:
        for idx in range(count):
            if skip_forward[idx]:
                continue
            visual_embed_out[idx] = visual_transformer_block(
                visual_embed_out[idx],
                text_embed_out[idx],
                time_embed_out[idx],
                visual_rope_out[idx],
                sparse_params_list[idx],
                attention_mask_list[idx],
            )
            if self._check_interrupt():
                return None
    for idx in range(count):
        visual_rope_out[idx] = None

    for idx in range(count):
        if skip_forward[idx]:
            residual = residual_visual_embed[idx]
        else:
            torch.sub(visual_embed_out[idx], ori_visual_embed[idx], out=ori_visual_embed[idx])
            residual = ori_visual_embed[idx]
        self.residual_cache[stream_ids[idx]] = residual

    outputs = []
    for idx in range(count):
        outputs.append(
            self.after_blocks(
                visual_embed_out[idx],
                visual_shape_list[idx],
                to_fractal_list[idx],
                text_embed_out[idx],
                time_embed_out[idx],
            )
        )

    self.cnt = cnt
    if self.cnt >= self.num_steps:
        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
        if hasattr(self, "consecutive_skips"):
            self.consecutive_skips = [0, 0]
    return outputs


def magcache_forward(
    self,
    x,
    text_embed,
    pooled_text_embed,
    time,
    visual_rope_pos,
    text_rope_pos,
    scale_factor=(1.0, 1.0, 1.0),
    sparse_params=None,
    attention_mask=None,
):
    if isinstance(x, (list, tuple)):
        return _magcache_forward_joint(
            self,
            list(x),
            text_embed,
            pooled_text_embed,
            time,
            visual_rope_pos,
            text_rope_pos,
            scale_factor,
            sparse_params,
            attention_mask,
        )
    return _magcache_forward_single(
        self,
        x,
        text_embed,
        pooled_text_embed,
        time,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=scale_factor,
        sparse_params=sparse_params,
        attention_mask=attention_mask,
    )
