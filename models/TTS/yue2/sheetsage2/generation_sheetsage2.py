"""Prompt grammar, cached generation and overlap context from the evaluated model."""
import copy
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch

FULL_TASK_PROMPTS = ("timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full")
FIELD_TO_INDEX = {"timestamp": 0, "rhythm": 1, "structure": 2, "key": 3, "chord": 4, "melody": 5}


class PromptGrammarState:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_events = 0
        self.in_shift = True
        self.shift_run = 0
        self.payload_count = 0
        self.last_field_index = -1
        self.incomplete = None

    def _allow_field_starts(self, allowed):
        tokenizer = self.tokenizer
        if self.last_field_index < FIELD_TO_INDEX["timestamp"]:
            allowed[tokenizer.time_token_start : tokenizer.time_token_end] = True
        if self.last_field_index < FIELD_TO_INDEX["rhythm"]:
            allowed[tokenizer.meter_token_start : tokenizer.meter_token_end] = True
            allowed[
                tokenizer.eighth_position_token_start : tokenizer.eighth_position_token_end
            ] = True
        if self.last_field_index < FIELD_TO_INDEX["structure"]:
            allowed[
                tokenizer.structure_token_start : tokenizer.structure_token_end
            ] = True
        if self.last_field_index < FIELD_TO_INDEX["key"]:
            allowed[tokenizer.key_token_start : tokenizer.key_token_end] = True
        if self.last_field_index < FIELD_TO_INDEX["chord"]:
            allowed[
                tokenizer.full_chord_token_start : tokenizer.full_chord_token_end
            ] = True
        if self.last_field_index <= FIELD_TO_INDEX["melody"]:
            allowed[tokenizer.pitch_token_start : tokenizer.pitch_token_end] = True

    def allowed(self, device):
        tokenizer = self.tokenizer
        allowed = torch.zeros(tokenizer.n_tokens, dtype=torch.bool, device=device)
        can_end = self.payload_count > 0

        if can_end:
            allowed[tokenizer.eos_token] = True
        if self.payload_count > 0 or self.in_shift:
            if self.shift_run < 4:
                allowed[
                    tokenizer.subbeat_shift_token_start : tokenizer.subbeat_shift_token_end
                ] = True

        if self.incomplete == "rhythm_after_meter":
            allowed[
                tokenizer.eighth_position_token_start : tokenizer.eighth_position_token_end
            ] = True
            return allowed

        if self.incomplete == "melody_after_pitch":
            allowed[tokenizer.duration_token_start : tokenizer.duration_token_end] = True
            allowed[tokenizer.pitch_token_start : tokenizer.pitch_token_end] = True
            return allowed

        self._allow_field_starts(allowed)
        return allowed

    def update(self, token):
        tokenizer = self.tokenizer
        token = int(token)
        token_type = tokenizer.token_type(token)
        if token == tokenizer.eos_token:
            return True
        if token_type == "subbeat_shift":
            if not self.in_shift and self.payload_count > 0:
                self.generated_events += 1
                self.payload_count = 0
                self.last_field_index = -1
                self.incomplete = None
            self.in_shift = True
            self.shift_run += 1
            return False

        self.in_shift = False
        self.shift_run = 0
        self.payload_count += 1
        if token_type == "time":
            self.last_field_index = FIELD_TO_INDEX["timestamp"]
            self.incomplete = None
        elif token_type == "meter":
            self.last_field_index = FIELD_TO_INDEX["rhythm"]
            self.incomplete = "rhythm_after_meter"
        elif token_type == "eighth_position":
            self.last_field_index = FIELD_TO_INDEX["rhythm"]
            self.incomplete = None
        elif token_type == "structure":
            self.last_field_index = FIELD_TO_INDEX["structure"]
            self.incomplete = None
        elif token_type == "key":
            self.last_field_index = FIELD_TO_INDEX["key"]
            self.incomplete = None
        elif token_type == "chord_full":
            self.last_field_index = FIELD_TO_INDEX["chord"]
            self.incomplete = None
        elif token_type == "pitch":
            self.last_field_index = FIELD_TO_INDEX["melody"]
            self.incomplete = "melody_after_pitch"
        elif token_type == "duration":
            self.last_field_index = FIELD_TO_INDEX["melody"]
            self.incomplete = None
        else:
            raise RuntimeError(f"Unexpected prompt token type {token_type!r}")
        return False


def inference_autocast(device, dtype=torch.bfloat16):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def select_cache_batch(cache, indices, batch_size):
    if cache is None:
        return None
    batch_select = getattr(cache, "batch_select_indices", None)
    if callable(batch_select):
        batch_select(indices)
        return cache
    if torch.is_tensor(cache):
        if cache.ndim > 0 and cache.shape[0] == batch_size:
            return cache.index_select(0, indices)
        return cache
    if isinstance(cache, tuple):
        return tuple(select_cache_batch(item, indices, batch_size) for item in cache)
    if isinstance(cache, list):
        return [select_cache_batch(item, indices, batch_size) for item in cache]
    if isinstance(cache, dict):
        return {
            key: select_cache_batch(value, indices, batch_size)
            for key, value in cache.items()
        }
    return cache


@torch.inference_mode()
def constrained_prompt_generate_batch(
    model,
    audio,
    prompts,
    max_sequence_length,
    prefix_tokens=None,
    autocast_dtype=torch.bfloat16,
    stop_time_seconds=None,
    progress_callback=None,
    memory=None,
    step_callback=None,
):
    tokenizer = model.tokenizer
    prefix = list(prefix_tokens) if prefix_tokens is not None else tokenizer.prompt_prefix(prompts)
    if not prefix or prefix[0] != tokenizer.sos_token:
        raise ValueError("generation prefix must begin with <|sos|>")
    if prefix[-1] == tokenizer.eos_token:
        prefix = prefix[:-1]
    if audio.ndim != 2 or audio.shape[0] < 1:
        raise ValueError(f"audio must have shape [batch, samples], got {tuple(audio.shape)}")
    states = [PromptGrammarState(tokenizer) for _ in range(audio.shape[0])]
    stop_times = [
        None if stop_time_seconds is None else float(stop_time_seconds)
        for _ in states
    ]
    out_index = prefix.index(tokenizer.out_token)
    for state in states:
        for token in prefix[out_index + 1 :]:
            state.update(token)
    output_tokens = [list(prefix) for _ in states]
    active_sample_ids = list(range(audio.shape[0]))
    decoder_input = torch.tensor(
        [prefix] * audio.shape[0],
        dtype=torch.long,
        device=audio.device,
    )
    past_key_values = None
    if memory is None:
        with inference_autocast(audio.device, autocast_dtype):
            memory = model.encode(audio)
    current_length = len(prefix)
    while active_sample_ids and current_length < int(max_sequence_length):
        with inference_autocast(audio.device, autocast_dtype):
            logits, past_key_values = model.decode(
                memory,
                decoder_input,
                use_cache=True,
                past_key_values=past_key_values,
            )
        next_logits = logits[:, -1].float()
        allowed = torch.stack(
            [state.allowed(next_logits.device) for state in states],
            dim=0,
        )
        masked_logits = next_logits.masked_fill(~allowed, float("-inf"))
        next_tokens = masked_logits.argmax(dim=-1)
        if step_callback is not None:
            step_callback(current_length, active_sample_ids, next_logits, masked_logits)
        next_token_values = next_tokens.tolist()
        keep_positions = []
        for position, (sample_id, state, stop_time, token) in enumerate(
            zip(active_sample_ids, states, stop_times, next_token_values)
        ):
            output_tokens[sample_id].append(int(token))
            finished = state.update(token)
            if (
                not finished
                and stop_time is not None
                and tokenizer.time_token_start <= int(token) < tokenizer.time_token_end
                and tokenizer.token_to_time_id(token) / tokenizer.time_hz >= stop_time
            ):
                output_tokens[sample_id].append(tokenizer.eos_token)
                finished = True
            if not finished:
                keep_positions.append(position)
        current_length += 1
        if progress_callback is not None:
            progress_callback(current_length)
        if not keep_positions:
            break

        old_batch_size = len(active_sample_ids)
        decoder_input = next_tokens[:, None]
        if len(keep_positions) == old_batch_size:
            continue

        keep = torch.tensor(keep_positions, dtype=torch.long, device=audio.device)
        active_sample_ids = [active_sample_ids[position] for position in keep_positions]
        states = [states[position] for position in keep_positions]
        stop_times = [stop_times[position] for position in keep_positions]
        memory = memory.index_select(0, keep)
        past_key_values = select_cache_batch(past_key_values, keep, old_batch_size)
        decoder_input = decoder_input.index_select(0, keep)

    for tokens in output_tokens:
        if tokens[-1] != tokenizer.eos_token:
            tokens.append(tokenizer.eos_token)
    return [torch.tensor(tokens, dtype=torch.long) for tokens in output_tokens]


@torch.inference_mode()
def constrained_prompt_generate(
    model,
    audio,
    prompts,
    max_sequence_length,
    prefix_tokens=None,
    autocast_dtype=torch.bfloat16,
    stop_time_seconds=None,
    progress_callback=None,
    memory=None,
    step_callback=None,
):
    if audio.shape[0] != 1:
        raise ValueError(
            "constrained_prompt_generate expects batch size 1; "
            "use constrained_prompt_generate_batch for batched inference"
        )
    return constrained_prompt_generate_batch(
        model,
        audio,
        prompts,
        max_sequence_length,
        prefix_tokens=prefix_tokens,
        autocast_dtype=autocast_dtype,
        stop_time_seconds=stop_time_seconds,
        progress_callback=progress_callback,
        memory=memory,
        step_callback=step_callback,
    )[0]


def decode_generated_tokens(tokenizer, tokens, song_id, window_index=None):
    try:
        return tokenizer.decode_sequence(tokens, strict=True), None
    except ValueError as exc:
        recoverable_errors = (
            "empty event at subbeat",
            "belongs to inactive output field",
        )
        if not any(message in str(exc) for message in recoverable_errors):
            raise
        decoded = tokenizer.decode_sequence(tokens, strict=False)
        warning = {
            "song_id": song_id,
            "warning": "strict decode failed; recovered in non-strict decode",
            "error": str(exc),
        }
        if window_index is not None:
            warning["window_index"] = int(window_index)
        return decoded, warning


def event_time_map(decoded, target_seconds):
    anchors = []
    for event in decoded["events"]:
        value = event["values"].get("timestamp")
        if value is not None:
            anchors.append((int(event["subbeat"]), float(value)))
    if not anchors:
        return lambda step: min(float(target_seconds), max(0.0, float(step) * 0.125))
    anchors = sorted(dict(anchors).items())
    steps = np.asarray([item[0] for item in anchors], dtype=np.float64)
    times = np.asarray([item[1] for item in anchors], dtype=np.float64)
    if len(anchors) >= 2:
        step_seconds = float(np.median(np.diff(times) / np.maximum(np.diff(steps), 1)))
        if not np.isfinite(step_seconds) or step_seconds <= 0:
            step_seconds = 0.125
    else:
        step_seconds = 0.125

    def lookup(step):
        step = float(step)
        if step <= steps[0]:
            return float(np.clip(times[0] + (step - steps[0]) * step_seconds, 0, target_seconds))
        if step >= steps[-1]:
            return float(np.clip(times[-1] + (step - steps[-1]) * step_seconds, 0, target_seconds))
        return float(np.interp(step, steps, times))

    return lookup


def field_text(event, tokenizer):
    parts = []
    for field in tokenizer.event_field_order:
        value = event["values"].get(field)
        if value is None:
            continue
        if field == "melody":
            note_parts = []
            for note in value:
                duration = note["duration_bin"]
                note_parts.append(
                    f"pitch={note['pitch']}:track={note['track']}:dur_bin={duration}:dur_steps={note['duration_steps']}"
                )
            parts.append("melody=[" + ",".join(note_parts) + "]")
        elif isinstance(value, dict):
            parts.append(field + "=" + ",".join(f"{k}:{v}" for k, v in value.items()))
        else:
            parts.append(f"{field}={value}")
    return "; ".join(parts)


def event_start_time(event, time_lookup):
    if "time" in event:
        return float(event["time"])
    return float(time_lookup(event["subbeat"]))


def write_events_tsv(decoded, tokenizer, time_lookup, path):
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("event_index\tsubbeat\ttime\tfields\ttokens\n")
        for index, event in enumerate(decoded["events"]):
            token_text = " ".join(
                tokenizer.describe(token)
                for field in tokenizer.event_field_order
                for token in event["tokens_by_field"].get(field, ())
            )
            f.write(
                f"{index}\t{event['subbeat']}\t{event_start_time(event, time_lookup):.6f}\t"
                f"{field_text(event, tokenizer)}\t{token_text}\n"
            )


def clone_decoded_event(event):
    return {
        "subbeat": int(event["subbeat"]),
        "tokens_by_field": {
            field: [int(token) for token in tokens]
            for field, tokens in event["tokens_by_field"].items()
        },
        "values": copy.deepcopy(event["values"]),
    }


def refresh_event_values(event, tokenizer, prompts):
    event["values"] = {
        field: tokenizer._decode_field(field, tokens, prompts)
        for field, tokens in event["tokens_by_field"].items()
        if tokens
    }


def set_event_local_timestamp(event, tokenizer, local_time):
    time_id = int(round(float(local_time) * tokenizer.time_hz))
    time_id = max(0, min(time_id, tokenizer.n_time_tokens - 1))
    event["tokens_by_field"]["timestamp"] = [tokenizer.time_id_to_token(time_id)]


def active_context_before(events, tokenizer, time_abs):
    state = {}
    for event in events:
        event_time = event.get("time")
        if event_time is None or float(event_time) > float(time_abs) + 1e-6:
            continue
        for field in ("structure", "key", "chord"):
            tokens = event["tokens_by_field"].get(field)
            if tokens:
                state[field] = [int(token) for token in tokens]
        rhythm_tokens = event["tokens_by_field"].get("rhythm", ())
        meter_tokens = [
            int(token)
            for token in rhythm_tokens
            if tokenizer.token_type(token) == "meter"
        ]
        if meter_tokens:
            state["meter"] = meter_tokens[:1]
    return state


def apply_prefix_context(event, context, tokenizer, prompts):
    for field in ("structure", "key", "chord"):
        if field not in event["tokens_by_field"] and field in context:
            event["tokens_by_field"][field] = list(context[field])

    rhythm_tokens = list(event["tokens_by_field"].get("rhythm", ()))
    has_meter = any(tokenizer.token_type(token) == "meter" for token in rhythm_tokens)
    has_eighth = any(
        tokenizer.token_type(token) == "eighth_position" for token in rhythm_tokens
    )
    if has_eighth and not has_meter and "meter" in context:
        event["tokens_by_field"]["rhythm"] = list(context["meter"]) + rhythm_tokens

    refresh_event_values(event, tokenizer, prompts)


def build_overlap_prefix_tokens(
    stitched_events,
    tokenizer,
    prompts,
    window_start,
    prefix_end,
):
    eps = 1e-4
    source_events = [
        event
        for event in stitched_events
        if float(window_start) - eps <= float(event.get("time", -1.0)) < float(prefix_end) - eps
    ]
    source_events.sort(
        key=lambda event: (
            int(
                event.get(
                    "global_subbeat",
                    event.get("source_subbeat", event["subbeat"]),
                )
            ),
            float(event.get("time", 0.0)),
        )
    )
    first_beat_index = next(
        (
            index
            for index, event in enumerate(source_events)
            if "timestamp" in event["values"] or "rhythm" in event["values"]
        ),
        None,
    )
    if first_beat_index is None:
        return None, None, None

    source_events = source_events[first_beat_index:]
    base_subbeat = int(
        source_events[0].get(
            "global_subbeat",
            source_events[0].get("source_subbeat", source_events[0]["subbeat"]),
        )
    )
    context = active_context_before(
        stitched_events,
        tokenizer,
        source_events[0]["time"],
    )
    prefix_events = []
    for source in source_events:
        event = clone_decoded_event(source)
        source_subbeat = int(
            source.get(
                "global_subbeat",
                source.get("source_subbeat", source["subbeat"]),
            )
        )
        event["subbeat"] = max(0, source_subbeat - base_subbeat)
        if "timestamp" in event["tokens_by_field"]:
            set_event_local_timestamp(
                event,
                tokenizer,
                float(source["time"]) - float(window_start),
            )
        refresh_event_values(event, tokenizer, prompts)
        prefix_events.append(event)

    apply_prefix_context(prefix_events[0], context, tokenizer, prompts)
    prefix_decoded = {
        "schema_version": tokenizer.schema_version,
        "prompts": prompts,
        "events": prefix_events,
        "has_eos": False,
    }
    return (
        prefix_decoded,
        tokenizer.encode_decoded_sequence(prefix_decoded),
        base_subbeat,
    )


def stitched_window_events(
    decoded,
    time_lookup,
    window_start,
    accept_start,
    accept_end,
    song_duration,
    window_index,
    global_subbeat_base=0,
):
    accepted = []
    eps = 1e-4
    for event in decoded["events"]:
        local_time = float(time_lookup(event["subbeat"]))
        abs_time = float(window_start) + local_time
        if abs_time < float(accept_start) - eps:
            continue
        if abs_time >= float(accept_end) - eps or abs_time >= float(song_duration) - eps:
            continue

        output = clone_decoded_event(event)
        output["time"] = float(np.clip(abs_time, 0.0, song_duration))
        output["window_index"] = int(window_index)
        output["window_start"] = float(window_start)
        output["source_subbeat"] = int(event["subbeat"])
        output["global_subbeat"] = int(global_subbeat_base) + int(event["subbeat"])
        if "timestamp" in output["values"]:
            output["values"]["timestamp"] = output["time"]

        notes = output["values"].get("melody")
        if notes is not None:
            fixed_notes = []
            for note in notes:
                fixed_note = copy.deepcopy(note)
                duration_steps = int(fixed_note["duration_steps"])
                local_end = float(time_lookup(int(event["subbeat"]) + duration_steps))
                end_time = float(window_start) + local_end
                end_time = min(
                    float(song_duration),
                    max(output["time"] + 0.04, end_time),
                )
                fixed_note["end_time"] = end_time
                fixed_notes.append(fixed_note)
            output["values"]["melody"] = fixed_notes
        accepted.append(output)
    return accepted


def write_window_tokens(path, windows, tokenizer):
    with Path(path).open("w", encoding="utf-8") as f:
        for window in windows:
            f.write(
                f"# window_index={window['window_index']} "
                f"start={window['start']:.6f} end={window['end']:.6f} "
                f"prefix_end={window['prefix_end']:.6f} "
                f"accept=[{window['accept_start']:.6f},{window['accept_end']:.6f}) "
                f"generation_stop={window['generation_stop']} "
                f"prefix_tokens={window['prefix_tokens']} "
                f"tokens={int(window['tokens'].numel())}\n"
            )
            for index, token in enumerate(window["tokens"].tolist()):
                f.write(f"{index}\t{token}\t{tokenizer.describe(token)}\n")
            f.write("\n")


@torch.inference_mode()
def generate(model, input_values, prompts=None, *, attention_mask=None, max_length=None,
             max_new_tokens=None, prefix_tokens=None,
             autocast_dtype=torch.bfloat16, stop_time_seconds=None,
             return_dict_in_generate=False, output_logits=False, output_scores=False,
             progress_callback=None):
    """Generate symbolic token sequences with the model's event grammar.

    Inputs are 24 kHz waveforms [batch, samples]. Scores/logits are per generated
    step, after the prompt prefix, with completed batch items filled by NaNs.
    """
    from transformers.utils import ModelOutput
    if input_values.ndim == 1:
        input_values = input_values.unsqueeze(0)
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    input_values = input_values.to(next(model.parameters()).device)
    prompts = model.tokenizer.normalize_prompts(prompts or FULL_TASK_PROMPTS)
    prefix = list(prefix_tokens) if prefix_tokens is not None else model.tokenizer.prompt_prefix(prompts)
    prefix_length = len(prefix) - (1 if prefix and prefix[-1] == model.tokenizer.eos_token else 0)
    if max_new_tokens is not None:
        if max_length is not None:
            raise ValueError("Specify max_new_tokens or max_length, not both.")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        max_length = prefix_length + max_new_tokens
    limit = model.max_output_seq_len if max_length is None else max_length
    if not isinstance(limit, int) or isinstance(limit, bool) or not prefix_length < limit <= model.max_output_seq_len:
        raise ValueError("The token limit must exceed the prefix length and fit the decoder context.")
    memory = None
    if attention_mask is not None:
        with inference_autocast(input_values.device, autocast_dtype):
            memory = model.get_audio_features(input_values, attention_mask=attention_mask).encoder_last_hidden_state
    raw, scores, positions = [], [], []
    batch_size = input_values.shape[0]
    def capture(position, ids, logits, masked):
        positions.append(position)
        for enabled, values, output in ((output_logits, logits, raw), (output_scores, masked, scores)):
            if enabled:
                row = torch.full((batch_size, values.shape[-1]), float('nan'), dtype=torch.float32)
                row[ids] = values.detach().cpu()
                output.append(row)
    tokens = constrained_prompt_generate_batch(
        model, input_values, prompts, limit,
        prefix_tokens=prefix_tokens, autocast_dtype=autocast_dtype,
        stop_time_seconds=stop_time_seconds, progress_callback=progress_callback,
        step_callback=capture if output_logits or output_scores else None,
        memory=memory,
    )
    sequences = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=model.tokenizer.pad_token)
    if return_dict_in_generate or output_logits or output_scores:
        return ModelOutput(sequences=sequences, logits=tuple(raw) if output_logits else None,
                           scores=tuple(scores) if output_scores else None,
                           token_positions=torch.tensor(positions, dtype=torch.long))
    return sequences
