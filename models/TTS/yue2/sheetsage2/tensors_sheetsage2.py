"""Aligned audio features and token predictions, in memory or on disk."""
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .generation_sheetsage2 import inference_autocast


class WindowTensorWriter:
    def __init__(self, output_dir, index, window, logits, scores):
        self.directory = Path(output_dir) / "tensors" / f"window-{index:04d}" if output_dir is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.window = window
        self.logits, self.scores = logits, scores
        self.pending, self.files = [], []
        self.data = {"window_index": index, "token_batches": [], "decoder_batches": []} if self.directory is None else None

    def _store(self, name, payload, group):
        if self.directory is not None:
            save_file(payload, str(self.directory / name))
            self.files.append(name)
        elif group == "audio":
            self.data["audio"] = payload
        else:
            self.data[group].append(payload)

    def audio_features(self, model, audio, dtype, hidden_states):
        with inference_autocast(audio.device, dtype):
            features = model.get_audio_features(audio, output_hidden_states=hidden_states)
        payload = {}
        for name, value in features.items():
            if torch.is_tensor(value):
                payload[name] = value.detach().cpu().contiguous().clone()
            elif isinstance(value, tuple):
                for index, layer in enumerate(value):
                    if torch.is_tensor(layer):
                        payload[f"{name}.{index}"] = layer.detach().cpu().contiguous().clone()
        memory = features.encoder_last_hidden_state
        frames = memory.shape[1]
        valid_seconds = self.window["end"] - self.window["start"]
        payload["frame_times"] = torch.arange(frames, dtype=torch.float64) / 25 + self.window["start"]
        payload["feature_attention_mask"] = (torch.arange(frames) / 25 < valid_seconds)[None]
        self._store("audio.safetensors", payload, "audio")
        return memory

    def capture(self, position, ids, logits, masked):
        # Transcription processes one window at a time; keep at most 32 rows.
        row = {"token_position": torch.tensor(position, dtype=torch.long)}
        if self.logits:
            row["logits"] = logits[0].detach().cpu().contiguous()
        if self.scores:
            row["scores"] = masked[0].detach().cpu().contiguous()
        self.pending.append(row)
        if len(self.pending) == 32:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        first = int(self.pending[0]["token_position"])
        name = f"tokens-{first:04d}.safetensors"
        self._store(name, {key: torch.stack([row[key] for row in self.pending]) for key in self.pending[0]},
                    "token_batches")
        self.pending.clear()

    def decoder_features(self, model, audio, tokens, dtype, memory=None):
        with inference_autocast(audio.device, dtype):
            if memory is None:
                memory = model.encode(audio)
            cache = None
            for start in range(0, len(tokens) - 1, 64):
                ids = tokens[start:min(start + 64, len(tokens) - 1)][None].to(audio.device)
                output = model.decoder(
                    input_ids=ids,
                    attention_mask=ids.ne(model.tokenizer.pad_token) if cache is None else None,
                    encoder_hidden_states=memory, encoder_attention_mask=None,
                    past_key_values=cache, use_cache=True, return_dict=True,
                )
                cache = output.past_key_values
                name = f"decoder-{start:04d}.safetensors"
                self._store(name, {"decoder_hidden_state": output.last_hidden_state.detach().cpu().contiguous(),
                           "input_token_ids": ids.cpu(),
                           "token_positions": torch.arange(start, start + ids.shape[1])},
                            "decoder_batches")

    def finish(self):
        self.flush()
        metadata = dict(self.window, frame_rate=25)
        if self.directory is not None:
            metadata["files"] = self.files
        metadata["token_position_convention"] = "logits predict the token at token_position; decoder embeddings represent input_token_ids"
        if self.directory is not None:
            (self.directory / "index.json").write_text(json.dumps(metadata, indent=2) + "\n")
        else:
            self.data["metadata"] = metadata
        return self.data
