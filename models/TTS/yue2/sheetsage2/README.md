# SheetSage2 in WanGP

Local transcription implementation from `m-a-p/SheetSage2` revision
`eab522a8168e8b8b8c4856bf8609cd86198f01fe`, with MERT-v2-FullSong revision
`d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`.

Sources: https://huggingface.co/m-a-p/SheetSage2 and
https://huggingface.co/m-a-p/MERT-v2-FullSong. See the accompanying licenses and
third-party notices. No browser renderer or piano soundfont assets are bundled.

WanGP loads the merged model through MMGP only while extracting a source song's
score, then releases it. The six-layer BART transcription decoder uses PyTorch
SDPA and its encoder/decoder KV cache. YuE2's subsequent generation continues to
use the configured legacy, CG or vLLM engine. MERT attention uses shared WanGP
attention with SDPA. Progress and cancellation are attached temporarily to the
encoder layers and decoder, and the upstream overlapping-window transcription
and ABC export are preserved. The current Transformers cache API replaces the
deprecated tuple cache. File decoding uses WanGP's FFmpeg binary.

## Reproducing the checkpoint

Download each pinned repository's `model.safetensors` into
`<checkpoint_root>/sheetsage2/_source/<repository_name>/`, alongside a
`revision.txt` containing its revision above. Run from the WanGP root:

```text
python -m models.TTS.yue2.sheetsage2.convert_weights <checkpoint_root>
```

The converter verifies the parent checksum, merges attention adapters in FP32,
then stores BF16 weights with MMGP shared-embedding metadata. It verifies every
saved tensor. The four MERT audio frontend buffers retain their upstream FP32
precision: log-mel normalization statistics, mel filters and STFT window.
After a successful load and transcription check, delete both source checkpoints.

Output: `sheetsage2/SheetSage2_MERT2_bf16.safetensors` (1,354,429,922 bytes).
The distribution URL is under `DeepBeepMeep/TTS/sheetsage2/`; local preparation
does not upload it automatically.
