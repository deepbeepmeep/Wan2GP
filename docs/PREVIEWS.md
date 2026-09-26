# Generation previews

The Preview panel shows an approximate view of the image or video while generation runs. Video previews show a strip of up to four frames.

In **Configuration > General**, **TinyVAE Preview (when available)** offers:

- **Disabled** (default): keeps the existing RGB latent preview and downloads no TinyVAE checkpoint.
- **GPU Mode**: replaces RGB previews with a small neural decoder for supported architectures. The decoder downloads automatically on first use. Unsupported architectures retain RGB previews.

TinyVAE previews can show more recognizable detail as denoising progresses. They are approximate and do not change the final image or video. GPU Mode consumes additional VRAM and decoding time; use Disabled when memory is tight or generation speed matters more than preview detail. The first steps can still look noisy.

Available decoder families include baseline Wan 2.1/2.2, Hunyuan Video/1.5, LTX-2/2.3/2.5, MiniMax H3, Flux, Flux 2 Klein, Z-Image, Qwen Image 20B, Krea 2 and Ideogram 4. Availability depends on the exact architecture; it does not extend automatically to every derivative. Qwen Image 2.1, LTX MSR, Edit Anything and JoyAI Echo currently keep their existing preview.

Saving a different preview mode takes effect on the next generation. The tiny decoder shares GPU residency with generation components, and the usual model-unload actions release it. GPU Mode requires an MMGP build with wildcard cotenant support.

---

> Applies to: Live image and video generation previews, TinyVAE configuration, availability, and memory/performance tradeoffs.
