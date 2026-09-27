# Generation previews

The Preview panel shows an approximate view of the image or video while generation runs. Choose a strip of up to four sampled frames or a looping video to see how motion is developing.

In **Configuration > General**, **Generation Preview** offers:

- **Frames Selection using RGB Factors (fast)** (default): shows the existing approximate frame strip and downloads no TinyVAE checkpoint.
- **Frames Selection using Tiny VAE (when available, slower)**: shows a more detailed frame strip using a small GPU decoder.
- **Video using Tiny VAE (when available, even slower)**: shows a muted, looping video preview. Click the video to pause it, then click again to resume. It stays paused when a new preview arrives. Image generations still show images.

The Tiny VAE modes download a small decoder automatically on first use. Unsupported architectures retain RGB frame previews. Both modes use additional GPU memory and decoding time; video also needs more frame transfers and CPU encoding. Choose RGB when memory is tight or generation speed matters more than preview detail.

Previews update periodically during denoising and do not change the final image or video. The first steps can still look noisy. Video previews are small and silent, with reduced frame rate; they show the current generation window rather than joining all windows of a long video. On long clips, fewer frames are sampled to keep preview memory bounded.

Available decoder families include baseline Wan 2.1/2.2, Hunyuan Video/1.5, LTX-2/2.3/2.5, MiniMax H3, Flux, Flux 2 Klein, Z-Image, Qwen Image 20B, Krea 2 and Ideogram 4. Availability depends on the exact architecture; it does not extend automatically to every derivative. Qwen Image 2.1, LTX MSR, Edit Anything and JoyAI Echo currently keep their existing preview.

Saving a different preview mode takes effect on the next generation. The tiny decoder shares GPU residency with generation components, and the usual model-unload actions release it. Tiny VAE modes require an MMGP build with wildcard cotenant support.

---

> Applies to: Live image and video generation previews, TinyVAE configuration, availability, and memory/performance tradeoffs.
