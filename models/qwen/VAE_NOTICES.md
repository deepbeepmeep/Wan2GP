# Alternative Qwen Image VAE checkpoints

These community checkpoints are optional alternatives for RGB Qwen Image latent-space models. They are not official Krea AI VAE releases. WanGP preserves their original FP32 tensors and uses the existing Wan/Qwen key converter at load time. The published files are byte-identical to the sources below.

## Krea 2 Real

- Original author: suede2031691, [Krea 2 real vae on Civitai](https://civitai.com/models/2729562/krea-2-real-vae).
- Download mirror: [artsyww/KREA2REALVAE](https://huggingface.co/artsyww/KREA2REALVAE), revision `55a3e8648f2eba2d9b4890b1c9ccaf187b630468`.
- Original filename: `krea2RealVae_v10.safetensors`; WanGP filename: `krea2_real_vae_fp32.safetensors`.
- SHA-256: `0dbbe0baeca04c2b98d2f3809c6f595608939809c88b695ba971368f17c874b8`.
- The author describes adapting the decoder from [spacepxl/Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x), averaging the output head to restore ordinary-resolution RGB output and combining it with the original VAE. Spacepxl publishes its model under Apache-2.0. Preserve upstream terms and attribution; the mirror itself supplies no model card or separate license.

## Krea 2 HD

- Author and source: [wikeeyang/Krea2-Turbo-HD-V1](https://huggingface.co/wikeeyang/Krea2-Turbo-HD-V1), revision `4ebaa9abc4145ad1176099e5d7f8d491fc016e34`.
- Original filename: `Krea2-HD-vae.safetensors`; WanGP filename: `krea2_hd_vae_fp32.safetensors`.
- SHA-256: `3401bbf1953cdc920c3d20ec178cb0368f5527a96cb78abe30904b27a82361f2`.
- The source repository declares the Krea 2 Community License and supplies `LICENSE.pdf`. Its original license is distributed beside the mirrored checkpoint as `krea2_hd_vae_LICENSE.pdf`.

Both alternatives contain 194 FP32 tensors and retain the standard 16-channel, 8× spatial-compression RGB architecture. They can change texture, colors and contrast. They are not replacements for the RGBA VAEs used by Qwen Layered or Ming Image, or the 64-channel Qwen Image 2.1 VAE. Selecting the Spacepxl 2× VAE upsampler takes priority over the selected alternative.

Reproduce preparation with `tools/prepare_qwen_vae_variants.py`; the generated `vae_provenance.json` records the source revisions, hashes, sizes, precision and metadata. The checkpoint key conversion does not change tensor values or dtypes.
