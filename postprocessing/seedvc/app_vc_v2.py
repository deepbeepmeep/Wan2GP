import gradio as gr
import torch
import yaml

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16
def load_models(args):
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    from ._paths import resolve_path
    cfg = DictConfig(yaml.safe_load(open(resolve_path("configs/v2/vc_wrapper.yaml"), "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                cfm_checkpoint_path=args.cfm_checkpoint_path)
    vc_wrapper.to(device)
    vc_wrapper.eval()

    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True

        if hasattr(torch._inductor.config, "fx_graph_cache"):
            # Experimental feature to reduce compilation times, will be on by default in future
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()
        # vc_wrapper.compile_cfm()

    return vc_wrapper

def main(args):
    vc_wrapper = load_models(args)
    
    # Set up Gradio interface
    description = ("Zero-shot voice conversion with in-context learning. For local deployment please check [GitHub repository](https://github.com/Plachtaa/seed-vc) "
                   "for details and updates.<br>Note that any reference audio will be forcefully clipped to 25s if beyond this length.<br> "
                   "If total duration of source and reference audio exceeds 30s, source audio will be processed in chunks.<br> "
                   "无需训练的 zero-shot 语音/歌声转换模型，若需本地部署查看[GitHub页面](https://github.com/Plachtaa/seed-vc)<br>"
                   "请注意，参考音频若超过 25 秒，则会被自动裁剪至此长度。<br>若源音频和参考音频的总时长超过 30 秒，源音频将被分段处理。")
    
    inputs = [
        gr.Audio(type="filepath", label="Source Audio / 源音频"),
        gr.Audio(type="filepath", label="Reference Audio / 参考音频"),
        gr.Slider(minimum=1, maximum=200, value=30, step=1, label="Diffusion Steps / 扩散步数", 
                 info="30 by default, 50~100 for best quality / 默认为 30，50~100 为最佳质量"),
        gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Length Adjust / 长度调整", 
                 info="<1.0 for speed-up speech, >1.0 for slow-down speech / <1.0 加速语速，>1.0 减慢语速"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Intelligibility CFG Rate",
                 info="has subtle influence / 有微小影响"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Similarity CFG Rate",
                  info="has subtle influence / 有微小影响"),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.9, label="Top-p",
                 info="Controls diversity of generated audio / 控制生成音频的多样性"),
        gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Temperature",
                 info="Controls randomness of generated audio / 控制生成音频的随机性"),
        gr.Slider(minimum=1.0, maximum=3.0, step=0.1, value=1.0, label="Repetition Penalty",
                 info="Penalizes repetition in generated audio / 惩罚生成音频中的重复"),
        gr.Checkbox(label="convert style", value=False),
        gr.Checkbox(label="anonymization only", value=False),
    ]
    
    examples = [
        ["examples/source/yae_0.wav", "examples/reference/dingzhen_0.wav", 50, 1.0, 0.5, 0.5, 0.9, 1.0, 1.0, False, False],
        ["examples/source/jay_0.wav", "examples/reference/azuma_0.wav", 50, 1.0, 0.5, 0.5, 0.9, 1.0, 1.0, False, False],
    ]
    
    outputs = [
        gr.Audio(label="Stream Output Audio / 流式输出", streaming=True, format='mp3'),
        gr.Audio(label="Full Output Audio / 完整输出", streaming=False, format='wav')
    ]
    
    # Launch the Gradio interface
    gr.Interface(
        fn=vc_wrapper.convert_voice_with_streaming,
        description=description,
        inputs=inputs,
        outputs=outputs,
        title="Seed Voice Conversion V2",
        examples=examples,
        cache_examples=False,
    ).launch()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile")
    # V2 custom checkpoints
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    args = parser.parse_args()
    main(args)