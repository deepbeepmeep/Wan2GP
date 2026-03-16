"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

import os
import click

@click.command()
@click.option(
    "--checkpoint_path",
    type=str,
    default="",
    help="Path to the checkpoint directory. Downloads automatically if empty.",
)
@click.option(
    "--server_name",
    type=str,
    default="127.0.0.1",
    help="The server name to use for the Gradio app.",
)
@click.option(
    "--port", type=int, default=7865, help="The port to use for the Gradio app."
)
@click.option("--device_id", type=int, default=0, help="The CUDA device ID to use.")
@click.option(
    "--share",
    type=click.BOOL,
    default=False,
    help="Whether to create a public, shareable link for the Gradio app.",
)
@click.option(
    "--bf16",
    type=click.BOOL,
    default=True,
    help="Whether to use bfloat16 precision. Turn off if using MPS.",
)
@click.option(
    "--torch_compile", type=click.BOOL, default=False, help="Whether to use torch.compile."
)
@click.option(
    "--cpu_offload", type=bool, default=False, help="Whether to use CPU offloading (only load current stage's model to GPU)"
)
@click.option(
    "--overlapped_decode", type=bool, default=False, help="Whether to use overlapped decoding (run dcae and vocoder using sliding windows)"
)
def main(checkpoint_path, server_name, port, device_id, share, bf16, torch_compile, cpu_offload, overlapped_decode):
    """
    Main function to launch the ACE Step pipeline demo.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    from .ui.components import create_main_demo_ui
    from .pipeline_ace_step import ACEStepPipeline
    from .data_sampler import DataSampler

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode
    )
    data_sampler = DataSampler()

    demo = create_main_demo_ui(
        text2music_process_func=model_demo.__call__,
        sample_data_func=data_sampler.sample,
        load_data_func=data_sampler.load_json,
    )
    demo.launch(server_name=server_name, server_port=port, share=share)


if __name__ == "__main__":
    main()
