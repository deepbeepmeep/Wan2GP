def query_speaker_separator_download_def():
    return [
        {
            "repoId": "DeepBeepMeep/Wan2.1",
            "sourceFolderList": ["pyannote"],
            "fileList": [["pyannote_model_wespeaker-voxceleb-resnet34-LM.bin", "pytorch_model_segmentation-3.0.bin"]],
        },
        {
            "repoId": "DeepBeepMeep/LTX-2",
            "sourceFolderList": ["sherpa"],
            "fileList": [["sherpa-onnx-pyannote-segmentation-3-0/model.onnx", "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"]],
        },
    ]


def download_speaker_separator(send_cmd=None, status_text="Downloading speaker separator model files..."):
    from shared.utils.download import process_files_def_if_needed

    return process_files_def_if_needed(query_speaker_separator_download_def(), send_cmd=send_cmd, status_text=status_text)
