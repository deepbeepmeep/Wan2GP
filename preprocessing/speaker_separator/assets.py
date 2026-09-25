def query_speaker_separator_download_def(use_sherpa=None):
    definitions = [
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
    return definitions if use_sherpa is None else [definitions[1 if use_sherpa else 0]]


def download_speaker_separator(send_cmd=None, status_text="Downloading Speaker Separator Model Files", gen=None, process_files=None, use_sherpa=None):
    from shared.utils.download import process_files_def_if_needed

    if use_sherpa is None:
        from .separator import USE_SHERPA_ONNX_SPEAKER_DIARIZATION
        use_sherpa = USE_SHERPA_ONNX_SPEAKER_DIARIZATION
    return process_files_def_if_needed(query_speaker_separator_download_def(use_sherpa), send_cmd=send_cmd, status_text=status_text, gen=gen, process_files=process_files)
