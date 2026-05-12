import os
import torch
import sys
import librosa
sys.path.append('../OpenVoice')
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

ckpt_converter = '../OpenVoice/checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

def convert(source_path, reference_path, output_path):
    target_se, audio_name = se_extractor.get_se(reference_path, tone_color_converter, vad=False)
    source_se, audio_name = se_extractor.get_se(source_path, tone_color_converter, vad=False)

    tone_color_converter.convert(
                audio_src_path=source_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path,
                message="@Myshell",)
    ref_wav_16k, _ = librosa.load(reference_path, sr=16000)
    output_wav_16k, _ = librosa.load(output_path, sr=16000)
    ref_wav_16k = torch.tensor(ref_wav_16k).unsqueeze(0)
    output_wav_16k = torch.tensor(output_wav_16k).unsqueeze(0)
    return ref_wav_16k, output_wav_16k