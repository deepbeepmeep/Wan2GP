import os
import torch
import sys
import librosa
sys.path.append('../CosyVoice')
import sys
sys.path.append("../CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
# from modelscope import snapshot_download
# snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='pretrained_models/CosyVoice-300M-25Hz')
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def convert(source_path, reference_path, output_path):
    prompt_speech_16k = load_wav(reference_path, 16000)
    source_speech_16k = load_wav(source_path, 16000)

    for i in cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False):
        output_wav_22k = i['tts_speech']
    output_wav_16k = torchaudio.functional.resample(output_wav_22k, 22050, 16000)
    return prompt_speech_16k, output_wav_16k