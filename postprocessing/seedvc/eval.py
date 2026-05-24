import shutil
import warnings
import argparse
import torch
import os
import os.path as osp
import yaml

warnings.simplefilter("ignore")

# load packages
import random

from tqdm import tqdm
from .modules.commons import *
import time

import torchaudio
import librosa
import torchaudio.compliance.kaldi as kaldi

from .hf_utils import load_custom_model_from_hf
from resemblyzer import preprocess_wav, VoiceEncoder

# Load model and configuration

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from transformers import Wav2Vec2Processor, HubertForCTC

import jiwer
import string

from .baselines.dnsmos.dnsmos_computor import DNSMOSComputer

def calc_mos(computor, audio, orin_sr):
    # only 16k audio is supported
    target_sr = 16000
    if orin_sr != 16000:
        audio = librosa.resample(
            audio, orig_sr=orin_sr, target_sr=target_sr, res_type="kaiser_fast"
        )
    result = computor.compute(audio, target_sr, False)
    sig, bak, ovr = result["SIG"], result["BAK"], result["OVRL"]

    if ovr == 0:
        print("calculate dns mos failed")
    return sig, bak, ovr

mos_computer = DNSMOSComputer(
    "baselines/dnsmos/sig_bak_ovr.onnx",
    "baselines/dnsmos/model_v8.onnx",
    device="cuda",
    device_id=0,
)

def load_models(args):
    dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                     "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                                                                     "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from .modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type

    if vocoder_type == 'bigvgan':
        from .modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == 'hifigan':
        from .modules.hifigan.generator import HiFTGenerator
        from .modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_gen.load_state_dict(torch.load(hift_config['pretrained_model_path'], map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config['model_params'])
        vocos = build_model(vocos_model_params, stage='mel_vocos')
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path,
                                         load_only_params=True, ignore_modules=[], is_distributed=False)
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(sum(p.numel() for p in vocos[key].parameters() if p.requires_grad) for key in vocos.keys())
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unsupported vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    elif speech_tokenizer_type == 'cnhubert':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            HubertModel,
        )
        hubert_model_name = config['model_params']['speech_tokenizer']['name']
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                  return_tensors="pt",
                                                  return_attention_mask=True,
                                                  padding=True,
                                                  sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    elif speech_tokenizer_type == 'xlsr':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   padding=True,
                                                   sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    else:
        raise ValueError(f"Unsupported speech tokenizer type: {model_params.speech_tokenizer.type}")
    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from .modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )


@torch.no_grad()
def main(args):
    # init xvector models
    if args.xvector_extractor == "wavlm":
        wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        wavlm_model = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        ).to(device)
    elif args.xvector_extractor == "resemblyzer":
        resemblyzer_encoder = VoiceEncoder()
    elif args.xvector_extractor == 'wavlm-large':
        import sys
        sys.path.append("../UniSpeech/downstreams/speaker_verification")
        from verification import init_model
        wavlm_model = init_model("wavlm_large", "D:/wavlm_large_finetune.pth")
        wavlm_model.cuda()
        wavlm_model.eval()
    else:
        raise ValueError(f"Unknown xvector extractor: {args.xvector_extractor}")

    # init asr model
    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    asr_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)

    (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    ) = load_models(args)
    sr = mel_fn_args["sampling_rate"]

    source_dir = args.source
    target_dir = args.target
    diffusion_steps = args.diffusion_steps
    length_adjust = args.length_adjust
    inference_cfg_rate = args.inference_cfg_rate
    baseline = args.baseline
    max_samples = args.max_samples
    try:
        source_audio_list = open(osp.join(source_dir, "index.tsv"), "r").readlines()
    except FileNotFoundError:
        source_audio_list = os.listdir(source_dir)
        source_audio_list = [f for f in source_audio_list if f.endswith(".wav")]
    target_audio_list = os.listdir(target_dir)

    conversion_result_dir = args.output
    if baseline:
        conversion_result_dir = os.path.join(conversion_result_dir, baseline)
    os.makedirs(conversion_result_dir, exist_ok=True)

    similarity_list = []
    gt_wer_list = []
    gt_cer_list = []
    vc_wer_list = []
    vc_cer_list = []
    dnsmos_list = []
    for source_i, source_line in enumerate(tqdm(source_audio_list)):
        if source_i >= max_samples:
            break
        source_index, source_transcript = source_line.strip().split("\t")
        source_path = osp.join(source_dir, f"{source_index}.wav")
        for target_i, target_name in enumerate(target_audio_list):
            target_path = osp.join(target_dir, target_name)
            print(f"Processing {source_path} -> {target_path}")

            if os.path.exists(osp.join(conversion_result_dir, source_index, f"{target_name}")):
                # already converted, load the converted file
                vc_wave_16k, _ = librosa.load(
                    osp.join(conversion_result_dir, source_index, f"{target_name}"), sr=16000
                )
                vc_wave_16k = torch.tensor(vc_wave_16k).unsqueeze(0)
                ref_waves_16k, _ = librosa.load(target_path, sr=16000)
                ref_waves_16k = torch.tensor(ref_waves_16k).unsqueeze(0)
            else:
                if baseline == "openvoice":
                    from .baselines.openvoice import convert as openvoice_convert
                    ref_waves_16k, vc_wave_16k = openvoice_convert(source_path, target_path, "temp.wav")
                elif baseline == "cosyvoice":
                    from .baselines.cosyvoice import convert as cosyvoice_convert
                    ref_waves_16k, vc_wave_16k = cosyvoice_convert(source_path, target_path, "temp.wav")
                else:
                    ref_waves_16k, vc_wave = convert(
                        source_path,
                        target_path,
                        model,
                        semantic_fn,
                        vocoder_fn,
                        campplus_model,
                        to_mel,
                        mel_fn_args,
                        sr,
                        length_adjust,
                        diffusion_steps,
                        inference_cfg_rate,
                        remove_prompt=args.remove_prompt,
                    )
                    vc_wave_16k = torchaudio.functional.resample(vc_wave, sr, 16000)
                os.makedirs(osp.join(conversion_result_dir, source_index), exist_ok=True)
                torchaudio.save(
                    osp.join(conversion_result_dir, source_index, f"{target_name}"),
                    vc_wave_16k.cpu(),
                    16000,
                )
            if args.xvector_extractor == "wavlm":
                ref_inputs = wavlm_feature_extractor(
                    ref_waves_16k.squeeze(0).cpu(), padding=True, return_tensors="pt"
                ).to(device)
                ref_embeddings = wavlm_model(**ref_inputs).embeddings
                ref_embeddings = torch.nn.functional.normalize(ref_embeddings, dim=-1).cpu()

                vc_inputs = wavlm_feature_extractor(
                    vc_wave_16k.squeeze(0).cpu(), padding=True, return_tensors="pt"
                ).to(device)
                vc_embeddings = wavlm_model(**vc_inputs).embeddings
                vc_embeddings = torch.nn.functional.normalize(vc_embeddings, dim=-1).cpu()

                similarity = torch.nn.functional.cosine_similarity(
                    ref_embeddings, vc_embeddings, dim=-1
                )
            elif args.xvector_extractor == "resemblyzer":
                ref_wav_resemblyzer = preprocess_wav(target_path)
                vc_wav_resemblyzer = preprocess_wav(
                    osp.join(conversion_result_dir, source_index, f"{target_name}")
                )
                ref_embed = resemblyzer_encoder.embed_utterance(ref_wav_resemblyzer)
                vc_embed = resemblyzer_encoder.embed_utterance(vc_wav_resemblyzer)
                similarity = np.inner(ref_embed, vc_embed)
            elif args.xvector_extractor == 'wavlm-large':
                ref_embed = wavlm_model(ref_waves_16k.to(device)).cpu()
                vc_embed = wavlm_model(vc_wave_16k.to(device)).cpu()
                similarity = torch.nn.functional.cosine_similarity(ref_embed, vc_embed, dim=-1)
            else:
                raise ValueError(f"Unknown xvector extractor: {args.xvector_extractor}")
            print(f"Similarity: {similarity}")
            similarity_list.append(similarity)

            # perform asr
            vc_asr_inputs = asr_processor(
                vc_wave_16k.squeeze(0).cpu(), return_tensors="pt", padding=True
            ).to(device)
            vc_asr_logits = asr_model(**vc_asr_inputs).logits
            predicted_ids = torch.argmax(vc_asr_logits, dim=-1)
            vc_transcription = asr_processor.decode(predicted_ids[0])

            # perform asr on source 16k
            source_wav_16k = librosa.load(source_path, sr=16000)[0]
            source_asr_inputs = asr_processor(
                source_wav_16k, return_tensors="pt", padding=True
            ).to(device)
            source_asr_logits = asr_model(**source_asr_inputs).logits
            source_predicted_ids = torch.argmax(source_asr_logits, dim=-1)
            source_transcription = asr_processor.decode(source_predicted_ids[0])

            # convert transcriptions to all lower to calculate WER and CER
            source_transcript = source_transcript.lower()
            # remove punctuations in source_transcript
            source_transcript = source_transcript.translate(str.maketrans("", "", string.punctuation))
            source_transcription = source_transcription.lower()
            vc_transcription = vc_transcription.lower()

            # calculate WER and CER
            gt_wer = jiwer.wer(source_transcript, source_transcription)
            gt_cer = jiwer.cer(source_transcript, source_transcription)
            vc_wer = jiwer.wer(source_transcript, vc_transcription)
            vc_cer = jiwer.cer(source_transcript, vc_transcription)

            print(f"GT WER: {gt_wer}, CER: {gt_cer}")
            print(f"VC WER: {vc_wer}, CER: {vc_cer}")
            gt_wer_list.append(gt_wer)
            gt_cer_list.append(gt_cer)
            vc_wer_list.append(vc_wer)
            vc_cer_list.append(vc_cer)

            # calculate dnsmos
            sig, bak, ovr = calc_mos(mos_computer, vc_wave_16k.squeeze(0).cpu().numpy(), 16000)
            dnsmos_list.append((sig, bak, ovr))

        print(f"Average GT WER: {sum(gt_wer_list) / len(gt_wer_list)}")
        print(f"Average GT CER: {sum(gt_cer_list) / len(gt_cer_list)}")
        print(f"Average VC WER: {sum(vc_wer_list) / len(vc_wer_list)}")
        print(f"Average VC CER: {sum(vc_cer_list) / len(vc_cer_list)}")
        print(f"Average similarity: {sum(similarity_list) / len(similarity_list)}")

        print(f"Average DNS MOS SIG: {sum([x[0] for x in dnsmos_list]) / len(dnsmos_list)}")
        print(f"Average DNS MOS BAK: {sum([x[1] for x in dnsmos_list]) / len(dnsmos_list)}")
        print(f"Average DNS MOS OVR: {sum([x[2] for x in dnsmos_list]) / len(dnsmos_list)}")

        # save wer and cer result into this directory as a txt
        with open(osp.join(conversion_result_dir, source_index, "result.txt"), 'w') as f:
            f.write(f"GT WER: {sum(gt_wer_list[-len(target_audio_list):]) / len(target_audio_list)}\n")
            f.write(f"GT CER: {sum(gt_cer_list[-len(target_audio_list):]) / len(target_audio_list)}\n")
            f.write(f"VC WER: {sum(vc_wer_list[-len(target_audio_list):]) / len(target_audio_list)}\n")
            f.write(f"VC CER: {sum(vc_cer_list[-len(target_audio_list):]) / len(target_audio_list)}\n")
            f.write(f"Average similarity: {sum(similarity_list[-len(target_audio_list):]) / len(target_audio_list)}\n")

    print(f"Average WER: {sum(gt_wer_list) / len(gt_wer_list)}")
    print(f"Average CER: {sum(gt_cer_list) / len(gt_cer_list)}")
    print(f"Average WER: {sum(vc_wer_list) / len(vc_wer_list)}")
    print(f"Average CER: {sum(vc_cer_list) / len(vc_cer_list)}")
    print(f"Average similarity: {sum(similarity_list) / len(similarity_list)}")
    # save similarity list
    with open(osp.join(conversion_result_dir, f"{args.xvector_extractor}_similarity.tsv"), "w") as f:
        f.write("\n".join([str(s) for s in similarity_list]))
    # save wer and cer result into this directory as a txt
    with open(osp.join(conversion_result_dir, "result.txt"), 'w') as f:
        f.write(f"GT WER: {sum(gt_wer_list) / len(gt_wer_list)}\n")
        f.write(f"GT CER: {sum(gt_cer_list) / len(gt_cer_list)}\n")
        f.write(f"VC WER: {sum(vc_wer_list) / len(vc_wer_list)}\n")
        f.write(f"VC CER: {sum(vc_cer_list) / len(vc_cer_list)}\n")

    print(f"Average DNS MOS SIG: {sum([x[0] for x in dnsmos_list]) / len(dnsmos_list)}")
    print(f"Average DNS MOS BAK: {sum([x[1] for x in dnsmos_list]) / len(dnsmos_list)}")
    print(f"Average DNS MOS OVR: {sum([x[2] for x in dnsmos_list]) / len(dnsmos_list)}")


def convert(
    source_path,
    target_path,
    model,
    semantic_fn,
    vocoder_fn,
    campplus_model,
    to_mel,
    mel_fn_args,
    sr,
    length_adjust,
    diffusion_steps,
    inference_cfg_rate,
    remove_prompt=False,
):
    source_audio = librosa.load(source_path, sr=sr)[0]
    ref_audio = librosa.load(target_path, sr=sr)[0]
    # decoded_wav = encodec_model.decoder(encodec_latent)
    # torchaudio.save("test.wav", decoded_wav.cpu().squeeze(0), 24000)
    # crop only the first 30 seconds
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(device)

    if source_audio.size(1) + ref_audio.size(1) > 30 * sr:
        print(f"reference audio clipped from {ref_audio.size(1)/sr} seconds to {30 * sr - source_audio.size(1)} seconds")
        ref_audio = ref_audio[:, :30 * sr - source_audio.size(1)]


    source_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)

    S_alt = semantic_fn(source_waves_16k)
    S_ori = semantic_fn(ref_waves_16k)

    mel = to_mel(source_audio.to(device).float())
    mel2 = to_mel(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))
    # Length regulation
    cond = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=None
    )[0]
    prompt_condition = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
    )[0]
    if remove_prompt:
        cat_condition = cond
        mel2 = torch.zeros([mel2.size(0), mel2.size(1), 0]).to(mel2.device)
    else:
        cat_condition = torch.cat([prompt_condition, cond], dim=1)

    vc_target = model.cfm.inference(
        cat_condition,
        torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
        mel2,
        style2,
        None,
        diffusion_steps,
        inference_cfg_rate=inference_cfg_rate,
    )
    vc_target = vc_target[:, :, mel2.size(-1) :]

    # Convert to waveform
    vc_wave = vocoder_fn(vc_target).squeeze(1)

    return ref_waves_16k, vc_wave


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, default="./examples/libritts-test-clean/"
    )
    parser.add_argument("--target", type=str, default="./examples/reference/")
    parser.add_argument("--output", type=str, default="./examples/eval/converted/")
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument(
        "--xvector-extractor", type=str, default="wavlm-large"
    )  # wavlm or resemblyzer
    parser.add_argument("--baseline", type=str, default="") # use "" for Seed-VC
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--remove-prompt", type=bool, default=False)
    args = parser.parse_args()
    main(args)
