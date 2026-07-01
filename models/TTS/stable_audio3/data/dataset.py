import numpy as np
import json
import os
import dill
import random
import time
import torch
import torchaudio

from os import path
from torchaudio import transforms as T
from typing import Optional, Callable, List

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T, VolumeNorm, strip_trailing_silence

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus'],
    filelist_path=None
):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames

        if filelist_path is None:
            # Check for filelist.txt at the root of the directory
            filelist_path = os.path.join(path, "filelist.txt")
            
        if os.path.exists(filelist_path):
            with open(filelist_path, "r") as f:
                files = f.readlines()
                files = [os.path.join(path, file.strip()) for file in files]
                filenames.extend(files)
            continue

        if keywords is not None:
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames

def get_latent_filenames(
    paths,  # directories in which to search
    extension='npy',
    filelist_path=None
):
    "recursively get a list of pre-encoded filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames

        if filelist_path is None:
            # Check for filelist.txt at the root of the directory
            filelist_path = os.path.join(path, "filelist.txt")
        
        if os.path.exists(filelist_path):
            with open(filelist_path, "r") as f:
                files = f.readlines()
                files = [os.path.join(path, file.strip()) for file in files]
                filenames.extend(files)
            continue

        _, files = fast_scandir(path, [extension])
        filenames.extend(files)

    # Filter out silence.npy (used for silence latent padding, not a data sample)
    filenames = [f for f in filenames if os.path.basename(f) != "silence.npy"]

    # Add metadata paths
    filenames = [(filename, filename.replace(f".{extension}", ".json")) for filename in filenames]

    return filenames

class LocalDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        keywords: Optional[List[str]]=None,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        filelist_path = None,
        weight: float = 1.0,
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.keywords = keywords
        self.filelist_path = filelist_path
        self.weight = weight

class LatentDatasetConfig(LocalDatasetConfig):
    def __init__(
        self,
        latent_extension: str = "npy",
        filelist_path = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.latent_extension = latent_extension
        self.filelist_path = filelist_path
        # weight is inherited from LocalDatasetConfig via **kwargs

class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs,
        sample_size=65536,
        sample_rate=48000,
        random_crop=True,
        force_channels="stereo",
        volume_norm=False,
        volume_norm_param=(-16, 2),
        strip_silence=False,
        pad=True,
    ):
        super().__init__()
        self.filenames = []
        self.sample_weights = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
            #nn.Identity()
        )


        self.root_paths = []

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop, pad=pad)
        self.strip_silence = strip_silence

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity()
        )

        self.sr = sample_rate

        self.volume_norm = VolumeNorm(volume_norm_param, self.sr) if volume_norm else torch.nn.Identity()

        self.custom_metadata_fns = {}

        for config in configs:
            self.root_paths.append(config.path)
            new_files = get_audio_filenames(config.path, config.keywords, filelist_path=config.filelist_path)
            self.filenames.extend(new_files)
            self.sample_weights.extend([config.weight] * len(new_files))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = dill.dumps(config.custom_metadata_fn)

        print(f'Found {len(self.filenames)} files')

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        try:
            start_time = time.time()
            audio = self.load_file(audio_filename)

            audio = self.volume_norm(audio)

            if self.strip_silence:
                audio = strip_trailing_silence(audio, self.sr)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            # Check for silence
            if is_silence(audio):
                return self[random.randrange(len(self))]

            # Run augmentations on this sample (including random crop)
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)

            info = {}

            info["path"] = audio_filename

            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = [padding_mask]
            info["sample_rate"] = self.sr

            end_time = time.time()

            info["load_time"] = end_time - start_time

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in audio_filename:
                    custom_metadata_fn = dill.loads(self.custom_metadata_fns[custom_md_path])
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                # Provide audio inputs as their own dictionary to be merged into info, each audio element will be normalized in the same way as the main audio
                if "__audio__" in info:
                    for audio_key, audio_value in info["__audio__"].items():
                        # Process the audio_value tensor, which should be a torch tensor
                        audio_value, _, _, _, _, _ = self.pad_crop(audio_value)
                        audio_value = audio_value.clamp(-1, 1)
                        if self.encoding is not None:
                            audio_value = self.encoding(audio_value)
                        info[audio_key] = audio_value
                
                    del info["__audio__"]

            return (audio, info)
        except Exception as e:
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self[random.randrange(len(self))]


class PreEncodedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs: List[LatentDatasetConfig],
        latent_crop_length=None,
        min_length_sec=None,
        max_length_sec=None,
        random_crop=False,
        tokenizers: Optional[dict] = None,
    ):
        super().__init__()
        self.filenames = []
        self.sample_weights = []

        self.custom_metadata_fns = {}

        self.silence_latents = {}

        for config in configs:
            new_files = get_latent_filenames(config.path, config.latent_extension, config.filelist_path)
            self.filenames.extend(new_files)
            self.sample_weights.extend([config.weight] * len(new_files))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = dill.dumps(config.custom_metadata_fn)

            # Load silence latent if available (for variable-length padding)
            paths = config.path if isinstance(config.path, list) else [config.path]
            for path in paths:
                silence_path = os.path.join(path, "silence.npy")
                if os.path.exists(silence_path):
                    self.silence_latents[path] = np.load(silence_path).squeeze(0)  # [C, N]
                    print(f'Loaded silence latent from {silence_path}')

        self.latent_crop_length = latent_crop_length
        self.random_crop = random_crop

        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec

        # tokenizers: dict mapping metadata key -> (tokenizer, max_length)
        # If provided, text fields will be pre-tokenized in DataLoader workers
        self.tokenizers = tokenizers

        print(f'Found {len(self.filenames)} files')

    def __len__(self):
        return len(self.filenames)

    def _get_silence_for_file(self, latent_filename):
        """Return the silence latent for the dataset that contains this file, or None."""
        for path, silence in self.silence_latents.items():
            if path in latent_filename:
                return silence
        return None

    def __getitem__(self, idx):
        latent_filename, md_filename = self.filenames[idx]
        try:
            latents = torch.from_numpy(np.load(latent_filename)) # [C, N]

            with open(md_filename, "r") as f:
                try:
                    info = json.load(f)
                except:
                    raise Exception(f"Couldn't load metadata file {md_filename}")

            info["latent_filename"] = latent_filename

            if self.latent_crop_length is not None:
                stored_length = latents.shape[1]

                if stored_length > self.latent_crop_length:
                    # Crop to latent_crop_length (existing logic)
                    # Get the last index from the padding mask, the index of the last 1 in the sequence
                    last_ix = len(info["padding_mask"]) - 1 - info["padding_mask"][::-1].index(1)

                    if self.random_crop and last_ix > self.latent_crop_length:
                        start = random.randint(0, last_ix - self.latent_crop_length)
                    else:
                        start = 0

                    latents = latents[:, start:start+self.latent_crop_length]
                    info["padding_mask"] = info["padding_mask"][start:start+self.latent_crop_length]
                    info["latent_crop_start"] = start

                elif stored_length < self.latent_crop_length:
                    # Pad with silence latent to reach latent_crop_length
                    pad_needed = self.latent_crop_length - stored_length
                    silence = self._get_silence_for_file(latent_filename)

                    if silence is not None:
                        # Slice or tile silence latent to cover pad_needed frames
                        if silence.shape[1] >= pad_needed:
                            silence_pad = silence[:, :pad_needed]
                        else:
                            silence_pad = np.tile(silence, (1, (pad_needed // silence.shape[1]) + 1))[:, :pad_needed]
                        latents = torch.cat([latents, torch.from_numpy(silence_pad)], dim=1)
                    else:
                        # No silence latent available â€” zero-pad as fallback
                        latents = torch.nn.functional.pad(latents, (0, pad_needed))

                    # Build padding_mask: valid frames from stored mask, zeros for padding
                    info["padding_mask"] = info["padding_mask"][:stored_length] + [0] * pad_needed
                    info["latent_crop_start"] = 0

                else:
                    # Exact match
                    info["latent_crop_start"] = 0

                info["latent_crop_length"] = self.latent_crop_length

            info["padding_mask"] = [torch.tensor(info["padding_mask"])]

            seconds_total = info["seconds_total"]

            if self.min_length_sec is not None and seconds_total < self.min_length_sec:
                return self[random.randrange(len(self))]

            if self.max_length_sec is not None and seconds_total > self.max_length_sec:
                return self[random.randrange(len(self))]

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in latent_filename:
                    custom_metadata_fn = dill.loads(self.custom_metadata_fns[custom_md_path])
                    custom_metadata = custom_metadata_fn(info, latents)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                if "__replace__" in info and info["__replace__"] is not None:
                    # Replace the latents with the new latents if the custom metadata function returns a new set of latents
                    latents = info["__replace__"]

            info["audio"] = latents

            # Pre-tokenize text fields in DataLoader workers to avoid
            # CPU contention with the main training thread
            if self.tokenizers is not None:
                for key, (tokenizer, max_length) in self.tokenizers.items():
                    if key in info and isinstance(info[key], str):
                        # Save raw text before replacing with tokens (needed by CLAP and other text-based losses)
                        info[f"{key}_text"] = info[key]
                        encoded = tokenizer(
                            info[key],
                            truncation=True,
                            max_length=max_length,
                            padding="max_length",
                            return_tensors="pt",
                        )
                        info[key] = {
                            "input_ids": encoded["input_ids"].squeeze(0),
                            "attention_mask": encoded["attention_mask"].squeeze(0),
                        }

            return (latents, info)
        except Exception as e:
            print(f'Couldn\'t load file {latent_filename}: {e}')
            return self[random.randrange(len(self))]

# get_dbmax and is_silence copied from https://github.com/drscotthawley/aeiou/blob/main/aeiou/core.py under Apache 2.0 License
# License can be found in LICENSES/LICENSE_AEIOU.txt
def get_dbmax(
    audio,       # torch tensor of (multichannel) audio
    ):
    "finds the loudest value in the entire clip and puts that into dB (full scale)"
    return 20*torch.log10(torch.flatten(audio.abs()).max()).cpu().numpy()

def is_silence(
    audio,       # torch tensor of (multichannel) audio
    thresh=-60,  # threshold in dB below which we declare to be silence
    ):
    "checks if entire clip is 'silence' below some dB threshold"
    dBmax = get_dbmax(audio)
    return dBmax < thresh


def collation_fn(samples):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result



