import torch
import torchaudio
import numpy as np
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import logging

verbose_output = True
USE_SHERPA_ONNX_SPEAKER_DIARIZATION = True
SHERPA_ONNX_SEGMENTATION_MODEL = "sherpa/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
SHERPA_ONNX_EMBEDDING_MODEL = "sherpa/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
SHERPA_ONNX_NUM_SPEAKERS = 2
SHERPA_ONNX_CLUSTER_THRESHOLD = 0.5
SHERPA_ONNX_MIN_DURATION_ON = 0.3
SHERPA_ONNX_MIN_DURATION_OFF = 0.5

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg", "soundfile"]
if not hasattr(torchaudio, "AudioMetaData"):
    from collections import namedtuple
    torchaudio.AudioMetaData = namedtuple("AudioMetaData", "sample_rate num_frames num_channels bits_per_sample encoding")
def _torchaudio_info(uri, backend=None, format=None, buffer_size=4096):
    import soundfile as sf
    info = sf.info(uri)
    bits = 0
    for suffix in ("8", "16", "24", "32", "64"):
        if suffix in str(info.subtype):
            bits = int(suffix)
            break
    return torchaudio.AudioMetaData(sample_rate=info.samplerate, num_frames=info.frames, num_channels=info.channels, bits_per_sample=bits, encoding=info.subtype or info.format)
def _torchaudio_load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, buffer_size=4096, backend=None):
    import soundfile as sf
    start = max(0, int(frame_offset or 0))
    stop = None if num_frames is None or int(num_frames) < 0 else start + int(num_frames)
    audio_data, sample_rate = sf.read(uri, start=start, stop=stop, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio_data.T.copy() if channels_first else audio_data.copy())
    return waveform, sample_rate
def _torchaudio_save(uri, src, sample_rate, channels_first=True, format=None, encoding=None, bits_per_sample=None, buffer_size=4096, backend=None, compression=None):
    import soundfile as sf
    audio_data = src.detach().cpu().float().numpy() if torch.is_tensor(src) else np.asarray(src, dtype=np.float32)
    if channels_first and audio_data.ndim == 2:
        audio_data = audio_data.T
    sf.write(uri, audio_data, int(sample_rate))
torchaudio.info = _torchaudio_info
torchaudio.load = _torchaudio_load
torchaudio.save = _torchaudio_save

try:
    import lightning_fabric.utilities.cloud_io as lightning_cloud_io
    _lightning_load = lightning_cloud_io._load
    if not getattr(_lightning_load, "_wgp_pyannote_trusted_load", False):
        def _load_pyannote_trusted_checkpoint(path_or_url, map_location=None, weights_only=None):
            return _lightning_load(path_or_url, map_location=map_location, weights_only=False if weights_only is None else weights_only)
        _load_pyannote_trusted_checkpoint._wgp_pyannote_trusted_load = True
        lightning_cloud_io._load = _load_pyannote_trusted_checkpoint
except Exception:
    pass

# Suppress specific warnings before importing pyannote
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*was deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Module 'speechbrain.pretrained'.*", category=UserWarning)
# logging.getLogger('speechbrain').setLevel(logging.WARNING)
# logging.getLogger('speechbrain.utils.checkpoints').setLevel(logging.WARNING)
os.environ["SB_LOG_LEVEL"] = "WARNING"   

def xprint(t = None, force: bool = False):
    if verbose_output or force:
        text = str(t)
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))

# Configure TF32 before any CUDA operations to avoid reproducibility warnings
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if USE_SHERPA_ONNX_SPEAKER_DIARIZATION:
    PYANNOTE_AVAILABLE = True
else:
    import speechbrain
    try:
        from pyannote.audio import Pipeline
        PYANNOTE_AVAILABLE = True
    except ImportError:
        PYANNOTE_AVAILABLE = False
        print("Install: pip install pyannote.audio")


class _SimpleSegment:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _SimpleDiarization:
    def __init__(self, segments: List[Tuple[float, float, str]], uri: str = None):
        self.segments = segments
        self.uri = uri

    def labels(self):
        return sorted({speaker for _, _, speaker in self.segments})

    def itertracks(self, yield_label: bool = False):
        for index, (start, end, speaker) in enumerate(self.segments):
            segment = _SimpleSegment(start, end)
            yield (segment, index, speaker) if yield_label else (segment, index)

    def label_timeline(self, speaker: str):
        return [_SimpleSegment(start, end) for start, end, label in self.segments if label == speaker]


class OptimizedPyannote31SpeakerSeparator:
    def __init__(self, hf_token: str = None, local_model_path: str = None, 
                 vad_onset: float = 0.2, vad_offset: float = 0.8):
        """
        Initialize with Pyannote 3.1 pipeline with tunable VAD sensitivity.
        """
        self.hf_token = hf_token
        self._overlap_pipeline = None
        self.use_sherpa_onnx = USE_SHERPA_ONNX_SPEAKER_DIARIZATION
        if self.use_sherpa_onnx:
            self._init_sherpa_onnx_pipeline()
            return

        from shared.utils import files_locator as fl
        embedding_path = fl.locate_file("pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin")
        segmentation_path = fl.locate_file("pyannote/pytorch_model_segmentation-3.0.bin")


        xprint(f"Loading segmentation model from: {segmentation_path}")
        xprint(f"Loading embedding model from: {embedding_path}")
        
        try:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import SpeakerDiarization
            
            # Load models directly
            segmentation_model = Model.from_pretrained(segmentation_path)
            embedding_model = Model.from_pretrained(embedding_path)
            xprint("Models loaded successfully!")
            
            # Create pipeline manually
            self.pipeline = SpeakerDiarization(
                segmentation=segmentation_model,
                embedding=embedding_model,
                clustering='AgglomerativeClustering'
            )
                        
            # Instantiate with default parameters
            self.pipeline.instantiate({
                'clustering': {
                    'method': 'centroid',
                    'min_cluster_size': 12,
                    'threshold': 0.7045654963945799
                },
                'segmentation': {
                    'min_duration_off': 0.0
                }
            })
            xprint("Pipeline instantiated successfully!")
            
            # Send to GPU if available
            if torch.cuda.is_available():
                xprint("CUDA available, moving pipeline to GPU...")
                self.pipeline.to(torch.device("cuda"))
            else:
                xprint("CUDA not available, using CPU...")
                
        except Exception as e:
            xprint(f"Error loading pipeline: {e}")
            xprint(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _init_sherpa_onnx_pipeline(self):
        from shared.utils import files_locator as fl
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ImportError("sherpa-onnx is required for speaker separation when USE_SHERPA_ONNX_SPEAKER_DIARIZATION is True.") from exc

        segmentation_path = fl.locate_file(SHERPA_ONNX_SEGMENTATION_MODEL)
        embedding_path = fl.locate_file(SHERPA_ONNX_EMBEDDING_MODEL)
        xprint(f"Loading Sherpa-ONNX segmentation model from: {segmentation_path}")
        xprint(f"Loading Sherpa-ONNX embedding model from: {embedding_path}")

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation_path)
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding_path),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=SHERPA_ONNX_NUM_SPEAKERS, threshold=SHERPA_ONNX_CLUSTER_THRESHOLD),
            min_duration_on=SHERPA_ONNX_MIN_DURATION_ON,
            min_duration_off=SHERPA_ONNX_MIN_DURATION_OFF,
        )
        if not config.validate():
            raise RuntimeError("Invalid Sherpa-ONNX speaker diarization config. Check the pyannote ONNX and embedding checkpoints.")
        self.pipeline = sherpa_onnx.OfflineSpeakerDiarization(config)
        xprint(f"Sherpa-ONNX speaker diarization ready at {self.pipeline.sample_rate}Hz")

    def separate_audio(self, audio_path: str, output1, output2, audio_original_path: str = None, return_masks: bool = False, speech_masks_only: bool = False) -> Dict[str, str]:
        """Optimized main separation function with memory management."""
        xprint("Starting optimized audio separation...")
        self._current_audio_path = os.path.abspath(audio_path)        
        
        # Suppress warnings during processing
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Load audio
            waveform, sample_rate = self.load_audio(audio_path)
            
            # Perform diarization
            diarization = self.perform_optimized_diarization(audio_path)
            
            # Create masks
            masks = self.create_optimized_speaker_masks(diarization, waveform.shape[1], sample_rate)
            
            # Apply background preservation
            final_masks = self.apply_optimized_background_preservation(masks, waveform.shape[1], sample_rate)
            speech_activity = np.zeros(waveform.shape[1], dtype=bool)
            for speaker in final_masks:
                speech_activity |= masks.get(speaker, np.zeros(waveform.shape[1], dtype=np.float32)) > 0.5
            speech_masks = {speaker: np.asarray(final_masks[speaker] * speech_activity, dtype=np.float32) for speaker in final_masks}
        
        # Clear intermediate results
        del masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Save outputs efficiently
        if audio_original_path is None:
            waveform_original = waveform
        else:
            waveform_original, sample_rate = self.load_audio(audio_original_path)
        save_masks = speech_masks if speech_masks_only else final_masks
        output_paths = self._save_outputs_optimized(waveform_original, save_masks, sample_rate, audio_path, output1, output2)
        
        if return_masks:
            return output_paths, save_masks, sample_rate
        return output_paths

    def _extract_both_speaking_regions(
            self,
            diarization,
            audio_length: int,
            sample_rate: int
        ) -> np.ndarray:
        """
        Detect regions where ≥2 speakers talk simultaneously
        using pyannote/overlapped-speech-detection.
        Falls back to manual pair-wise detection if the model
        is unavailable.
        """
        xprint("Extracting overlap with dedicated pipeline…")
        both_speaking_mask = np.zeros(audio_length, dtype=bool)

        # ── 1) try the proper overlap model ────────────────────────────────
        # overlap_pipeline = self._get_overlap_pipeline() # doesnt work anyway
        overlap_pipeline = None

        # try the path stored by separate_audio – otherwise whatever the
        # diarization object carries (may be None)
        audio_uri = getattr(self, "_current_audio_path", None) \
                    or getattr(diarization, "uri", None)
        if overlap_pipeline and audio_uri:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    overlap_annotation = overlap_pipeline(audio_uri)
                    
                for seg in overlap_annotation.get_timeline().support():
                    s = max(0, int(seg.start * sample_rate))
                    e = min(audio_length, int(seg.end   * sample_rate))
                    if s < e:
                        both_speaking_mask[s:e] = True
                t = np.sum(both_speaking_mask) / sample_rate
                xprint(f"  Found {t:.1f}s of overlapped speech (model) ")
                return both_speaking_mask
            except Exception as e:
                xprint(f"  ⚠ Overlap model failed: {e}")

        # ── 2) fallback = brute-force pairwise intersection ────────────────
        xprint("  Falling back to manual overlap detection…")
        timeline_tracks = list(diarization.itertracks(yield_label=True))
        for i, (turn1, _, spk1) in enumerate(timeline_tracks):
            for j, (turn2, _, spk2) in enumerate(timeline_tracks):
                if i >= j or spk1 == spk2:
                    continue
                o_start, o_end = max(turn1.start, turn2.start), min(turn1.end, turn2.end)
                if o_start < o_end:
                    s = max(0, int(o_start * sample_rate))
                    e = min(audio_length, int(o_end   * sample_rate))
                    if s < e:
                        both_speaking_mask[s:e] = True
        t = np.sum(both_speaking_mask) / sample_rate
        xprint(f"  Found {t:.1f}s of overlapped speech (manual) ")
        return both_speaking_mask

    def _configure_vad(self, vad_onset: float, vad_offset: float):
        """Configure VAD parameters efficiently."""
        xprint("Applying more sensitive VAD parameters...")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                if hasattr(self.pipeline, '_vad'):
                    self.pipeline._vad.instantiate({
                        "onset": vad_onset,
                        "offset": vad_offset,
                        "min_duration_on": 0.1,
                        "min_duration_off": 0.1,
                        "pad_onset": 0.1,
                        "pad_offset": 0.1,
                    })
                    xprint(f"✓ VAD parameters updated: onset={vad_onset}, offset={vad_offset}")
                else:
                    xprint("⚠ Could not access VAD component directly")
        except Exception as e:
            xprint(f"⚠ Could not modify VAD parameters: {e}")

    def _get_overlap_pipeline(self):
        """
        Build a pyannote-3-native OverlappedSpeechDetection pipeline.

        • uses the open-licence `pyannote/segmentation-3.0` checkpoint
        • only `min_duration_on/off` can be tuned (API 3.x)
        """
        if self._overlap_pipeline is not None:
            return None if self._overlap_pipeline is False else self._overlap_pipeline

        try:
            from pyannote.audio.pipelines import OverlappedSpeechDetection

            xprint("Building OverlappedSpeechDetection with segmentation-3.0…")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # 1) constructor → segmentation model ONLY
                ods = OverlappedSpeechDetection(
                    segmentation="pyannote/segmentation-3.0"
                )

                # 2) instantiate → **single dict** with the two valid knobs
                ods.instantiate({
                    "min_duration_on": 0.06,   # ≈ your previous 0.055 s
                    "min_duration_off": 0.10,  # ≈ your previous 0.098 s
                })

                if torch.cuda.is_available():
                    ods.to(torch.device("cuda"))

            self._overlap_pipeline = ods
            xprint("✓ Overlap pipeline ready (segmentation-3.0)")
            return ods

        except Exception as e:
            xprint(f"⚠ Could not build overlap pipeline ({e}). "
                "Falling back to manual pair-wise detection.")
            self._overlap_pipeline = False
            return None
    
    def _xprint_setup_instructions(self):
        """xprint setup instructions."""
        xprint("\nTo use Pyannote 3.1:")
        xprint("1. Get token: https://huggingface.co/settings/tokens")
        xprint("2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1")
        xprint("3. Run with: --token YOUR_TOKEN")
    
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio efficiently."""
        xprint(f"Loading audio: {audio_path}")
        import soundfile as sf

        audio_data, sample_rate = sf.read(os.fspath(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio_data.T.copy())
        
        # Convert to mono efficiently
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        xprint(f"Audio: {waveform.shape[1]} samples at {sample_rate}Hz")
        return waveform, sample_rate
    
    def perform_optimized_diarization(self, audio_path: str) -> object:
        """
        Optimized diarization with efficient parameter testing.
        """
        if self.use_sherpa_onnx:
            return self._perform_sherpa_onnx_diarization(audio_path)

        xprint("Running optimized Pyannote 3.1 diarization...")
        
        # Optimized strategy order - most likely to succeed first
        strategies = [
            {"min_speakers": 2, "max_speakers": 2},  # Most common case
            {"num_speakers": 2},                     # Direct specification
            {"min_speakers": 2, "max_speakers": 3},  # Slight flexibility
            {"min_speakers": 1, "max_speakers": 2},  # Fallback
            {"min_speakers": 2, "max_speakers": 4},  # More flexibility
            {}                                       # No constraints
        ]
        
        for i, params in enumerate(strategies):
            try:
                xprint(f"Strategy {i+1}: {params}")
                
                # Clear GPU memory before each attempt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    diarization = self.pipeline(audio_path, **params)
                    
                speakers = list(diarization.labels())
                speaker_count = len(speakers)
                
                xprint(f"  → Detected {speaker_count} speakers: {speakers}")
                
                # Accept first successful result with 2+ speakers
                if speaker_count >= 2:
                    xprint(f"✓ Success with strategy {i+1}! Using {speaker_count} speakers")
                    return diarization
                elif speaker_count == 1 and i == 0:
                    # Store first result as fallback
                    fallback_diarization = diarization
                    
            except Exception as e:
                xprint(f"  Strategy {i+1} failed: {e}")
                continue
        
        # If we only got 1 speaker, try one aggressive attempt
        if 'fallback_diarization' in locals():
            xprint("Attempting aggressive clustering for single speaker...")
            try:
                aggressive_diarization = self._try_aggressive_clustering(audio_path)
                if aggressive_diarization and len(list(aggressive_diarization.labels())) >= 2:
                    return aggressive_diarization
            except Exception as e:
                xprint(f"Aggressive clustering failed: {e}")
            
            xprint("Using single speaker result")
            return fallback_diarization
        
        # Last resort - run without constraints
        xprint("Last resort: running without constraints...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return self.pipeline(audio_path)

    def _perform_sherpa_onnx_diarization(self, audio_path: str) -> object:
        xprint("Running Sherpa-ONNX speaker diarization...")
        import librosa
        import soundfile as sf

        audio_data, sample_rate = sf.read(os.fspath(audio_path), dtype="float32", always_2d=True)
        audio = audio_data.mean(axis=1)
        target_sample_rate = self.pipeline.sample_rate
        if sample_rate != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        result = self.pipeline.process(audio).sort_by_start_time()
        segments = [(float(segment.start), float(segment.end), f"SPEAKER_{int(segment.speaker):02d}") for segment in result]
        speakers = sorted({speaker for _, _, speaker in segments})
        xprint(f"  -> Detected {len(speakers)} speakers: {speakers}")
        for start, end, speaker in segments:
            xprint(f"  {speaker}: {start:.3f}s - {end:.3f}s")
        return _SimpleDiarization(segments, uri=os.path.abspath(audio_path))
    
    def _try_aggressive_clustering(self, audio_path: str) -> object:
        """Try aggressive clustering parameters."""
        try:
            from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Create aggressive pipeline
                temp_pipeline = SpeakerDiarization(
                    segmentation=self.pipeline.segmentation,
                    embedding=self.pipeline.embedding,
                    clustering="AgglomerativeClustering"
                )
                
                temp_pipeline.instantiate({
                    "clustering": {
                        "method": "centroid",
                        "min_cluster_size": 1,
                        "threshold": 0.1,
                    },
                    "segmentation": {
                        "min_duration_off": 0.0,
                        "min_duration_on": 0.1,
                    }
                })
                
                return temp_pipeline(audio_path, min_speakers=2)
            
        except Exception as e:
            xprint(f"Aggressive clustering setup failed: {e}")
            return None
    
    def create_optimized_speaker_masks(self, diarization, audio_length: int, sample_rate: int) -> Dict[str, np.ndarray]:
        """Optimized mask creation using vectorized operations."""
        xprint("Creating optimized speaker masks...")
        
        speakers = list(diarization.labels())
        first_starts = self._get_speaker_first_starts(diarization, speakers, sample_rate)
        self._speaker_first_starts = first_starts
        speakers = sorted(speakers, key=lambda speaker: (first_starts.get(speaker, audio_length), speaker))
        xprint(f"Processing speakers: {speakers}")
        
        # Handle edge cases
        if len(speakers) == 0:
            xprint("WARNING: no speakers found; returning silent speaker outputs.", force=True)
            return {
                "SPEAKER_SILENT_00": np.zeros(audio_length, dtype=np.float32),
                "SPEAKER_SILENT_01": np.zeros(audio_length, dtype=np.float32)
            }
        
        if len(speakers) == 1:
            speaker = speakers[0]
            xprint(f"WARNING: only one speaker found ({speaker}); no artificial second speaker split will be created.", force=True)
            segments = []
            for segment in diarization.label_timeline(speaker):
                start_sample = max(0, int(segment.start * sample_rate))
                end_sample = min(audio_length, int(segment.end * sample_rate))
                if start_sample < end_sample:
                    segments.append((start_sample, end_sample))
            return {
                speaker: self._create_mask_vectorized(segments, audio_length),
                "SPEAKER_SILENT": np.zeros(audio_length, dtype=np.float32)
            }
        
        # Extract both-speaking regions from diarization timeline
        both_speaking_regions = self._extract_both_speaking_regions(diarization, audio_length, sample_rate)
        
        # Optimized mask creation for multiple speakers
        masks = {}
        
        # Batch process all speakers
        for speaker in speakers:
            # Get all segments for this speaker at once
            segments = []
            speaker_timeline = diarization.label_timeline(speaker)
            for segment in speaker_timeline:
                start_sample = max(0, int(segment.start * sample_rate))
                end_sample = min(audio_length, int(segment.end * sample_rate))
                if start_sample < end_sample:
                    segments.append((start_sample, end_sample))
            
            # Vectorized mask creation
            if segments:
                mask = self._create_mask_vectorized(segments, audio_length)
                masks[speaker] = mask
                speaking_time = np.sum(mask) / sample_rate
                xprint(f"  {speaker}: {speaking_time:.1f}s speaking time")
            else:
                masks[speaker] = np.zeros(audio_length, dtype=np.float32)
        
        # Store both-speaking info for later use
        self._both_speaking_regions = both_speaking_regions
        
        return masks

    def _get_speaker_first_starts(self, diarization, speakers: List[str], sample_rate: int) -> Dict[str, int]:
        first_starts = {speaker: np.iinfo(np.int64).max for speaker in speakers}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker in first_starts:
                first_starts[speaker] = min(first_starts[speaker], max(0, int(turn.start * sample_rate)))
        return first_starts
    
    def _create_mask_vectorized(self, segments: List[Tuple[int, int]], audio_length: int) -> np.ndarray:
        """Create mask using vectorized operations."""
        mask = np.zeros(audio_length, dtype=np.float32)
        
        if not segments:
            return mask
        
        # Convert segments to arrays for vectorized operations
        segments_array = np.array(segments)
        starts = segments_array[:, 0]
        ends = segments_array[:, 1]
        
        # Use advanced indexing for bulk assignment
        for start, end in zip(starts, ends):
            mask[start:end] = 1.0
        
        return mask
    
    def apply_optimized_background_preservation(self, masks: Dict[str, np.ndarray], audio_length: int, sample_rate: int) -> Dict[str, np.ndarray]:
        """
        Heavily optimized background preservation using pure vectorized operations.
        """
        if any(speaker.startswith("SPEAKER_SILENT") for speaker in masks):
            return masks

        xprint("Applying optimized voice separation logic...")
        
        # Ensure exactly 2 speakers
        speaker_keys = self._get_top_speakers(masks, audio_length)
        
        # Pre-allocate final masks
        final_masks = {
            speaker: np.zeros(audio_length, dtype=np.float32) 
            for speaker in speaker_keys
        }
        
        # Get active masks (vectorized)
        active_0 = masks.get(speaker_keys[0], np.zeros(audio_length)) > 0.5
        active_1 = masks.get(speaker_keys[1], np.zeros(audio_length)) > 0.5
        
        # Vectorized mask assignment
        both_active = active_0 & active_1
        only_0 = active_0 & ~active_1
        only_1 = ~active_0 & active_1
        neither = ~active_0 & ~active_1
        
        # Apply assignments (all vectorized)
        final_masks[speaker_keys[0]][both_active] = 1.0
        final_masks[speaker_keys[1]][both_active] = 1.0
        
        final_masks[speaker_keys[0]][only_0] = 1.0
        final_masks[speaker_keys[1]][only_0] = 0.0
        
        final_masks[speaker_keys[0]][only_1] = 0.0
        final_masks[speaker_keys[1]][only_1] = 1.0
        
        # Handle ambiguous regions efficiently
        if np.any(neither):
            ambiguous_assignments = self._compute_ambiguous_assignments_vectorized(
                masks, speaker_keys, neither, audio_length
            )
            
            # Apply ambiguous assignments
            final_masks[speaker_keys[0]][neither] = (ambiguous_assignments == 0).astype(np.float32) * 0.5
            final_masks[speaker_keys[1]][neither] = (ambiguous_assignments == 1).astype(np.float32) * 0.5
        
        # xprint statistics (vectorized)
        xprint(f"  Both speaking clearly: {np.sum(both_active)/sample_rate:.1f}s")
        xprint(f"  {speaker_keys[0]} only: {np.sum(only_0)/sample_rate:.1f}s")
        xprint(f"  {speaker_keys[1]} only: {np.sum(only_1)/sample_rate:.1f}s")
        xprint(f"  Ambiguous (assigned): {np.sum(neither)/sample_rate:.1f}s")
        
        # Apply minimum duration smoothing to prevent rapid switching
        final_masks = self._apply_minimum_duration_smoothing(final_masks, sample_rate)
        
        return final_masks
    
    def _get_top_speakers(self, masks: Dict[str, np.ndarray], audio_length: int) -> List[str]:
        """Get top 2 speakers by speaking time."""
        speaker_keys = list(masks.keys())
        
        if len(speaker_keys) > 2:
            # Vectorized speaking time calculation
            speaking_times = {k: np.sum(v) for k, v in masks.items()}
            speaker_keys = sorted(speaking_times.keys(), key=lambda x: speaking_times[x], reverse=True)[:2]
            first_starts = getattr(self, "_speaker_first_starts", {})
            speaker_keys = sorted(speaker_keys, key=lambda speaker: (first_starts.get(speaker, audio_length), speaker))
            xprint(f"Keeping top 2 speakers: {speaker_keys}")
        return speaker_keys
    
    def _compute_ambiguous_assignments_vectorized(self, masks: Dict[str, np.ndarray], 
                                                speaker_keys: List[str], 
                                                ambiguous_mask: np.ndarray, 
                                                audio_length: int) -> np.ndarray:
        """Compute speaker assignments for ambiguous regions using vectorized operations."""
        ambiguous_indices = np.where(ambiguous_mask)[0]
        
        if len(ambiguous_indices) == 0:
            return np.array([])
        
        # Get speaker segments efficiently
        speaker_segments = {}
        for speaker in speaker_keys:
            if speaker in masks and speaker != "SPEAKER_SILENT":
                mask = masks[speaker] > 0.5
                # Find segments using vectorized operations
                diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                speaker_segments[speaker] = np.column_stack([starts, ends])
            else:
                speaker_segments[speaker] = np.array([]).reshape(0, 2)
        
        # Vectorized distance calculations
        distances = {}
        for speaker in speaker_keys:
            segments = speaker_segments[speaker]
            if len(segments) == 0:
                distances[speaker] = np.full(len(ambiguous_indices), np.inf)
            else:
                # Compute distances to all segments at once
                distances[speaker] = self._compute_distances_to_segments(ambiguous_indices, segments)
        
        # Assign based on minimum distance with late-audio bias
        assignments = self._assign_based_on_distance(
            distances, speaker_keys, ambiguous_indices, audio_length
        )
        
        return assignments
    
    def _apply_minimum_duration_smoothing(self, masks: Dict[str, np.ndarray], 
                                        sample_rate: int, min_duration_ms: int = 600) -> Dict[str, np.ndarray]:
        """
        Apply minimum duration smoothing with STRICT timer enforcement.
        Uses original both-speaking regions from diarization.
        """
        xprint(f"Applying STRICT minimum duration smoothing ({min_duration_ms}ms)...")
        
        min_samples = int(min_duration_ms * sample_rate / 1000)
        speaker_keys = list(masks.keys())
        
        if len(speaker_keys) != 2:
            return masks
        
        mask0 = masks[speaker_keys[0]]
        mask1 = masks[speaker_keys[1]]
        
        # Use original both-speaking regions from diarization
        both_speaking_original = getattr(self, '_both_speaking_regions', np.zeros(len(mask0), dtype=bool))
        
        # Identify regions based on original diarization info
        ambiguous_original = (mask0 < 0.3) & (mask1 < 0.3) & ~both_speaking_original
        
        # Clear dominance: one speaker higher, and not both-speaking or ambiguous
        remaining_mask = ~both_speaking_original & ~ambiguous_original
        speaker0_dominant = (mask0 > mask1) & remaining_mask
        speaker1_dominant = (mask1 > mask0) & remaining_mask
        
        # Create preference signal including both-speaking as valid state
        # -1=ambiguous, 0=speaker0, 1=speaker1, 2=both_speaking
        preference_signal = np.full(len(mask0), -1, dtype=int)
        preference_signal[speaker0_dominant] = 0
        preference_signal[speaker1_dominant] = 1
        preference_signal[both_speaking_original] = 2
        
        # STRICT state machine enforcement
        smoothed_assignment = np.full(len(mask0), -1, dtype=int)
        corrections = 0
        
        # State variables
        current_state = -1  # -1=unset, 0=speaker0, 1=speaker1, 2=both_speaking
        samples_remaining = 0  # Samples remaining in current state's lock period
        
        # Process each sample with STRICT enforcement
        for i in range(len(preference_signal)):
            preference = preference_signal[i]
            
            # If we're in a lock period, enforce the current state
            if samples_remaining > 0:
                # Force current state regardless of preference
                smoothed_assignment[i] = current_state
                samples_remaining -= 1
                
                # Count corrections if this differs from preference
                if preference >= 0 and preference != current_state:
                    corrections += 1
                    
            else:
                # Lock period expired - can consider new state
                
                if preference >= 0:
                    # Clear preference available (including both-speaking)
                    if current_state != preference:
                        # Switch to new state and start new lock period
                        current_state = preference
                        samples_remaining = min_samples - 1  # -1 because we use this sample
                        
                    smoothed_assignment[i] = current_state
                    
                else:
                    # Ambiguous preference
                    if current_state >= 0:
                        # Continue with current state if we have one
                        smoothed_assignment[i] = current_state
                    else:
                        # No current state and ambiguous - leave as ambiguous
                        smoothed_assignment[i] = -1
        
        # Convert back to masks based on smoothed assignment
        smoothed_masks = {}
        
        for i, speaker in enumerate(speaker_keys):
            new_mask = np.zeros_like(mask0)
            
            # Assign regions where this speaker is dominant
            speaker_regions = smoothed_assignment == i
            new_mask[speaker_regions] = 1.0
            
            # Assign both-speaking regions (state 2) to both speakers
            both_speaking_regions = smoothed_assignment == 2
            new_mask[both_speaking_regions] = 1.0
            
            # Handle ambiguous regions that remain unassigned
            unassigned_ambiguous = smoothed_assignment == -1
            if np.any(unassigned_ambiguous):
                # Use original ambiguous values only for truly unassigned regions
                original_ambiguous_mask = ambiguous_original & unassigned_ambiguous
                new_mask[original_ambiguous_mask] = masks[speaker][original_ambiguous_mask]
            
            smoothed_masks[speaker] = new_mask
        
        # Calculate and xprint statistics
        both_speaking_time = np.sum(smoothed_assignment == 2) / sample_rate
        speaker0_time = np.sum(smoothed_assignment == 0) / sample_rate
        speaker1_time = np.sum(smoothed_assignment == 1) / sample_rate
        ambiguous_time = np.sum(smoothed_assignment == -1) / sample_rate
        
        xprint(f"  Both speaking clearly: {both_speaking_time:.1f}s")
        xprint(f"  {speaker_keys[0]} only: {speaker0_time:.1f}s")
        xprint(f"  {speaker_keys[1]} only: {speaker1_time:.1f}s")
        xprint(f"  Ambiguous (assigned): {ambiguous_time:.1f}s")
        xprint(f"  Enforced minimum duration on {corrections} samples ({corrections/sample_rate:.2f}s)")
        
        return smoothed_masks
    
    def _compute_distances_to_segments(self, indices: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """Compute minimum distances from indices to segments (vectorized)."""
        if len(segments) == 0:
            return np.full(len(indices), np.inf)
        
        # Broadcast for vectorized computation
        indices_expanded = indices[:, np.newaxis]  # Shape: (n_indices, 1)
        starts = segments[:, 0]  # Shape: (n_segments,)
        ends = segments[:, 1]    # Shape: (n_segments,)
        
        # Compute distances to all segments
        dist_to_start = np.maximum(0, starts - indices_expanded)  # Shape: (n_indices, n_segments)
        dist_from_end = np.maximum(0, indices_expanded - ends)    # Shape: (n_indices, n_segments)
        
        # Minimum of distance to start or from end for each segment
        distances = np.minimum(dist_to_start, dist_from_end)
        
        # Return minimum distance to any segment for each index
        return np.min(distances, axis=1)
    
    def _assign_based_on_distance(self, distances: Dict[str, np.ndarray], 
                                speaker_keys: List[str], 
                                ambiguous_indices: np.ndarray, 
                                audio_length: int) -> np.ndarray:
        """Assign speakers based on distance with late-audio bias."""
        speaker_0_distances = distances[speaker_keys[0]]
        speaker_1_distances = distances[speaker_keys[1]]
        
        # Basic assignment by minimum distance
        assignments = (speaker_1_distances < speaker_0_distances).astype(int)
        
        # Apply late-audio bias (vectorized)
        late_threshold = int(audio_length * 0.6)
        late_indices = ambiguous_indices > late_threshold
        
        if np.any(late_indices) and len(speaker_keys) > 1:
            # Simple late-audio bias: prefer speaker 1 in later parts
            assignments[late_indices] = 1
        
        return assignments
    
    def _save_outputs_optimized(self, waveform: torch.Tensor, masks: Dict[str, np.ndarray], 
                               sample_rate: int, audio_path: str, output1, output2) -> Dict[str, str]:
        """Optimized output saving with parallel processing."""
        output_paths = {}
        
        def save_speaker_audio(speaker_mask_pair, output):
            speaker, mask = speaker_mask_pair
            # Convert mask to tensor efficiently
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            
            # Apply mask
            masked_audio = waveform * mask_tensor
            
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                import soundfile as sf
                sf.write(output, masked_audio.squeeze(0).detach().cpu().numpy(), sample_rate)
            
            xprint(f"✓ Saved {speaker}: {output}")
            return speaker, output
        
        # Use ThreadPoolExecutor for parallel saving
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(save_speaker_audio, masks.items(), [output1, output2]))
        
        output_paths = dict(results)
        return output_paths
    
    def print_summary(self, audio_path: str):
        """xprint diarization summary."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            diarization = self.perform_optimized_diarization(audio_path)
        
        xprint("\n=== Diarization Summary ===")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            xprint(f"{speaker}: {turn.start:.1f}s - {turn.end:.1f}s")

def extract_dual_audio(audio, output1, output2, verbose = False, audio_original = None, return_masks: bool = False, speech_masks_only: bool = False):
    global verbose_output
    from preprocessing.speaker_separator.assets import download_speaker_separator

    download_speaker_separator()
    verbose_output = verbose
    separator = OptimizedPyannote31SpeakerSeparator(
        None, 
        None,
        vad_onset=0.2,
        vad_offset=0.8
    )
    # Separate audio
    import time
    start_time = time.time()
    
    result = separator.separate_audio(audio, output1, output2, audio_original, return_masks=return_masks, speech_masks_only=speech_masks_only)
    outputs = result[0] if return_masks else result
    
    elapsed_time = time.time() - start_time
    xprint(f"\n=== SUCCESS (completed in {elapsed_time:.2f}s) ===")
    for speaker, path in outputs.items():
        xprint(f"{speaker}: {path}")
    return result

def main():
    
    parser = argparse.ArgumentParser(description="Optimized Pyannote 3.1 Speaker Separator")
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--token", help="Hugging Face token")
    parser.add_argument("--local-model", help="Path to local 3.1 model")
    parser.add_argument("--summary", action="store_true", help="xprint summary")
    
    # VAD sensitivity parameters
    parser.add_argument("--vad-onset", type=float, default=0.2, 
                       help="VAD onset threshold (lower = more sensitive to speech start, default: 0.2)")
    parser.add_argument("--vad-offset", type=float, default=0.8,
                       help="VAD offset threshold (higher = keeps speech longer, default: 0.8)")
    
    args = parser.parse_args()
    
    xprint("=== Optimized Pyannote 3.1 Speaker Separator ===")
    xprint("Performance optimizations: vectorized operations, memory management, parallel processing")
    xprint(f"Audio: {args.audio}")
    xprint(f"Output: {args.output}")
    xprint(f"VAD onset: {args.vad_onset}")
    xprint(f"VAD offset: {args.vad_offset}")
    xprint()
    
    if not os.path.exists(args.audio):
        xprint(f"ERROR: Audio file not found: {args.audio}")
        return
    
    try:
        # Initialize with VAD parameters
        separator = OptimizedPyannote31SpeakerSeparator(
            args.token, 
            args.local_model,
            vad_onset=args.vad_onset,
            vad_offset=args.vad_offset
        )
        
        # print summary if requested
        if args.summary:
            separator.print_summary(args.audio)
        
        # Separate audio
        import time
        start_time = time.time()

        audio_name = Path(args.audio).stem
        output_filename = f"{audio_name}_speaker0.wav"
        output_filename1 = f"{audio_name}_speaker1.wav"
        output_path = os.path.join(args.output, output_filename)
        output_path1 = os.path.join(args.output, output_filename1)

        outputs = separator.separate_audio(args.audio, output_path, output_path1)
        
        elapsed_time = time.time() - start_time
        xprint(f"\n=== SUCCESS (completed in {elapsed_time:.2f}s) ===")
        for speaker, path in outputs.items():
            xprint(f"{speaker}: {path}")
            
    except Exception as e:
        xprint(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
