import dataclasses
import math


@dataclasses.dataclass
class SequenceConfig:
    # general
    duration: float

    # audio
    sampling_rate: int
    spectrogram_frame_rate: int
    latent_downsample_rate: int = 2

    # visual
    clip_frame_rate: int = 8
    sync_frame_rate: int = 25
    sync_num_frames_per_segment: int = 16
    sync_step_size: int = 8
    sync_downsample_rate: int = 2

    @property
    def num_audio_frames(self) -> int:
        # we need an integer number of latents
        return self.latent_seq_len * self.spectrogram_frame_rate * self.latent_downsample_rate

    @property
    def latent_seq_len(self) -> int:
        return int(
            math.ceil(self.duration * self.sampling_rate / self.spectrogram_frame_rate /
                      self.latent_downsample_rate))

    @property
    def clip_seq_len(self) -> int:
        return int(self.duration * self.clip_frame_rate)

    @property
    def sync_seq_len(self) -> int:
        num_frames = self.duration * self.sync_frame_rate
        num_segments = (num_frames - self.sync_num_frames_per_segment) // self.sync_step_size + 1
        return int(num_segments * self.sync_num_frames_per_segment / self.sync_downsample_rate)


CONFIG_16K = SequenceConfig(duration=8.0, sampling_rate=16000, spectrogram_frame_rate=256)
CONFIG_44K = SequenceConfig(duration=8.0, sampling_rate=44100, spectrogram_frame_rate=512)

if __name__ == '__main__':
    assert CONFIG_16K.latent_seq_len == 250
    assert CONFIG_16K.clip_seq_len == 64
    assert CONFIG_16K.sync_seq_len == 192
    assert CONFIG_16K.num_audio_frames == 128000

    assert CONFIG_44K.latent_seq_len == 345
    assert CONFIG_44K.clip_seq_len == 64
    assert CONFIG_44K.sync_seq_len == 192
    assert CONFIG_44K.num_audio_frames == 353280

    print('Passed')
