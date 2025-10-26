# Reference: # https://github.com/bytedance/Make-An-Audio-2

import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5, norm_fn=torch.log10):
    return norm_fn(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes, norm_fn):
    output = dynamic_range_compression_torch(magnitudes, norm_fn=norm_fn)
    return output


class STFTConverter(nn.Module):

    def __init__(
        self,
        *,
        sampling_rate: float = 16_000,
        n_fft: int = 1024,
        num_mels: int = 128,
        hop_size: int = 256,
        win_size: int = 1024,
        fmin: float = 0,
        fmax: float = 8_000,
        norm_fn=torch.log,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.norm_fn = norm_fn

        mel = librosa_mel_fn(sr=self.sampling_rate,
                             n_fft=self.n_fft,
                             n_mels=self.num_mels,
                             fmin=self.fmin,
                             fmax=self.fmax)
        mel_basis = torch.from_numpy(mel).float()
        hann_window = torch.hann_window(self.win_size)

        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('hann_window', hann_window)

    @property
    def device(self):
        return self.hann_window.device

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # input: batch_size * length
        bs = waveform.shape[0]
        waveform = waveform.clamp(min=-1., max=1.)

        spec = torch.stft(waveform,
                          self.n_fft,
                          hop_length=self.hop_size,
                          win_length=self.win_size,
                          window=self.hann_window,
                          center=True,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=True,
                          return_complex=True)

        spec = torch.view_as_real(spec)
        # print('After stft', spec.shape, spec.min(), spec.max(), spec.mean())

        power = (spec.pow(2).sum(-1))**(0.5)
        angle = torch.atan2(spec[..., 1], spec[..., 0])

        print('power 1', power.shape, power.min(), power.max(), power.mean())
        print('angle 1', angle.shape, angle.min(), angle.max(), angle.mean(), angle[:, :2, :2])

        # print('mel', self.mel_basis.shape, self.mel_basis.min(), self.mel_basis.max(),
        #       self.mel_basis.mean())

        # spec = self.mel_transform(spec)

        # power = torch.matmul(self.mel_basis, power)

        spec = rearrange(spec, 'b f t c -> (b c) f t')
        spec = self.mel_basis.unsqueeze(0) @ spec
        spec = rearrange(spec, '(b c) f t -> b f t c', b=bs)

        power = (spec.pow(2).sum(-1))**(0.5)
        angle = torch.atan2(spec[..., 1], spec[..., 0])

        print('power', power.shape, power.min(), power.max(), power.mean())
        print('angle', angle.shape, angle.min(), angle.max(), angle.mean(), angle[:, :2, :2])

        # print('After mel', spec.shape, spec.min(), spec.max(), spec.mean())

        # spec = spectral_normalize_torch(spec, self.norm_fn)

        # print('After norm', spec.shape, spec.min(), spec.max(), spec.mean())

        # compute magnitude
        # magnitude = torch.sqrt((spec**2).sum(-1))
        # normalize by magnitude
        # scaled_magnitude = torch.log10(magnitude.clamp(min=1e-5)) * 10
        # spec = spec / magnitude.unsqueeze(-1) * scaled_magnitude.unsqueeze(-1)

        # power = torch.log10(power.clamp(min=1e-5)) * 10
        power = torch.log10(power.clamp(min=1e-8))

        print('After scaling', power.shape, power.min(), power.max(), power.mean())

        # spec = torch.stack([power, angle], dim=-1)

        # spec = rearrange(spec, '(b c) f t -> b c f t', b=bs)
        # spec = rearrange(spec, 'b f t c -> b c f t', b=bs)

        # spec[:, :, 400:] = 0

        return power, angle
        # return spec[..., 0], spec[..., 1]

    def invert(self, spec: torch.Tensor, length: int) -> torch.Tensor:

        power, angle = spec

        bs = power.shape[0]

        # spec = rearrange(spec, 'b c f t -> (b c) f t')
        # print(spec.shape, self.mel_basis.shape)
        # spec = torch.linalg.lstsq(self.mel_basis.unsqueeze(0), spec).solution
        # spec = torch.linalg.pinv(self.mel_basis.unsqueeze(0)) @ spec

        # spec = self.invmel_transform(spec)

        # spec = rearrange(spec, 'b c f t -> b f t c', b=bs).contiguous()

        # spec[..., 0] = 10**(spec[..., 0] / 10)

        # power = spec[..., 0]
        power = 10**power

        # print('After unscaling', spec[..., 0].shape, spec[..., 0].min(), spec[..., 0].max(),
        #       spec[..., 0].mean())

        unit_vector = torch.stack([
            torch.cos(angle),
            torch.sin(angle),
        ], dim=-1)

        spec = power.unsqueeze(-1) * unit_vector

        # power = torch.linalg.lstsq(self.mel_basis.unsqueeze(0), power).solution
        spec = rearrange(spec, 'b f t c -> (b c) f t')
        spec = torch.linalg.pinv(self.mel_basis.unsqueeze(0)) @ spec
        # spec = torch.linalg.lstsq(self.mel_basis.unsqueeze(0), spec).solution
        spec = rearrange(spec, '(b c) f t -> b f t c', b=bs).contiguous()

        power = (spec.pow(2).sum(-1))**(0.5)
        angle = torch.atan2(spec[..., 1], spec[..., 0])

        print('power 2', power.shape, power.min(), power.max(), power.mean())
        print('angle 2', angle.shape, angle.min(), angle.max(), angle.mean(), angle[:, :2, :2])

        # spec = rearrange(spec, '(b c) f t -> b f t c', b=bs).contiguous()
        spec = torch.view_as_complex(spec)

        waveform = torch.istft(
            spec,
            self.n_fft,
            length=length,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=False,
        )

        return waveform


if __name__ == '__main__':

    converter = STFTConverter(sampling_rate=16000)

    signal = torchaudio.load('./output/ZZ6GRocWW38_000090.wav')[0]
    # resample signal at 44100 Hz
    # signal = torchaudio.transforms.Resample(16_000, 44_100)(signal)

    L = signal.shape[1]
    print('Input signal', signal.shape)
    spec = converter(signal)

    power, angle = spec

    # print(power.shape, angle.shape)
    # print(power, power.min(), power.max(), power.mean())
    # power = power.clamp(-1, 1)
    # angle = angle.clamp(-1, 1)

    import matplotlib.pyplot as plt

    # Visualize power
    plt.figure()
    plt.imshow(power[0].detach().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Power')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig('./output/power.png')

    # Visualize angle
    plt.figure()
    plt.imshow(angle[0].detach().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Angle')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig('./output/angle.png')

    # print('Final spec', spec.shape)

    signal_recon = converter.invert(spec, length=L)
    print('Output signal', signal_recon.shape, signal_recon.min(), signal_recon.max(),
          signal_recon.mean())

    print('MSE', torch.nn.functional.mse_loss(signal, signal_recon))
    torchaudio.save('./output/ZZ6GRocWW38_000090_recon.wav', signal_recon, 16000)
