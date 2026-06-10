import torch

from torch import nn
from torch.nn.utils import weight_norm
from torchaudio import transforms as T
from einops import rearrange

from ..inference.audio_utils import prepare_audio
from .transformer import TransformerBlock

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, **kwargs):
        return rearrange(x, '... a b -> ... b a')

def _zero_pad_modulo_sequence(x, size, dim=-2):
    input_len = x.shape[dim]
    pad_len = (size - input_len % size) % size
    if pad_len > 0:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        x = torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)
    return x

class TransformerResamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sliding_window = None, chunk_size = 128, chunk_midpoint_shift = False, type = 'encoder', transformer_depth = 3, checkpointing = False,
                 conformer = False, layer_scale = False, dim_heads = 128, differential = True, variable_stride = False, feat_scale = False,
                 sinusoidal_blocks = 0, mask_noise = 0, ff_mult = 3, mapping_bias = True, cross_attn = False, dyt = True, conv_mapping = False, freeze_backbone = False, **kwargs):
        super().__init__()
        if type not in ['encoder', 'decoder']:
            raise ValueError(f"Unknown type {type}. Must be 'encoder' or 'decoder'")

        self.checkpointing = checkpointing

        transformer_dim = out_channels if type == 'encoder' else in_channels
        transformers = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable_stride = variable_stride
        self.stride = stride
        self.mapping = WNConv1d(in_channels, out_channels, 3 if conv_mapping else 1, padding = 'same', bias = mapping_bias) if in_channels != out_channels else nn.Identity()
        self.chunk_size = chunk_size
        self.chunk_midpoint_shift = chunk_midpoint_shift
        self.type = type
        self.mask_noise = mask_noise
        self.sliding_window_latents = sliding_window

        self.sliding_window_seq = self._get_sliding_window_size(sliding_window, stride)
        self.input_seg_size, self.output_seg_size, self.sub_chunk_size = self._get_seg_sizes(stride)
        self.transformer_depth = transformer_depth
        for i in range(transformer_depth):
            sinusoidal = True if ((transformer_depth - i) < sinusoidal_blocks) else False
            transformers.append(TransformerBlock(transformer_dim,
                                                 dim_heads = dim_heads,
                                                 causal = False,
                                                 zero_init_branch_outputs = True if not layer_scale else False,
                                                 norm_type = 'dyt' if dyt else 'rms_norm',
                                                 conformer = conformer,
                                                 layer_scale = layer_scale,
                                                 add_rope = True,
                                                 attn_kwargs={'qk_norm': "dyt" if dyt else "rms", "qk_norm_eps": 1e-3, "differential": differential, "feat_scale": feat_scale},
                                                 ff_kwargs={'mult': ff_mult, 'no_bias': False, "sinusoidal": sinusoidal},
                                                 norm_kwargs = {'eps': 1e-3},
                                                 cross_attend = cross_attn))

        self.new_tokens = nn.Parameter(1e-5 * torch.randn(1, self.output_seg_size if not self.variable_stride else 1, out_channels if type == 'encoder' else in_channels))

        self.transformers = nn.ModuleList(transformers)

        if freeze_backbone:
            for param in self.transformers.parameters():
                param.requires_grad = False
            self.new_tokens.requires_grad = False

    def _get_sliding_window_size(self, window, stride, prepend_cond_length = 0):
        if window is None:
            return None
        else:
            return [(win * (stride + 1 + prepend_cond_length)) for win in window]

    def _get_seg_sizes(self, stride, prepend_cond_length = 0):
        sub_chunk_size = stride + 1 + prepend_cond_length
        if self.sliding_window_latents is None:
            assert (self.chunk_size % stride) == 0, f"Stride must fit evenly into chunk size:{self.chunk_size}"
        input_seg_size = stride if self.type == 'encoder' else 1
        output_seg_size = 1 if self.type == 'encoder' else stride
        return input_seg_size, output_seg_size, sub_chunk_size

    def forward(self, x, stride = None, return_features = False, override_new_tokens = None, prepend_cond = None, cross_attn_cond = None):
        batch_size = x.shape[0]
        input_length = x.shape[-1]
        if return_features:
            features = []

        if stride == None:
            input_seg_size = self.input_seg_size
            output_seg_size = self.output_seg_size
            sub_chunk_size = self.sub_chunk_size
            sliding_window = self.sliding_window_seq
        else:
            if not self.variable_stride:
                print("cannot override stride if variable_stride is not set")
            prepend_cond_length = prepend_cond.shape[-2] if prepend_cond is not None else 0
            input_seg_size, output_seg_size, sub_chunk_size = self._get_seg_sizes(stride, prepend_cond_length)
            sliding_window = self._get_sliding_window_size(self.sliding_window_latents, stride, prepend_cond_length)

        if self.type == 'encoder':
            # Pad before mapping so silence zeros get projected through the mapping,
            # rather than inserting raw zeros in the mapped space
            if self.transformer_depth > 0:
                if sliding_window is None:
                    pad_modulo = self.chunk_size
                else:
                    pad_modulo = input_seg_size
                x = _zero_pad_modulo_sequence(x, pad_modulo, dim=-1)
            x = self.mapping(x)

        if self.transformer_depth > 0:
            x = rearrange(x, '... a b -> ... b a')
            if return_features:
                features.append(x)
            if self.type != 'encoder':
                if sliding_window is None:
                    active_stride = stride if stride is not None else self.stride
                    pad_modulo = self.chunk_size // active_stride
                    x = _zero_pad_modulo_sequence(x, pad_modulo)
                else:
                    x = _zero_pad_modulo_sequence(x, input_seg_size)
            x = rearrange(x, 'b (n c) d -> (b n) c d', c = input_seg_size)
            new_token_seq_dim = -1 if not self.variable_stride else output_seg_size
            new_tokens = self.new_tokens.expand([x.shape[0],new_token_seq_dim,-1])
            if override_new_tokens is not None:
                #print(f"Using override new tokens with shape {override_new_tokens.shape}, new tokens shape {new_tokens.shape}, x shape {x.shape}")
                override_new_tokens = rearrange(override_new_tokens, 'b (n c) d -> (b n) c d', c = output_seg_size)
                new_tokens = new_tokens + override_new_tokens
            elif self.mask_noise > 0:
                new_tokens = new_tokens + torch.randn_like(new_tokens) * self.mask_noise
            x = torch.cat([x,new_tokens], dim = -2)
            if prepend_cond is not None:
                n = x.shape[0] // batch_size
                cond_folded = prepend_cond.unsqueeze(1).expand(batch_size, n, prepend_cond.shape[-2], x.shape[-1]).reshape(n * batch_size, prepend_cond.shape[-2], x.shape[-1])
                x = torch.cat([cond_folded, x], dim=-2)   
            x = rearrange(x, '(b n) c d -> b (n c) d', b=batch_size)#.contiguous()

            # Fold into contiguous chunks if no sliding window
            if sliding_window is None:
                prepend_cond_length = prepend_cond.shape[-2] if prepend_cond is not None else 0
                effective_chunk_size = self.chunk_size + self.chunk_size * (1 + prepend_cond_length) // (stride if stride is not None else self.stride)

            if sliding_window is None and self.chunk_midpoint_shift:
                split = self.transformer_depth // 2
                shift = effective_chunk_size // 2

                # First half: standard chunks
                nc = x.shape[1] // effective_chunk_size
                x = rearrange(x, 'b (nc cc) d -> (b nc) cc d', cc=effective_chunk_size)
                cross_attn_first = None
                if cross_attn_cond is not None:
                    cross_attn_first = cross_attn_cond.repeat_interleave(nc, dim=0)
                for layer in self.transformers[:split]:
                    if self.checkpointing:
                        x = checkpoint(layer, x, context = cross_attn_first, self_attention_flash_sliding_window = None)
                    else:
                        x = layer(x, context = cross_attn_first)
                    if return_features:
                        features.append(rearrange(x, '(b nc) cc d -> b (nc cc) d', b=batch_size))
                x = rearrange(x, '(b nc) cc d -> b (nc cc) d', b=batch_size)

                # Second half: shifted chunks with sub-chunk repeat padding
                # shift is always a multiple of sub_chunk_size, so slicing
                # by shift gives whole sub-chunks in their original order
                x = torch.cat([x[:, :shift, :], x, x[:, -shift:, :]], dim=1)
                nc_shifted = x.shape[1] // effective_chunk_size
                x = rearrange(x, 'b (nc cc) d -> (b nc) cc d', cc=effective_chunk_size)
                cross_attn_second = None
                if cross_attn_cond is not None:
                    cross_attn_second = cross_attn_cond.repeat_interleave(nc_shifted, dim=0)
                for layer in self.transformers[split:]:
                    if self.checkpointing:
                        x = checkpoint(layer, x, context = cross_attn_second, self_attention_flash_sliding_window = None)
                    else:
                        x = layer(x, context = cross_attn_second)
                    if return_features:
                        feat = rearrange(x, '(b nc) cc d -> b (nc cc) d', b=batch_size)
                        features.append(feat[:, shift:-shift, :])
                x = rearrange(x, '(b nc) cc d -> b (nc cc) d', b=batch_size)
                x = x[:, shift:-shift, :]
            else:
                if sliding_window is None:
                    x = rearrange(x, 'b (nc cc) d -> (b nc) cc d', cc=effective_chunk_size)

                for layer in self.transformers:
                    if self.checkpointing:
                        x = checkpoint(layer, x, context = cross_attn_cond, self_attention_flash_sliding_window = sliding_window)
                    else:
                        x = layer(x, context = cross_attn_cond, self_attention_flash_sliding_window = sliding_window)
                    if return_features:
                        features.append(x)

                # Unfold chunks back to original batch
                if sliding_window is None:
                    x = rearrange(x, '(b nc) cc d -> b (nc cc) d', b=batch_size)

            x = rearrange(x, 'b (n c) d -> (b n) c d', c=sub_chunk_size)
            x = x[:,-output_seg_size:,:]
            x = rearrange(x, '(b n) c d -> b d (n c)', b=batch_size)
        if self.type == 'decoder':
            x = self.mapping(x)
        if return_features:
            return x, features
        else:   
            return x


class SAMEEncoder(nn.Module):
    def __init__(self,
                 in_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults = [1, 2, 4, 8],
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 sliding_window = None,
                 checkpointing = False,
                 conformer = False,
                 layer_scale = False,
                 causal = False,
                 differential = True,
                 variable_stride = False,
                 mask_noise = 0.0,
                 conv_mapping = False,
                 freeze_backbone = False,
                 **kwargs
        ):
        super().__init__()
        self.in_channels = in_channels
        self.strides = strides

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [in_channels] + channel_dims

        self.depth = len(c_mults)

        layers = []

        for i in range(self.depth):
            layers += [TransformerResamplingBlock(in_channels=channel_dims[i], out_channels=channel_dims[i+1], stride=strides[i], transformer_depth = transformer_depths[i],
                                                  sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, causal = causal,
                                                  differential = differential, variable_stride = variable_stride, mask_noise = mask_noise, conv_mapping = conv_mapping,
                                                  freeze_backbone = freeze_backbone, **kwargs)]

        layers += [Transpose(), nn.Linear(channel_dims[-1], latent_dim), Transpose()]
        self.layers = nn.ModuleList(layers)

        if freeze_backbone:
            for param in self.layers[-2].parameters():
                param.requires_grad = False

    def forward(self, x, override_stride = None, return_features = False, **kwargs):
        if override_stride != None:
            assert isinstance(override_stride, list), "override_stride must be a list"
            assert len(override_stride) == self.depth, "override_stride must be a list containing strides for every layer"
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerResamplingBlock):
                if override_stride != None:
                    stride = override_stride[i]
                else:
                    stride = None
                if return_features:
                    x, features = layer(x, stride = stride, return_features = True)
                else:
                    x = layer(x, stride = stride)
            else:
                x = layer(x)
        if return_features:
            return x, features
        else:
            return x

class SAMEDecoder(nn.Module):
    def __init__(self,
                 out_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults = [1, 2, 4, 8],
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 sliding_window = None,
                 checkpointing = False,
                 conformer = False,
                 layer_scale = False,
                 causal = False,
                 differential = True,
                 variable_stride = False,
                 sinusoidal_blocks = [0,0,0,0],
                 mask_noise = 0.0,
                 conv_mapping = False,
                 freeze_backbone = False,
                 **kwargs
        ):
        super().__init__()

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [out_channels] + channel_dims

        self.depth = len(c_mults)

        layers = [Transpose(), nn.Linear(latent_dim, channel_dims[-1]), Transpose()]

        for i in range(self.depth, 0, -1):
            layers += [TransformerResamplingBlock(in_channels=channel_dims[i], out_channels=channel_dims[i-1], stride=strides[i-1], type = 'decoder', transformer_depth = transformer_depths[i-1],
                                                  sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, causal = causal, differential = differential,
                                                  variable_stride = variable_stride, sinusoidal_blocks = sinusoidal_blocks[i-1], mask_noise = mask_noise, conv_mapping = conv_mapping,
                                                  freeze_backbone = freeze_backbone, **kwargs)]

        self.layers = nn.ModuleList(layers)

        if freeze_backbone:
            for param in self.layers[1].parameters():
                param.requires_grad = False

    def forward(self, x, override_stride = None, **kwargs):
        if override_stride != None:
            assert isinstance(override_stride, list), "override_stride must be a list"
            assert len(override_stride) == self.depth, "override_stride must be a list containing strides for every layer"

        transformer_layer_index = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerResamplingBlock):
                if override_stride != None:
                    stride = override_stride[transformer_layer_index]
                else:
                    stride = None
                x = layer(x, stride = stride)
                transformer_layer_index += 1
            else:
                x = layer(x)
        return x


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck: nn.Module = None,
        pretransform: nn.Module = None,
        in_channels = None,
        out_channels = None,
        soft_clip = False,
        freeze_pretransform = False
    ):
        super().__init__()  
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels

        self.min_length = self.downsampling_ratio

        if in_channels is not None:
            self.in_channels = in_channels

        if out_channels is not None:
            self.out_channels = out_channels

        self.bottleneck = bottleneck

        self.encoder = encoder

        self.decoder = decoder

        self.pretransform = pretransform

        self.freeze_pretransform = freeze_pretransform
        if self.pretransform is not None:
            if self.freeze_pretransform:
                for p in self.pretransform.parameters():
                    p.requires_grad = False
            else:
                for p in self.pretransform.parameters():
                    p.requires_grad = True

        self.soft_clip = soft_clip

        self.is_discrete = False

    def encode(self, audio, return_info=False, skip_pretransform=False, iterate_batch=False, return_pretransform = False, **kwargs):

        info = {}
        if self.pretransform is not None and not skip_pretransform: 
            if self.pretransform.enable_grad:
                if iterate_batch:
                    audios = []
                    for i in range(audio.shape[0]):
                        audios.append(self.pretransform.encode(audio[i:i+1]))
                    audio = torch.cat(audios, dim=0)
                else:
                    audio = self.pretransform.encode(audio)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        audios = []
                        for i in range(audio.shape[0]):
                            audios.append(self.pretransform.encode(audio[i:i+1]))
                        audio = torch.cat(audios, dim=0)
                    else:
                        audio = self.pretransform.encode(audio)
        if self.encoder is not None:
            if iterate_batch:
                latents = []
                for i in range(audio.shape[0]):
                    latents.append(self.encoder(audio[i:i+1], **kwargs))
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.encoder(audio, **kwargs)
        else:
            latents = audio
        if self.bottleneck is not None:
            # TODO: Add iterate batch logic, needs to merge the info dicts
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)

            info.update(bottleneck_info)
        
        if return_info and return_pretransform:
            return latents, info, audio
        elif return_info:
            return latents, info
        elif return_pretransform:
            return latents, audio
        else:
            return latents

    def decode(self, latents, iterate_batch=False, return_loss = False, **kwargs):
        if self.bottleneck is not None:
            if iterate_batch:
                decoded = []
                for i in range(latents.shape[0]):
                    decoded.append(self.bottleneck.decode(latents[i:i+1]))
                latents = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)
        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                decoded.append(self.decoder(latents[i:i+1], **kwargs))
            decoded = torch.cat(decoded, dim=0)
        else:
            if return_loss:
                decoded, loss = self.decoder(latents, **kwargs)
            else:
                decoded = self.decoder(latents, **kwargs)
        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    decodeds = []
                    for i in range(decoded.shape[0]):
                        decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                    decoded = torch.cat(decodeds, dim=0)
                else:
                    decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        decodeds = []
                        for i in range(latents.shape[0]):
                            decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                        decoded = torch.cat(decodeds, dim=0)
                    else:
                        decoded = self.pretransform.decode(decoded)

        if self.soft_clip:
            decoded = torch.tanh(decoded)
        if return_loss:
            return decoded, loss
        else:
            return decoded
          
    
    def preprocess_audio_for_encoder(self, audio, in_sr):
        '''
        Preprocess single audio tensor (Channels x Length) to be compatible with the encoder.
        If the model is mono, stereo audio will be converted to mono.
        Audio will be silence-padded to be a multiple of the model's downsampling ratio.
        Audio will be resampled to the model's sample rate. 
        The output will have batch size 1 and be shape (1 x Channels x Length)
        '''
        return self.preprocess_audio_list_for_encoder([audio], [in_sr])

    def preprocess_audio_list_for_encoder(self, audio_list, in_sr_list):
        '''
        Preprocess a [list] of audio (Channels x Length) into a batch tensor to be compatable with the encoder. 
        The audio in that list can be of different lengths and channels. 
        in_sr can be an integer or list. If it's an integer it will be assumed it is the input sample_rate for every audio.
        All audio will be resampled to the model's sample rate. 
        Audio will be silence-padded to the longest length, and further padded to be a multiple of the model's downsampling ratio. 
        If the model is mono, all audio will be converted to mono. 
        The output will be a tensor of shape (Batch x Channels x Length)
        '''
        batch_size = len(audio_list)
        if isinstance(in_sr_list, int):
            in_sr_list = [in_sr_list]*batch_size
        assert len(in_sr_list) == batch_size, "list of sample rates must be the same length of audio_list"
        new_audio = []
        max_length = 0
        # resample & find the max length
        for i in range(batch_size):
            audio = audio_list[i]
            in_sr = in_sr_list[i]
            if len(audio.shape) == 3 and audio.shape[0] == 1:
                # batchsize 1 was given by accident. Just squeeze it.
                audio = audio.squeeze(0)
            elif len(audio.shape) == 1:
                # Mono signal, channel dimension is missing, unsqueeze it in
                audio = audio.unsqueeze(0)
            assert len(audio.shape)==2, "Audio should be shape (Channels x Length) with no batch dimension" 
            # Resample audio
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate).to(audio.device)
                audio = resample_tf(audio)
            new_audio.append(audio)
            if audio.shape[-1] > max_length:
                max_length = audio.shape[-1]
        # Pad every audio to the same length, multiple of model's downsampling ratio
        padded_audio_length = max_length + (self.min_length - (max_length % self.min_length)) % self.min_length
        for i in range(batch_size):
            # Pad it & if necessary, mixdown/duplicate stereo/mono channels to support model
            new_audio[i] = prepare_audio(new_audio[i], in_sr=in_sr, target_sr=in_sr, target_length=padded_audio_length, 
                target_channels=self.in_channels, device=new_audio[i].device).squeeze(0)
        # convert to tensor 
        return torch.stack(new_audio) 

    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples) 
        # and therefore you likely could use the same values with decode_audio. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        samples_per_latent = int(self.downsampling_ratio)
        if not chunked or audio.shape[-1] < chunk_size * samples_per_latent:
            return self.encode(audio, **kwargs)

        # chunk_size/overlap are in latent units; scale to samples for slicing.
        chunk_size_samples = chunk_size * samples_per_latent
        hop_samples = (chunk_size - overlap) * samples_per_latent
        total_samples = audio.shape[-1]

        # Anchor the final chunk to the signal end (may overlap more than `overlap`).
        chunk_starts = list(range(0, total_samples - chunk_size_samples + 1, hop_samples))
        if chunk_starts[-1] != total_samples - chunk_size_samples:
            chunk_starts.append(total_samples - chunk_size_samples)

        encoded_chunks = [self.encode(audio[..., s:s + chunk_size_samples]) for s in chunk_starts]

        total_latents = total_samples // samples_per_latent
        half_overlap_latents = overlap // 2
        output = audio.new_zeros(*encoded_chunks[0].shape[:-1], total_latents)
        num_chunks = len(chunk_starts)

        # Trim half the overlap off inner edges (edges facing a neighbour chunk).
        for i, (start_sample, chunk) in enumerate(zip(chunk_starts, encoded_chunks)):
            is_first = i == 0
            is_last = i == num_chunks - 1
            out_start = (total_latents - chunk_size) if is_last else (start_sample // samples_per_latent)
            left  = 0 if is_first else half_overlap_latents
            right = chunk_size if is_last else chunk_size - half_overlap_latents
            output[..., out_start + left : out_start + right] = chunk[..., left:right]

        return output
    
    def decode_audio(self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Decode latents to audio. 
        If chunked is True, split the latents into chunks of a given maximum size chunk_size, with given overlap, both of which are measured in number of latents. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked audio and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked or latents.shape[-1] < chunk_size:
            return self.decode(latents, **kwargs)

        # chunk_size/overlap are in latent units â€” same as `latents`, no scaling.
        samples_per_latent = int(self.downsampling_ratio)
        hop_latents = chunk_size - overlap
        total_latents = latents.shape[-1]

        # Anchor the final chunk to the signal end (may overlap more than `overlap`).
        chunk_starts = list(range(0, total_latents - chunk_size + 1, hop_latents))
        if chunk_starts[-1] != total_latents - chunk_size:
            chunk_starts.append(total_latents - chunk_size)

        decoded_chunks = [self.decode(latents[..., s:s + chunk_size]) for s in chunk_starts]

        total_samples = total_latents * samples_per_latent
        chunk_size_samples = chunk_size * samples_per_latent
        half_overlap_samples = (overlap // 2) * samples_per_latent
        output = latents.new_zeros(*decoded_chunks[0].shape[:-1], total_samples)
        num_chunks = len(chunk_starts)

        # Trim half the overlap off inner edges (edges facing a neighbour chunk).
        for i, (start_latent, chunk) in enumerate(zip(chunk_starts, decoded_chunks)):
            is_first = i == 0
            is_last = i == num_chunks - 1
            out_start = (total_samples - chunk_size_samples) if is_last else (start_latent * samples_per_latent)
            left  = 0 if is_first else half_overlap_samples
            right = chunk_size_samples if is_last else chunk_size_samples - half_overlap_samples
            output[..., out_start + left : out_start + right] = chunk[..., left:right]

        return output
        
