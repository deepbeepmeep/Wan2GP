import logging
from pathlib import Path

import einops
import torch
from omegaconf import OmegaConf
from timm.layers import trunc_normal_
from torch import nn

from .utils import check_if_file_exists_else_download
from .video_model_builder import VisionTransformer

FILE2URL = {
    # cfg
    'motionformer_224_16x4.yaml':
    'https://raw.githubusercontent.com/facebookresearch/Motionformer/bf43d50/configs/SSV2/motionformer_224_16x4.yaml',
    'joint_224_16x4.yaml':
    'https://raw.githubusercontent.com/facebookresearch/Motionformer/bf43d50/configs/SSV2/joint_224_16x4.yaml',
    'divided_224_16x4.yaml':
    'https://raw.githubusercontent.com/facebookresearch/Motionformer/bf43d50/configs/SSV2/divided_224_16x4.yaml',
    # ckpt
    'ssv2_motionformer_224_16x4.pyth':
    'https://dl.fbaipublicfiles.com/motionformer/ssv2_motionformer_224_16x4.pyth',
    'ssv2_joint_224_16x4.pyth':
    'https://dl.fbaipublicfiles.com/motionformer/ssv2_joint_224_16x4.pyth',
    'ssv2_divided_224_16x4.pyth':
    'https://dl.fbaipublicfiles.com/motionformer/ssv2_divided_224_16x4.pyth',
}


class MotionFormer(VisionTransformer):
    ''' This class serves three puposes:
            1. Renames the class to MotionFormer.
            2. Downloads the cfg from the original repo and patches it if needed.
            3. Takes care of feature extraction by redefining .forward()
                - if `extract_features=True` and `factorize_space_time=False`,
                    the output is of shape (B, T, D) where T = 1 + (224 // 16) * (224 // 16) * 8
                - if `extract_features=True` and `factorize_space_time=True`, the output is of shape (B*S, D)
                    and spatial and temporal transformer encoder layers are used.
                - if `extract_features=True` and `factorize_space_time=True` as well as `add_global_repr=True`
                    the output is of shape (B, D) and spatial and temporal transformer encoder layers
                    are used as well as the global representation is extracted from segments (extra pos emb
                    is added).
    '''

    def __init__(
        self,
        extract_features: bool = False,
        ckpt_path: str = None,
        factorize_space_time: bool = None,
        agg_space_module: str = None,
        agg_time_module: str = None,
        add_global_repr: bool = True,
        agg_segments_module: str = None,
        max_segments: int = None,
    ):
        self.extract_features = extract_features
        self.ckpt_path = ckpt_path
        self.factorize_space_time = factorize_space_time

        if self.ckpt_path is not None:
            check_if_file_exists_else_download(self.ckpt_path, FILE2URL)
            ckpt = torch.load(self.ckpt_path, map_location='cpu')
            mformer_ckpt2cfg = {
                'ssv2_motionformer_224_16x4.pyth': 'motionformer_224_16x4.yaml',
                'ssv2_joint_224_16x4.pyth': 'joint_224_16x4.yaml',
                'ssv2_divided_224_16x4.pyth': 'divided_224_16x4.yaml',
            }
            # init from motionformer ckpt or from our Stage I ckpt
            # depending on whether the feat extractor was pre-trained on AVCLIPMoCo or not, we need to
            # load the state dict differently
            was_pt_on_avclip = self.ckpt_path.endswith(
                '.pt')  # checks if it is a stage I ckpt (FIXME: a bit generic)
            if self.ckpt_path.endswith(tuple(mformer_ckpt2cfg.keys())):
                cfg_fname = mformer_ckpt2cfg[Path(self.ckpt_path).name]
            elif was_pt_on_avclip:
                # TODO: this is a hack, we should be able to get the cfg from the ckpt (earlier ckpt didn't have it)
                s1_cfg = ckpt.get('args', None)  # Stage I cfg
                if s1_cfg is not None:
                    s1_vfeat_extractor_ckpt_path = s1_cfg.model.params.vfeat_extractor.params.ckpt_path
                    # if the stage I ckpt was initialized from a motionformer ckpt or train from scratch
                    if s1_vfeat_extractor_ckpt_path is not None:
                        cfg_fname = mformer_ckpt2cfg[Path(s1_vfeat_extractor_ckpt_path).name]
                    else:
                        cfg_fname = 'divided_224_16x4.yaml'
                else:
                    cfg_fname = 'divided_224_16x4.yaml'
            else:
                raise ValueError(f'ckpt_path {self.ckpt_path} is not supported.')
        else:
            was_pt_on_avclip = False
            cfg_fname = 'divided_224_16x4.yaml'
            # logging.info(f'No ckpt_path provided, using {cfg_fname} config.')

        if cfg_fname in ['motionformer_224_16x4.yaml', 'divided_224_16x4.yaml']:
            pos_emb_type = 'separate'
        elif cfg_fname == 'joint_224_16x4.yaml':
            pos_emb_type = 'joint'

        self.mformer_cfg_path = Path(__file__).absolute().parent / cfg_fname

        check_if_file_exists_else_download(self.mformer_cfg_path, FILE2URL)
        mformer_cfg = OmegaConf.load(self.mformer_cfg_path)
        logging.info(f'Loading MotionFormer config from {self.mformer_cfg_path.absolute()}')

        # patch the cfg (from the default cfg defined in the repo `Motionformer/slowfast/config/defaults.py`)
        mformer_cfg.VIT.ATTN_DROPOUT = 0.0
        mformer_cfg.VIT.POS_EMBED = pos_emb_type
        mformer_cfg.VIT.USE_ORIGINAL_TRAJ_ATTN_CODE = True
        mformer_cfg.VIT.APPROX_ATTN_TYPE = 'none'  # guessing
        mformer_cfg.VIT.APPROX_ATTN_DIM = 64  # from ckpt['cfg']

        # finally init VisionTransformer with the cfg
        super().__init__(mformer_cfg)

        # load the ckpt now if ckpt is provided and not from AVCLIPMoCo-pretrained ckpt
        if (self.ckpt_path is not None) and (not was_pt_on_avclip):
            _ckpt_load_status = self.load_state_dict(ckpt['model_state'], strict=False)
            if len(_ckpt_load_status.missing_keys) > 0 or len(
                    _ckpt_load_status.unexpected_keys) > 0:
                logging.warning(f'Loading exact vfeat_extractor ckpt from {self.ckpt_path} failed.' \
                                f'Missing keys: {_ckpt_load_status.missing_keys}, ' \
                                f'Unexpected keys: {_ckpt_load_status.unexpected_keys}')
            else:
                logging.info(f'Loading vfeat_extractor ckpt from {self.ckpt_path} succeeded.')

        if self.extract_features:
            assert isinstance(self.norm,
                              nn.LayerNorm), 'early x[:, 1:, :] may not be safe for per-tr weights'
            # pre-logits are Sequential(nn.Linear(emb, emd), act) and `act` is tanh but see the logger
            self.pre_logits = nn.Identity()
            # we don't need the classification head (saving memory)
            self.head = nn.Identity()
            self.head_drop = nn.Identity()
            # avoiding code duplication (used only if agg_*_module is TransformerEncoderLayer)
            transf_enc_layer_kwargs = dict(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                activation=nn.GELU(),
                batch_first=True,
                dim_feedforward=self.mlp_ratio * self.embed_dim,
                dropout=self.drop_rate,
                layer_norm_eps=1e-6,
                norm_first=True,
            )
            # define adapters if needed
            if self.factorize_space_time:
                if agg_space_module == 'TransformerEncoderLayer':
                    self.spatial_attn_agg = SpatialTransformerEncoderLayer(
                        **transf_enc_layer_kwargs)
                elif agg_space_module == 'AveragePooling':
                    self.spatial_attn_agg = AveragePooling(avg_pattern='BS D t h w -> BS D t',
                                                           then_permute_pattern='BS D t -> BS t D')
                if agg_time_module == 'TransformerEncoderLayer':
                    self.temp_attn_agg = TemporalTransformerEncoderLayer(**transf_enc_layer_kwargs)
                elif agg_time_module == 'AveragePooling':
                    self.temp_attn_agg = AveragePooling(avg_pattern='BS t D -> BS D')
                elif 'Identity' in agg_time_module:
                    self.temp_attn_agg = nn.Identity()
            # define a global aggregation layer (aggregarate over segments)
            self.add_global_repr = add_global_repr
            if add_global_repr:
                if agg_segments_module == 'TransformerEncoderLayer':
                    # we can reuse the same layer as for temporal factorization (B, dim_to_agg, D) -> (B, D)
                    # we need to add pos emb (PE) because previously we added the same PE for each segment
                    pos_max_len = max_segments if max_segments is not None else 16  # 16 = 10sec//0.64sec + 1
                    self.global_attn_agg = TemporalTransformerEncoderLayer(
                        add_pos_emb=True,
                        pos_emb_drop=mformer_cfg.VIT.POS_DROPOUT,
                        pos_max_len=pos_max_len,
                        **transf_enc_layer_kwargs)
                elif agg_segments_module == 'AveragePooling':
                    self.global_attn_agg = AveragePooling(avg_pattern='B S D -> B D')

        if was_pt_on_avclip:
            # we need to filter out the state_dict of the AVCLIP model (has both A and V extractors)
            # and keep only the state_dict of the feat extractor
            ckpt_weights = dict()
            for k, v in ckpt['state_dict'].items():
                if k.startswith(('module.v_encoder.', 'v_encoder.')):
                    k = k.replace('module.', '').replace('v_encoder.', '')
                    ckpt_weights[k] = v
            _load_status = self.load_state_dict(ckpt_weights, strict=False)
            if len(_load_status.missing_keys) > 0 or len(_load_status.unexpected_keys) > 0:
                logging.warning(f'Loading exact vfeat_extractor ckpt from {self.ckpt_path} failed. \n' \
                                f'Missing keys ({len(_load_status.missing_keys)}): ' \
                                f'{_load_status.missing_keys}, \n' \
                                f'Unexpected keys ({len(_load_status.unexpected_keys)}): ' \
                                f'{_load_status.unexpected_keys} \n' \
                                f'temp_attn_agg are expected to be missing if ckpt was pt contrastively.')
            else:
                logging.info(f'Loading vfeat_extractor ckpt from {self.ckpt_path} succeeded.')

        # patch_embed is not used in MotionFormer, only patch_embed_3d, because cfg.VIT.PATCH_SIZE_TEMP > 1
        # but it used to calculate the number of patches, so we need to set keep it
        self.patch_embed.requires_grad_(False)

    def forward(self, x):
        '''
        x is of shape (B, S, C, T, H, W) where S is the number of segments.
        '''
        # Batch, Segments, Channels, T=frames, Height, Width
        B, S, C, T, H, W = x.shape
        # Motionformer expects a tensor of shape (1, B, C, T, H, W).
        # The first dimension (1) is a dummy dimension to make the input tensor and won't be used:
        # see `video_model_builder.video_input`.
        # x = x.unsqueeze(0)  # (1, B, S, C, T, H, W)

        orig_shape = (B, S, C, T, H, W)
        x = x.view(B * S, C, T, H, W)  # flatten batch and segments
        x = self.forward_segments(x, orig_shape=orig_shape)
        # unpack the segments (using rest dimensions to support different shapes e.g. (BS, D) or (BS, t, D))
        x = x.view(B, S, *x.shape[1:])
        # x is now of shape (B*S, D) or (B*S, t, D) if `self.temp_attn_agg` is `Identity`

        return x  # x is (B, S, ...)

    def forward_segments(self, x, orig_shape: tuple) -> torch.Tensor:
        '''x is of shape (1, BS, C, T, H, W) where S is the number of segments.'''
        x, x_mask = self.forward_features(x)

        assert self.extract_features

        # (BS, T, D) where T = 1 + (224 // 16) * (224 // 16) * 8
        x = x[:,
              1:, :]  # without the CLS token for efficiency (should be safe for LayerNorm and FC)
        x = self.norm(x)
        x = self.pre_logits(x)
        if self.factorize_space_time:
            x = self.restore_spatio_temp_dims(x, orig_shape)  # (B*S, D, t, h, w) <- (B*S, t*h*w, D)

            x = self.spatial_attn_agg(x, x_mask)  # (B*S, t, D)
            x = self.temp_attn_agg(
                x)  # (B*S, D) or (BS, t, D) if `self.temp_attn_agg` is `Identity`

        return x

    def restore_spatio_temp_dims(self, feats: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        '''
            feats are of shape (B*S, T, D) where T = 1 + (224 // 16) * (224 // 16) * 8
            Our goal is to make them of shape (B*S, t, h, w, D) where h, w are the spatial dimensions.
            From `self.patch_embed_3d`, it follows that we could reshape feats with:
                `feats.transpose(1, 2).view(B*S, D, t, h, w)`
        '''
        B, S, C, T, H, W = orig_shape
        D = self.embed_dim

        # num patches in each dimension
        t = T // self.patch_embed_3d.z_block_size
        h = self.patch_embed_3d.height
        w = self.patch_embed_3d.width

        feats = feats.permute(0, 2, 1)  # (B*S, D, T)
        feats = feats.view(B * S, D, t, h, w)  # (B*S, D, t, h, w)

        return feats


class BaseEncoderLayer(nn.TransformerEncoderLayer):
    '''
        This is a wrapper around nn.TransformerEncoderLayer that adds a CLS token
        to the sequence and outputs the CLS token's representation.
        This base class parents both SpatialEncoderLayer and TemporalEncoderLayer for the RGB stream
        and the FrequencyEncoderLayer and TemporalEncoderLayer for the audio stream stream.
        We also, optionally, add a positional embedding to the input sequence which
        allows to reuse it for global aggregation (of segments) for both streams.
    '''

    def __init__(self,
                 add_pos_emb: bool = False,
                 pos_emb_drop: float = None,
                 pos_max_len: int = None,
                 *args_transformer_enc,
                 **kwargs_transformer_enc):
        super().__init__(*args_transformer_enc, **kwargs_transformer_enc)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.self_attn.embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        # add positional embedding
        self.add_pos_emb = add_pos_emb
        if add_pos_emb:
            self.pos_max_len = 1 + pos_max_len  # +1 (for CLS)
            self.pos_emb = nn.Parameter(torch.zeros(1, self.pos_max_len, self.self_attn.embed_dim))
            self.pos_drop = nn.Dropout(pos_emb_drop)
            trunc_normal_(self.pos_emb, std=.02)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        ''' x is of shape (B, N, D); if provided x_mask is of shape (B, N)'''
        batch_dim = x.shape[0]

        # add CLS token
        cls_tokens = self.cls_token.expand(batch_dim, -1, -1)  # expanding to match batch dimension
        x = torch.cat((cls_tokens, x), dim=-2)  # (batch_dim, 1+seq_len, D)
        if x_mask is not None:
            cls_mask = torch.ones((batch_dim, 1), dtype=torch.bool,
                                  device=x_mask.device)  # 1=keep; 0=mask
            x_mask_w_cls = torch.cat((cls_mask, x_mask), dim=-1)  # (batch_dim, 1+seq_len)
            B, N = x_mask_w_cls.shape
            # torch expects (N, N) or (B*num_heads, N, N) mask (sadness ahead); torch masks
            x_mask_w_cls = x_mask_w_cls.reshape(B, 1, 1, N)\
                                       .expand(-1, self.self_attn.num_heads, N, -1)\
                                       .reshape(B * self.self_attn.num_heads, N, N)
            assert x_mask_w_cls.dtype == x_mask_w_cls.bool().dtype, 'x_mask_w_cls.dtype != bool'
            x_mask_w_cls = ~x_mask_w_cls  # invert mask (1=mask)
        else:
            x_mask_w_cls = None

        # add positional embedding
        if self.add_pos_emb:
            seq_len = x.shape[
                1]  # (don't even think about moving it before the CLS token concatenation)
            assert seq_len <= self.pos_max_len, f'Seq len ({seq_len}) > pos_max_len ({self.pos_max_len})'
            x = x + self.pos_emb[:, :seq_len, :]
            x = self.pos_drop(x)

        # apply encoder layer (calls nn.TransformerEncoderLayer.forward);
        x = super().forward(src=x, src_mask=x_mask_w_cls)  # (batch_dim, 1+seq_len, D)

        # CLS token is expected to hold spatial information for each frame
        x = x[:, 0, :]  # (batch_dim, D)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', 'pos_emb'}


class SpatialTransformerEncoderLayer(BaseEncoderLayer):
    ''' Aggregates spatial dimensions by applying attention individually to each frame. '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        ''' x is of shape (B*S, D, t, h, w) where S is the number of segments.
            if specified x_mask (B*S, t, h, w), 0=masked, 1=kept
            Returns a tensor of shape (B*S, t, D) pooling spatial information for each frame. '''
        BS, D, t, h, w = x.shape

        # time as a batch dimension and flatten spatial dimensions as sequence
        x = einops.rearrange(x, 'BS D t h w -> (BS t) (h w) D')
        # similar to mask
        if x_mask is not None:
            x_mask = einops.rearrange(x_mask, 'BS t h w -> (BS t) (h w)')

        # apply encoder layer (BaseEncoderLayer.forward) - it will add CLS token and output its representation
        x = super().forward(x=x, x_mask=x_mask)  # (B*S*t, D)

        # reshape back to (B*S, t, D)
        x = einops.rearrange(x, '(BS t) D -> BS t D', BS=BS, t=t)

        # (B*S, t, D)
        return x


class TemporalTransformerEncoderLayer(BaseEncoderLayer):
    ''' Aggregates temporal dimension with attention. Also used with pos emb as global aggregation
    in both streams. '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        ''' x is of shape (B*S, t, D) where S is the number of segments.
            Returns a tensor of shape (B*S, D) pooling temporal information. '''
        BS, t, D = x.shape

        # apply encoder layer (BaseEncoderLayer.forward) - it will add CLS token and output its representation
        x = super().forward(x)  # (B*S, D)

        return x  # (B*S, D)


class AveragePooling(nn.Module):

    def __init__(self, avg_pattern: str, then_permute_pattern: str = None) -> None:
        ''' patterns are e.g. "bs t d -> bs d" '''
        super().__init__()
        # TODO: need to register them as buffers (but fails because these are strings)
        self.reduce_fn = 'mean'
        self.avg_pattern = avg_pattern
        self.then_permute_pattern = then_permute_pattern

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        x = einops.reduce(x, self.avg_pattern, self.reduce_fn)
        if self.then_permute_pattern is not None:
            x = einops.rearrange(x, self.then_permute_pattern)
        return x
