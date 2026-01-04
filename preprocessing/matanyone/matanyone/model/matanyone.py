from typing import List, Dict, Iterable
import logging
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from huggingface_hub import PyTorchModelHubMixin

from .big_modules import PixelEncoder, UncertPred, KeyProjection, MaskEncoder, PixelFeatureFuser, MaskDecoder
from .aux_modules import AuxComputer
from .utils.memory_utils import get_affinity, readout
from .transformer.object_transformer import QueryTransformer
from .transformer.object_summarizer import ObjectSummarizer
from ...utils.tensor_utils import aggregate

log = logging.getLogger()
class MatAnyone(nn.Module,
                PyTorchModelHubMixin,
                library_name="matanyone",
                repo_url="https://github.com/pq-yang/MatAnyone",
                coders={
                    DictConfig: (
                        lambda x: OmegaConf.to_container(x),
                        lambda data: OmegaConf.create(data),
                    )
                },
        ):

    def __init__(self, cfg: DictConfig, *, single_object=False):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model
        self.ms_dims = model_cfg.pixel_encoder.ms_dims
        self.key_dim = model_cfg.key_dim
        self.value_dim = model_cfg.value_dim
        self.sensory_dim = model_cfg.sensory_dim
        self.pixel_dim = model_cfg.pixel_dim
        self.embed_dim = model_cfg.embed_dim
        self.single_object = single_object

        log.info(f'Single object: {self.single_object}')

        self.pixel_encoder = PixelEncoder(model_cfg)
        self.pix_feat_proj = nn.Conv2d(self.ms_dims[0], self.pixel_dim, kernel_size=1)
        self.key_proj = KeyProjection(model_cfg)
        self.mask_encoder = MaskEncoder(model_cfg, single_object=single_object)
        self.mask_decoder = MaskDecoder(model_cfg)
        self.pixel_fuser = PixelFeatureFuser(model_cfg, single_object=single_object)
        self.object_transformer = QueryTransformer(model_cfg)
        self.object_summarizer = ObjectSummarizer(model_cfg)
        self.aux_computer = AuxComputer(cfg)
        self.temp_sparity = UncertPred(model_cfg)

        self.register_buffer("pixel_mean", torch.Tensor(model_cfg.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(model_cfg.pixel_std).view(-1, 1, 1), False)

    def _get_others(self, masks: torch.Tensor) -> torch.Tensor:
        # for each object, return the sum of masks of all other objects
        if self.single_object:
            return None

        num_objects = masks.shape[1]
        if num_objects >= 1:
            others = (masks.sum(dim=1, keepdim=True) - masks).clamp(0, 1)
        else:
            others = torch.zeros_like(masks)
        return others

    def pred_uncertainty(self, last_pix_feat: torch.Tensor, cur_pix_feat: torch.Tensor, last_mask: torch.Tensor, mem_val_diff:torch.Tensor):
        logits = self.temp_sparity(last_frame_feat=last_pix_feat,
                                cur_frame_feat=cur_pix_feat,
                                last_mask=last_mask,
                                mem_val_diff=mem_val_diff)

        prob = torch.sigmoid(logits)
        mask = (prob > 0) + 0

        uncert_output = {"logits": logits,
                     "prob": prob,
                     "mask": mask}

        return uncert_output

    def encode_image(self, image: torch.Tensor, seq_length=None, last_feats=None) -> (Iterable[torch.Tensor], torch.Tensor): # type: ignore
        image = (image - self.pixel_mean) / self.pixel_std
        ms_image_feat = self.pixel_encoder(image, seq_length) # f16, f8, f4, f2, f1
        return ms_image_feat, self.pix_feat_proj(ms_image_feat[0])

    def encode_mask(
            self,
            image: torch.Tensor,
            ms_features: List[torch.Tensor],
            sensory: torch.Tensor,
            masks: torch.Tensor,
            *,
            deep_update: bool = True,
            chunk_size: int = -1,
            need_weights: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        image = (image - self.pixel_mean) / self.pixel_std
        others = self._get_others(masks)
        mask_value, new_sensory = self.mask_encoder(image,
                                                    ms_features,
                                                    sensory,
                                                    masks,
                                                    others,
                                                    deep_update=deep_update,
                                                    chunk_size=chunk_size)
        object_summaries, object_logits = self.object_summarizer(masks, mask_value, need_weights)
        return mask_value, new_sensory, object_summaries, object_logits

    def transform_key(self,
                      final_pix_feat: torch.Tensor,
                      *,
                      need_sk: bool = True,
                      need_ek: bool = True) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        key, shrinkage, selection = self.key_proj(final_pix_feat, need_s=need_sk, need_e=need_ek)
        return key, shrinkage, selection

    # Used in training only.
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key: torch.Tensor, query_selection: torch.Tensor,
                    memory_key: torch.Tensor, memory_shrinkage: torch.Tensor,
                    msk_value: torch.Tensor, obj_memory: torch.Tensor, pix_feat: torch.Tensor,
                    sensory: torch.Tensor, last_mask: torch.Tensor,
                    selector: torch.Tensor, uncert_output=None, seg_pass=False,
                    last_pix_feat=None, last_pred_mask=None) -> (torch.Tensor, Dict[str, torch.Tensor]):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        msk_value       : B * num_objects * CV * T * H * W
        obj_memory      : B * num_objects * T * num_summaries * C
        pixel_feature   : B * C * H * W
        """
        batch_size, num_objects = msk_value.shape[:2]

        uncert_mask = uncert_output["mask"] if uncert_output is not None else None

        # read using visual attention
        with torch.cuda.amp.autocast(enabled=False):
            affinity = get_affinity(memory_key.float(), memory_shrinkage.float(), query_key.float(),
                                    query_selection.float(), uncert_mask=uncert_mask)

            msk_value = msk_value.flatten(start_dim=1, end_dim=2).float()

            # B * (num_objects*CV) * H * W
            pixel_readout = readout(affinity, msk_value, uncert_mask)
            pixel_readout = pixel_readout.view(batch_size, num_objects, self.value_dim,
                                            *pixel_readout.shape[-2:])
            
            uncert_output = self.pred_uncertainty(last_pix_feat, pix_feat, last_pred_mask, pixel_readout[:,0]-msk_value[:,:,-1])
            uncert_prob = uncert_output["prob"].unsqueeze(1) # b n 1 h w
            pixel_readout = pixel_readout*uncert_prob + msk_value[:,:,-1].unsqueeze(1)*(1-uncert_prob)
                
        pixel_readout = self.pixel_fusion(pix_feat, pixel_readout, sensory, last_mask)


        # read from query transformer
        mem_readout, aux_features = self.readout_query(pixel_readout, obj_memory, selector=selector, seg_pass=seg_pass)

        aux_output = {
            'sensory': sensory,
            'q_logits': aux_features['logits'] if aux_features else None,
            'attn_mask': aux_features['attn_mask'] if aux_features else None,
        }

        return mem_readout, aux_output, uncert_output
    
    def read_first_frame_memory(self, pixel_readout,
                    obj_memory: torch.Tensor, pix_feat: torch.Tensor,
                    sensory: torch.Tensor, last_mask: torch.Tensor,
                    selector: torch.Tensor, seg_pass=False) -> (torch.Tensor, Dict[str, torch.Tensor]):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        msk_value       : B * num_objects * CV * T * H * W
        obj_memory      : B * num_objects * T * num_summaries * C
        pixel_feature   : B * C * H * W
        """
                
        pixel_readout = self.pixel_fusion(pix_feat, pixel_readout, sensory, last_mask)

        # read from query transformer
        mem_readout, aux_features = self.readout_query(pixel_readout, obj_memory, selector=selector, seg_pass=seg_pass)

        aux_output = {
            'sensory': sensory,
            'q_logits': aux_features['logits'] if aux_features else None,
            'attn_mask': aux_features['attn_mask'] if aux_features else None,
        }

        return mem_readout, aux_output

    def pixel_fusion(self,
                     pix_feat: torch.Tensor,
                     pixel: torch.Tensor,
                     sensory: torch.Tensor,
                     last_mask: torch.Tensor,
                     *,
                     chunk_size: int = -1) -> torch.Tensor:
        last_mask = F.interpolate(last_mask, size=sensory.shape[-2:], mode='area')
        last_others = self._get_others(last_mask)
        fused = self.pixel_fuser(pix_feat,
                                 pixel,
                                 sensory,
                                 last_mask,
                                 last_others,
                                 chunk_size=chunk_size)
        return fused

    def readout_query(self,
                      pixel_readout,
                      obj_memory,
                      *,
                      selector=None,
                      need_weights=False,
                      seg_pass=False) -> (torch.Tensor, Dict[str, torch.Tensor]):
        return self.object_transformer(pixel_readout,
                                       obj_memory,
                                       selector=selector,
                                       need_weights=need_weights,
                                       seg_pass=seg_pass)

    def segment(self,
                ms_image_feat: List[torch.Tensor],
                memory_readout: torch.Tensor,
                sensory: torch.Tensor,
                *,
                selector: bool = None,
                chunk_size: int = -1,
                update_sensory: bool = True,
                seg_pass: bool = False,
                clamp_mat: bool = True,
                last_mask=None,
                sigmoid_residual=False,
                seg_mat=False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        multi_scale_features is from the key encoder for skip-connection
        memory_readout is from working/long-term memory
        sensory is the sensory memory
        last_mask is the mask from the last frame, supplementing sensory memory
        selector is 1 if an object exists, and 0 otherwise. We use it to filter padded objects
            during training.
        """
        #### use mat head for seg data
        if seg_mat:
            assert seg_pass
            seg_pass = False
        ####
        sensory, logits = self.mask_decoder(ms_image_feat,
                                        memory_readout,
                                        sensory,
                                        chunk_size=chunk_size,
                                        update_sensory=update_sensory,
                                        seg_pass = seg_pass,
                                        last_mask=last_mask,
                                        sigmoid_residual=sigmoid_residual)
        if seg_pass:
            prob = torch.sigmoid(logits)
            if selector is not None:
                prob = prob * selector

            # Softmax over all objects[]
            logits = aggregate(prob, dim=1)
            prob = F.softmax(logits, dim=1)
        else:
            if clamp_mat:
                logits = logits.clamp(0.0, 1.0)
            logits = torch.cat([torch.prod(1 - logits, dim=1, keepdim=True), logits], 1)
            prob = logits
        
        return sensory, logits, prob

    def compute_aux(self, pix_feat: torch.Tensor, aux_inputs: Dict[str, torch.Tensor],
                    selector: torch.Tensor, seg_pass=False) -> Dict[str, torch.Tensor]:
        return self.aux_computer(pix_feat, aux_inputs, selector, seg_pass=seg_pass)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_weights(self, src_dict, init_as_zero_if_needed=False) -> None:
        if not self.single_object:
            # Map single-object weight to multi-object weight (4->5 out channels in conv1)
            for k in list(src_dict.keys()):
                if k == 'mask_encoder.conv1.weight':
                    if src_dict[k].shape[1] == 4:
                        log.info(f'Converting {k} from single object to multiple objects.')
                        pads = torch.zeros((64, 1, 7, 7), device=src_dict[k].device)
                        if not init_as_zero_if_needed:
                            nn.init.orthogonal_(pads)
                            log.info(f'Randomly initialized padding for {k}.')
                        else:
                            log.info(f'Zero-initialized padding for {k}.')
                        src_dict[k] = torch.cat([src_dict[k], pads], 1)
                elif k == 'pixel_fuser.sensory_compress.weight':
                    if src_dict[k].shape[1] == self.sensory_dim + 1:
                        log.info(f'Converting {k} from single object to multiple objects.')
                        pads = torch.zeros((self.value_dim, 1, 1, 1), device=src_dict[k].device)
                        if not init_as_zero_if_needed:
                            nn.init.orthogonal_(pads)
                            log.info(f'Randomly initialized padding for {k}.')
                        else:
                            log.info(f'Zero-initialized padding for {k}.')
                        src_dict[k] = torch.cat([src_dict[k], pads], 1)
        elif self.single_object:
            """
            If the model is multiple-object and we are training in single-object, 
            we strip the last channel of conv1.
            This is not supposed to happen in standard training except when users are trying to
            finetune a trained model with single object datasets.
            """
            if src_dict['mask_encoder.conv1.weight'].shape[1] == 5:
                log.warning('Converting mask_encoder.conv1.weight from multiple objects to single object.'
                            'This is not supposed to happen in standard training.')
                src_dict['mask_encoder.conv1.weight'] = src_dict['mask_encoder.conv1.weight'][:, :-1]
                src_dict['pixel_fuser.sensory_compress.weight'] = src_dict['pixel_fuser.sensory_compress.weight'][:, :-1]

        for k in src_dict:
            if k not in self.state_dict():
                log.info(f'Key {k} found in src_dict but not in self.state_dict()!!!')
        for k in self.state_dict():
            if k not in src_dict:
                log.info(f'Key {k} found in self.state_dict() but not in src_dict!!!')
                
        self.load_state_dict(src_dict, strict=False)

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device
