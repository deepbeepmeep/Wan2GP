import logging
from omegaconf import DictConfig
from typing import List, Dict
import torch

from .object_manager import ObjectManager
from .kv_memory_store import KeyValueMemoryStore
from ..model.matanyone import MatAnyone
from ..model.utils.memory_utils import get_similarity, do_softmax

log = logging.getLogger()


class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """
    def __init__(self, cfg: DictConfig, object_manager: ObjectManager):
        self.object_manager = object_manager
        self.sensory_dim = cfg.model.sensory_dim
        self.top_k = cfg.top_k
        self.chunk_size = cfg.chunk_size

        self.save_aux = cfg.save_aux

        self.use_long_term = cfg.use_long_term
        self.count_long_term_usage = cfg.long_term.count_usage
        # subtract 1 because the first-frame is now counted as "permanent memory"
        # and is not counted towards max_mem_frames
        # but we want to keep the hyperparameters consistent as before for the same behavior
        if self.use_long_term:
            self.max_mem_frames = cfg.long_term.max_mem_frames - 1
            self.min_mem_frames = cfg.long_term.min_mem_frames - 1
            self.num_prototypes = cfg.long_term.num_prototypes
            self.max_long_tokens = cfg.long_term.max_num_tokens
            self.buffer_tokens = cfg.long_term.buffer_tokens
        else:
            self.max_mem_frames = cfg.max_mem_frames - 1

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        # The sensory memory is stored as a dictionary indexed by object ids
        # each of shape bs * C^h * H * W
        self.sensory = {}

        # a dictionary indexed by object ids, each of shape bs * T * Q * C
        self.obj_v = {}

        self.work_mem = KeyValueMemoryStore(save_selection=self.use_long_term,
                                            save_usage=self.use_long_term)
        if self.use_long_term:
            self.long_mem = KeyValueMemoryStore(save_usage=self.count_long_term_usage)

        self.config_stale = True
        self.engaged = False

    def update_config(self, cfg: DictConfig) -> None:
        self.config_stale = True
        self.top_k = cfg['top_k']

        assert self.use_long_term == cfg.use_long_term, 'cannot update this'
        assert self.count_long_term_usage == cfg.long_term.count_usage, 'cannot update this'

        self.use_long_term = cfg.use_long_term
        self.count_long_term_usage = cfg.long_term.count_usage
        if self.use_long_term:
            self.max_mem_frames = cfg.long_term.max_mem_frames - 1
            self.min_mem_frames = cfg.long_term.min_mem_frames - 1
            self.num_prototypes = cfg.long_term.num_prototypes
            self.max_long_tokens = cfg.long_term.max_num_tokens
            self.buffer_tokens = cfg.long_term.buffer_tokens
        else:
            self.max_mem_frames = cfg.max_mem_frames - 1

    def _readout(self, affinity, v, uncert_mask=None) -> torch.Tensor:
        # affinity: bs*N*HW
        # v: bs*C*N or bs*num_objects*C*N
        # returns bs*C*HW or bs*num_objects*C*HW
        if len(v.shape) == 3:
            # single object
            if uncert_mask is not None:
                return v @ affinity * uncert_mask
            else:
                return v @ affinity
        else:
            bs, num_objects, C, N = v.shape
            v = v.view(bs, num_objects * C, N)
            out = v @ affinity
            if uncert_mask is not None:
                uncert_mask = uncert_mask.flatten(start_dim=2).expand(-1, C, -1)
                out = out * uncert_mask
            return out.view(bs, num_objects, C, -1)

    def _get_mask_by_ids(self, mask: torch.Tensor, obj_ids: List[int]) -> torch.Tensor:
        # -1 because the mask does not contain the background channel
        return mask[:, [self.object_manager.find_tmp_by_id(obj) - 1 for obj in obj_ids]]

    def _get_sensory_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        return torch.stack([self.sensory[obj] for obj in obj_ids], dim=1)

    def _get_object_mem_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        return torch.stack([self.obj_v[obj] for obj in obj_ids], dim=1)

    def _get_visual_values_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        # All the values that the object ids refer to should have the same shape
        value = torch.stack([self.work_mem.value[obj] for obj in obj_ids], dim=1)
        if self.use_long_term and obj_ids[0] in self.long_mem.value:
            lt_value = torch.stack([self.long_mem.value[obj] for obj in obj_ids], dim=1)
            value = torch.cat([lt_value, value], dim=-1)

        return value
    
    def read_first_frame(self, last_msk_value, pix_feat: torch.Tensor, 
             last_mask: torch.Tensor, network: MatAnyone, uncert_output=None) -> Dict[int, torch.Tensor]:
        """
        Read from all memory stores and returns a single memory readout tensor for each object

        pix_feat: (1/2) x C x H x W
        query_key: (1/2) x C^k x H x W
        selection:  (1/2) x C^k x H x W
        last_mask: (1/2) x num_objects x H x W (at stride 16)
        return a dict of memory readouts, indexed by object indices. Each readout is C*H*W
        """
        h, w = pix_feat.shape[-2:]
        bs = pix_feat.shape[0]
        assert last_mask.shape[0] == bs

        """
        Compute affinity and perform readout
        """
        all_readout_mem = {}
        buckets = self.work_mem.buckets
        for bucket_id, bucket in buckets.items():

            if self.chunk_size < 1:
                object_chunks = [bucket]
            else:
                object_chunks = [
                    bucket[i:i + self.chunk_size] for i in range(0, len(bucket), self.chunk_size)
                ]

            for objects in object_chunks:
                this_sensory = self._get_sensory_by_ids(objects)
                this_last_mask = self._get_mask_by_ids(last_mask, objects)
                this_msk_value = self._get_visual_values_by_ids(objects)  # (1/2)*num_objects*C*N
                pixel_readout = network.pixel_fusion(pix_feat, last_msk_value, this_sensory,
                                                     this_last_mask)
                this_obj_mem = self._get_object_mem_by_ids(objects).unsqueeze(2)
                readout_memory, aux_features = network.readout_query(pixel_readout, this_obj_mem)
                for i, obj in enumerate(objects):
                    all_readout_mem[obj] = readout_memory[:, i]

                if self.save_aux:
                    aux_output = {
                        # 'sensory': this_sensory,
                        # 'pixel_readout': pixel_readout,
                        'q_logits': aux_features['logits'] if aux_features else None,
                        # 'q_weights': aux_features['q_weights'] if aux_features else None,
                        # 'p_weights': aux_features['p_weights'] if aux_features else None,
                        # 'attn_mask': aux_features['attn_mask'].float() if aux_features else None,
                    }
                    self.aux = aux_output

        return all_readout_mem

    def read(self, pix_feat: torch.Tensor, query_key: torch.Tensor, selection: torch.Tensor,
             last_mask: torch.Tensor, network: MatAnyone, uncert_output=None, last_msk_value=None, ti=None,
             last_pix_feat=None, last_pred_mask=None) -> Dict[int, torch.Tensor]:
        """
        Read from all memory stores and returns a single memory readout tensor for each object

        pix_feat: (1/2) x C x H x W
        query_key: (1/2) x C^k x H x W
        selection:  (1/2) x C^k x H x W
        last_mask: (1/2) x num_objects x H x W (at stride 16)
        return a dict of memory readouts, indexed by object indices. Each readout is C*H*W
        """
        h, w = pix_feat.shape[-2:]
        bs = pix_feat.shape[0]
        assert query_key.shape[0] == bs
        assert selection.shape[0] == bs
        assert last_mask.shape[0] == bs

        uncert_mask = uncert_output["mask"] if uncert_output is not None else None

        query_key = query_key.flatten(start_dim=2)  # bs*C^k*HW
        selection = selection.flatten(start_dim=2)  # bs*C^k*HW
        """
        Compute affinity and perform readout
        """
        all_readout_mem = {}
        buckets = self.work_mem.buckets
        for bucket_id, bucket in buckets.items():
            if self.use_long_term and self.long_mem.engaged(bucket_id):
                # Use long-term memory
                long_mem_size = self.long_mem.size(bucket_id)
                memory_key = torch.cat([self.long_mem.key[bucket_id], self.work_mem.key[bucket_id]],
                                       -1)
                shrinkage = torch.cat(
                    [self.long_mem.shrinkage[bucket_id], self.work_mem.shrinkage[bucket_id]], -1)

                similarity = get_similarity(memory_key, shrinkage, query_key, selection)
                affinity, usage = do_softmax(similarity,
                                             top_k=self.top_k,
                                             inplace=True,
                                             return_usage=True)
                """
                Record memory usage for working and long-term memory
                """
                # ignore the index return for long-term memory
                work_usage = usage[:, long_mem_size:]
                self.work_mem.update_bucket_usage(bucket_id, work_usage)

                if self.count_long_term_usage:
                    # ignore the index return for working memory
                    long_usage = usage[:, :long_mem_size]
                    self.long_mem.update_bucket_usage(bucket_id, long_usage)
            else:
                # no long-term memory
                memory_key = self.work_mem.key[bucket_id]
                shrinkage = self.work_mem.shrinkage[bucket_id]
                similarity = get_similarity(memory_key, shrinkage, query_key, selection, uncert_mask=uncert_mask)

                if self.use_long_term:
                    affinity, usage = do_softmax(similarity,
                                                 top_k=self.top_k,
                                                 inplace=True,
                                                 return_usage=True)
                    self.work_mem.update_bucket_usage(bucket_id, usage)
                else:
                    affinity = do_softmax(similarity, top_k=self.top_k, inplace=True)

            if self.chunk_size < 1:
                object_chunks = [bucket]
            else:
                object_chunks = [
                    bucket[i:i + self.chunk_size] for i in range(0, len(bucket), self.chunk_size)
                ]

            for objects in object_chunks:
                this_sensory = self._get_sensory_by_ids(objects)
                this_last_mask = self._get_mask_by_ids(last_mask, objects)
                this_msk_value = self._get_visual_values_by_ids(objects)  # (1/2)*num_objects*C*N
                visual_readout = self._readout(affinity,
                                               this_msk_value, uncert_mask).view(bs, len(objects), self.CV, h, w)
                
                uncert_output = network.pred_uncertainty(last_pix_feat, pix_feat, last_pred_mask, visual_readout[:,0]-last_msk_value[:,0])

                if uncert_output is not None:
                    uncert_prob = uncert_output["prob"].unsqueeze(1) # b n 1 h w
                    visual_readout = visual_readout*uncert_prob + last_msk_value*(1-uncert_prob)

                pixel_readout = network.pixel_fusion(pix_feat, visual_readout, this_sensory,
                                                     this_last_mask)
                this_obj_mem = self._get_object_mem_by_ids(objects).unsqueeze(2)
                readout_memory, aux_features = network.readout_query(pixel_readout, this_obj_mem)
                for i, obj in enumerate(objects):
                    all_readout_mem[obj] = readout_memory[:, i]

                if self.save_aux:
                    aux_output = {
                        # 'sensory': this_sensory,
                        # 'pixel_readout': pixel_readout,
                        'q_logits': aux_features['logits'] if aux_features else None,
                        # 'q_weights': aux_features['q_weights'] if aux_features else None,
                        # 'p_weights': aux_features['p_weights'] if aux_features else None,
                        # 'attn_mask': aux_features['attn_mask'].float() if aux_features else None,
                    }
                    self.aux = aux_output

        return all_readout_mem

    def add_memory(self,
                   key: torch.Tensor,
                   shrinkage: torch.Tensor,
                   msk_value: torch.Tensor,
                   obj_value: torch.Tensor,
                   objects: List[int],
                   selection: torch.Tensor = None,
                   *,
                   as_permanent: bool = False) -> None:
        # key: (1/2)*C*H*W
        # msk_value: (1/2)*num_objects*C*H*W
        # obj_value: (1/2)*num_objects*Q*C
        # objects contains a list of object ids corresponding to the objects in msk_value/obj_value
        bs = key.shape[0]
        assert shrinkage.shape[0] == bs
        assert msk_value.shape[0] == bs
        assert obj_value.shape[0] == bs

        self.engaged = True
        if self.H is None or self.config_stale:
            self.config_stale = False
            self.H, self.W = msk_value.shape[-2:]
            self.HW = self.H * self.W
            # convert from num. frames to num. tokens
            self.max_work_tokens = self.max_mem_frames * self.HW
            if self.use_long_term:
                self.min_work_tokens = self.min_mem_frames * self.HW

        # key:   bs*C*N
        # value: bs*num_objects*C*N
        key = key.flatten(start_dim=2)
        shrinkage = shrinkage.flatten(start_dim=2)
        self.CK = key.shape[1]

        msk_value = msk_value.flatten(start_dim=3)
        self.CV = msk_value.shape[2]

        if selection is not None:
            # not used in non-long-term mode
            selection = selection.flatten(start_dim=2)

        # insert object values into object memory
        for obj_id, obj in enumerate(objects):
            if obj in self.obj_v:
                """streaming average
                each self.obj_v[obj] is (1/2)*num_summaries*(embed_dim+1)
                first embed_dim keeps track of the sum of embeddings
                the last dim keeps the total count
                averaging in done inside the object transformer

                incoming obj_value is (1/2)*num_objects*num_summaries*(embed_dim+1)
                self.obj_v[obj] = torch.cat([self.obj_v[obj], obj_value[:, obj_id]], dim=0)
                """
                last_acc = self.obj_v[obj][:, :, -1]
                new_acc = last_acc + obj_value[:, obj_id, :, -1]

                self.obj_v[obj][:, :, :-1] = (self.obj_v[obj][:, :, :-1] +
                                              obj_value[:, obj_id, :, :-1])
                self.obj_v[obj][:, :, -1] = new_acc
            else:
                self.obj_v[obj] = obj_value[:, obj_id]

        # convert mask value tensor into a dict for insertion
        msk_values = {obj: msk_value[:, obj_id] for obj_id, obj in enumerate(objects)}
        self.work_mem.add(key,
                          msk_values,
                          shrinkage,
                          selection=selection,
                          as_permanent=as_permanent)

        for bucket_id in self.work_mem.buckets.keys():
            # long-term memory cleanup
            if self.use_long_term:
                # Do memory compressed if needed
                if self.work_mem.non_perm_size(bucket_id) >= self.max_work_tokens:
                    # Remove obsolete features if needed
                    if self.long_mem.non_perm_size(bucket_id) >= (self.max_long_tokens -
                                                         self.num_prototypes):
                        self.long_mem.remove_obsolete_features(
                            bucket_id,
                            self.max_long_tokens - self.num_prototypes - self.buffer_tokens)

                    self.compress_features(bucket_id)
            else:
                # FIFO
                self.work_mem.remove_old_memory(bucket_id, self.max_work_tokens)

    def purge_except(self, obj_keep_idx: List[int]) -> None:
        # purge certain objects from the memory except the one listed
        self.work_mem.purge_except(obj_keep_idx)
        if self.use_long_term and self.long_mem.engaged():
            self.long_mem.purge_except(obj_keep_idx)
        self.sensory = {k: v for k, v in self.sensory.items() if k in obj_keep_idx}

        if not self.work_mem.engaged():
            # everything is removed!
            self.engaged = False

    def compress_features(self, bucket_id: int) -> None:

        # perform memory consolidation
        prototype_key, prototype_value, prototype_shrinkage = self.consolidation(
            *self.work_mem.get_all_sliced(bucket_id, 0, -self.min_work_tokens))

        # remove consolidated working memory
        self.work_mem.sieve_by_range(bucket_id,
                                     0,
                                     -self.min_work_tokens,
                                     min_size=self.min_work_tokens)

        # add to long-term memory
        self.long_mem.add(prototype_key,
                          prototype_value,
                          prototype_shrinkage,
                          selection=None,
                          supposed_bucket_id=bucket_id)

    def consolidation(self, candidate_key: torch.Tensor, candidate_shrinkage: torch.Tensor,
                      candidate_selection: torch.Tensor, candidate_value: Dict[int, torch.Tensor],
                      usage: torch.Tensor) -> (torch.Tensor, Dict[int, torch.Tensor], torch.Tensor):
        # find the indices with max usage
        bs = candidate_key.shape[0]
        assert bs in [1, 2]

        prototype_key = []
        prototype_selection = []
        for bi in range(bs):
            _, max_usage_indices = torch.topk(usage[bi], k=self.num_prototypes, dim=-1, sorted=True)
            prototype_indices = max_usage_indices.flatten()
            prototype_key.append(candidate_key[bi, :, prototype_indices])
            prototype_selection.append(candidate_selection[bi, :, prototype_indices])
        prototype_key = torch.stack(prototype_key, dim=0)
        prototype_selection = torch.stack(prototype_selection, dim=0)
        """
        Potentiation step
        """
        similarity = get_similarity(candidate_key, candidate_shrinkage, prototype_key,
                                    prototype_selection)
        affinity = do_softmax(similarity)

        # readout the values
        prototype_value = {k: self._readout(affinity, v) for k, v in candidate_value.items()}

        # readout the shrinkage term
        prototype_shrinkage = self._readout(affinity, candidate_shrinkage)

        return prototype_key, prototype_value, prototype_shrinkage

    def initialize_sensory_if_needed(self, sample_key: torch.Tensor, ids: List[int]):
        for obj in ids:
            if obj not in self.sensory:
                # also initializes the sensory memory
                bs, _, h, w = sample_key.shape
                self.sensory[obj] = torch.zeros((bs, self.sensory_dim, h, w),
                                                device=sample_key.device)

    def update_sensory(self, sensory: torch.Tensor, ids: List[int]):
        # sensory: 1*num_objects*C*H*W
        for obj_id, obj in enumerate(ids):
            self.sensory[obj] = sensory[:, obj_id]

    def get_sensory(self, ids: List[int]):
        # returns (1/2)*num_objects*C*H*W
        return self._get_sensory_by_ids(ids)
    
    def clear_non_permanent_memory(self):
        self.work_mem.clear_non_permanent_memory()
        if self.use_long_term:
            self.long_mem.clear_non_permanent_memory()

    def clear_sensory_memory(self):
        self.sensory = {}
    
    def clear_work_mem(self):
        self.work_mem = KeyValueMemoryStore(save_selection=self.use_long_term,
                                            save_usage=self.use_long_term)
    
    def clear_obj_mem(self):
        self.obj_v = {}
