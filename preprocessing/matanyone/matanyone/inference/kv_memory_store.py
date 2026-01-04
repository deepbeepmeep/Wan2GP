from typing import Dict, List, Optional, Literal
from collections import defaultdict
import torch


def _add_last_dim(dictionary, key, new_value, prepend=False):
    # append/prepend a new value to the last dimension of a tensor in a dictionary
    # if the key does not exist, put the new value in
    # append by default
    if key in dictionary:
        dictionary[key] = torch.cat([dictionary[key], new_value], -1)
    else:
        dictionary[key] = new_value


class KeyValueMemoryStore:
    """
    Works for key/value pairs type storage
    e.g., working and long-term memory
    """
    def __init__(self, save_selection: bool = False, save_usage: bool = False):
        """
        We store keys and values of objects that first appear in the same frame in a bucket.
        Each bucket contains a set of object ids.
        Each bucket is associated with a single key tensor
            and a dictionary of value tensors indexed by object id.

        The keys and values are stored as the concatenation of a permanent part and a temporary part.
        """
        self.save_selection = save_selection
        self.save_usage = save_usage

        self.global_bucket_id = 0  # does not reduce even if buckets are removed
        self.buckets: Dict[int, List[int]] = {}  # indexed by bucket id
        self.k: Dict[int, torch.Tensor] = {}  # indexed by bucket id
        self.v: Dict[int, torch.Tensor] = {}  # indexed by object id

        # indexed by bucket id; the end point of permanent memory
        self.perm_end_pt: Dict[int, int] = defaultdict(int)

        # shrinkage and selection are just like the keys
        self.s = {}
        if self.save_selection:
            self.e = {}  # does not contain the permanent memory part

        # usage
        if self.save_usage:
            self.use_cnt = {}  # indexed by bucket id, does not contain the permanent memory part
            self.life_cnt = {}  # indexed by bucket id, does not contain the permanent memory part

    def add(self,
            key: torch.Tensor,
            values: Dict[int, torch.Tensor],
            shrinkage: torch.Tensor,
            selection: torch.Tensor,
            supposed_bucket_id: int = -1,
            as_permanent: Literal['no', 'first', 'all'] = 'no') -> None:
        """
        key: (1/2)*C*N
        values: dict of values ((1/2)*C*N), object ids are used as keys
        shrinkage: (1/2)*1*N
        selection: (1/2)*C*N

        supposed_bucket_id: used to sync the bucket id between working and long-term memory
        if provided, the input should all be in a single bucket indexed by this id
        as_permanent: whether to store the input as permanent memory
            'no': don't
            'first': only store it as permanent memory if the bucket is empty
            'all': always store it as permanent memory
        """
        bs = key.shape[0]
        ne = key.shape[-1]
        assert len(key.shape) == 3
        assert len(shrinkage.shape) == 3
        assert not self.save_selection or len(selection.shape) == 3
        assert as_permanent in ['no', 'first', 'all']

        # add the value and create new buckets if necessary
        if supposed_bucket_id >= 0:
            enabled_buckets = [supposed_bucket_id]
            bucket_exist = supposed_bucket_id in self.buckets
            for obj, value in values.items():
                if bucket_exist:
                    assert obj in self.v
                    assert obj in self.buckets[supposed_bucket_id]
                    _add_last_dim(self.v, obj, value, prepend=(as_permanent == 'all'))
                else:
                    assert obj not in self.v
                    self.v[obj] = value
            self.buckets[supposed_bucket_id] = list(values.keys())
        else:
            new_bucket_id = None
            enabled_buckets = set()
            for obj, value in values.items():
                assert len(value.shape) == 3
                if obj in self.v:
                    _add_last_dim(self.v, obj, value, prepend=(as_permanent == 'all'))
                    bucket_used = [
                        bucket_id for bucket_id, object_ids in self.buckets.items()
                        if obj in object_ids
                    ]
                    assert len(bucket_used) == 1  # each object should only be in one bucket
                    enabled_buckets.add(bucket_used[0])
                else:
                    self.v[obj] = value
                    if new_bucket_id is None:
                        # create new bucket
                        new_bucket_id = self.global_bucket_id
                        self.global_bucket_id += 1
                        self.buckets[new_bucket_id] = []
                    # put the new object into the corresponding bucket
                    self.buckets[new_bucket_id].append(obj)
                    enabled_buckets.add(new_bucket_id)

        # increment the permanent size if necessary
        add_as_permanent = {}  # indexed by bucket id
        for bucket_id in enabled_buckets:
            add_as_permanent[bucket_id] = False
            if as_permanent == 'all':
                self.perm_end_pt[bucket_id] += ne
                add_as_permanent[bucket_id] = True
            elif as_permanent == 'first':
                if self.perm_end_pt[bucket_id] == 0:
                    self.perm_end_pt[bucket_id] = ne
                    add_as_permanent[bucket_id] = True

        # create new counters for usage if necessary
        if self.save_usage and as_permanent != 'all':
            new_count = torch.zeros((bs, ne), device=key.device, dtype=torch.float32)
            new_life = torch.zeros((bs, ne), device=key.device, dtype=torch.float32) + 1e-7

        # add the key to every bucket
        for bucket_id in self.buckets:
            if bucket_id not in enabled_buckets:
                # if we are not adding new values to a bucket, we should skip it
                continue

            _add_last_dim(self.k, bucket_id, key, prepend=add_as_permanent[bucket_id])
            _add_last_dim(self.s, bucket_id, shrinkage, prepend=add_as_permanent[bucket_id])
            if not add_as_permanent[bucket_id]:
                if self.save_selection:
                    _add_last_dim(self.e, bucket_id, selection)
                if self.save_usage:
                    _add_last_dim(self.use_cnt, bucket_id, new_count)
                    _add_last_dim(self.life_cnt, bucket_id, new_life)

    def update_bucket_usage(self, bucket_id: int, usage: torch.Tensor) -> None:
        # increase all life count by 1
        # increase use of indexed elements
        if not self.save_usage:
            return

        usage = usage[:, self.perm_end_pt[bucket_id]:]
        if usage.shape[-1] == 0:
            # if there is no temporary memory, we don't need to update
            return
        self.use_cnt[bucket_id] += usage.view_as(self.use_cnt[bucket_id])
        self.life_cnt[bucket_id] += 1

    def sieve_by_range(self, bucket_id: int, start: int, end: int, min_size: int) -> None:
        # keep only the temporary elements *outside* of this range (with some boundary conditions)
        # the permanent elements are ignored in this computation
        # i.e., concat (a[:start], a[end:])
        # bucket with size <= min_size are not modified

        assert start >= 0
        assert end <= 0

        object_ids = self.buckets[bucket_id]
        bucket_num_elements = self.k[bucket_id].shape[-1] - self.perm_end_pt[bucket_id]
        if bucket_num_elements <= min_size:
            return

        if end == 0:
            # negative 0 would not work as the end index!
            # effectively make the second part an empty slice
            end = self.k[bucket_id].shape[-1] + 1

        p_size = self.perm_end_pt[bucket_id]
        start = start + p_size

        k = self.k[bucket_id]
        s = self.s[bucket_id]
        if self.save_selection:
            e = self.e[bucket_id]
        if self.save_usage:
            use_cnt = self.use_cnt[bucket_id]
            life_cnt = self.life_cnt[bucket_id]

        self.k[bucket_id] = torch.cat([k[:, :, :start], k[:, :, end:]], -1)
        self.s[bucket_id] = torch.cat([s[:, :, :start], s[:, :, end:]], -1)
        if self.save_selection:
            self.e[bucket_id] = torch.cat([e[:, :, :start - p_size], e[:, :, end:]], -1)
        if self.save_usage:
            self.use_cnt[bucket_id] = torch.cat([use_cnt[:, :start - p_size], use_cnt[:, end:]], -1)
            self.life_cnt[bucket_id] = torch.cat([life_cnt[:, :start - p_size], life_cnt[:, end:]],
                                                 -1)
        for obj_id in object_ids:
            v = self.v[obj_id]
            self.v[obj_id] = torch.cat([v[:, :, :start], v[:, :, end:]], -1)

    def remove_old_memory(self, bucket_id: int, max_len: int) -> None:
        self.sieve_by_range(bucket_id, 0, -max_len, max_len)

    def remove_obsolete_features(self, bucket_id: int, max_size: int) -> None:
        # for long-term memory only
        object_ids = self.buckets[bucket_id]

        assert self.perm_end_pt[bucket_id] == 0  # permanent memory should be empty in LT memory

        # normalize with life duration
        usage = self.get_usage(bucket_id)
        bs = usage.shape[0]

        survivals = []

        for bi in range(bs):
            _, survived = torch.topk(usage[bi], k=max_size)
            survivals.append(survived.flatten())
            assert survived.shape[-1] == survivals[0].shape[-1]

        self.k[bucket_id] = torch.stack(
            [self.k[bucket_id][bi, :, survived] for bi, survived in enumerate(survivals)], 0)
        self.s[bucket_id] = torch.stack(
            [self.s[bucket_id][bi, :, survived] for bi, survived in enumerate(survivals)], 0)

        if self.save_selection:
            # Long-term memory does not store selection so this should not be needed
            self.e[bucket_id] = torch.stack(
                [self.e[bucket_id][bi, :, survived] for bi, survived in enumerate(survivals)], 0)
        for obj_id in object_ids:
            self.v[obj_id] = torch.stack(
                [self.v[obj_id][bi, :, survived] for bi, survived in enumerate(survivals)], 0)

        self.use_cnt[bucket_id] = torch.stack(
            [self.use_cnt[bucket_id][bi, survived] for bi, survived in enumerate(survivals)], 0)
        self.life_cnt[bucket_id] = torch.stack(
            [self.life_cnt[bucket_id][bi, survived] for bi, survived in enumerate(survivals)], 0)

    def get_usage(self, bucket_id: int) -> torch.Tensor:
        # return normalized usage
        if not self.save_usage:
            raise RuntimeError('I did not count usage!')
        else:
            usage = self.use_cnt[bucket_id] / self.life_cnt[bucket_id]
            return usage

    def get_all_sliced(
        self, bucket_id: int, start: int, end: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], torch.Tensor):
        # return k, sk, ek, value, normalized usage in order, sliced by start and end
        # this only queries the temporary memory

        assert start >= 0
        assert end <= 0

        p_size = self.perm_end_pt[bucket_id]
        start = start + p_size

        if end == 0:
            # negative 0 would not work as the end index!
            k = self.k[bucket_id][:, :, start:]
            sk = self.s[bucket_id][:, :, start:]
            ek = self.e[bucket_id][:, :, start - p_size:] if self.save_selection else None
            value = {obj_id: self.v[obj_id][:, :, start:] for obj_id in self.buckets[bucket_id]}
            usage = self.get_usage(bucket_id)[:, start - p_size:] if self.save_usage else None
        else:
            k = self.k[bucket_id][:, :, start:end]
            sk = self.s[bucket_id][:, :, start:end]
            ek = self.e[bucket_id][:, :, start - p_size:end] if self.save_selection else None
            value = {obj_id: self.v[obj_id][:, :, start:end] for obj_id in self.buckets[bucket_id]}
            usage = self.get_usage(bucket_id)[:, start - p_size:end] if self.save_usage else None

        return k, sk, ek, value, usage

    def purge_except(self, obj_keep_idx: List[int]):
        # purge certain objects from the memory except the one listed
        obj_keep_idx = set(obj_keep_idx)

        # remove objects that are not in the keep list from the buckets
        buckets_to_remove = []
        for bucket_id, object_ids in self.buckets.items():
            self.buckets[bucket_id] = [obj_id for obj_id in object_ids if obj_id in obj_keep_idx]
            if len(self.buckets[bucket_id]) == 0:
                buckets_to_remove.append(bucket_id)

        # remove object values that are not in the keep list
        self.v = {k: v for k, v in self.v.items() if k in obj_keep_idx}

        # remove buckets that are empty
        for bucket_id in buckets_to_remove:
            del self.buckets[bucket_id]
            del self.k[bucket_id]
            del self.s[bucket_id]
            if self.save_selection:
                del self.e[bucket_id]
            if self.save_usage:
                del self.use_cnt[bucket_id]
                del self.life_cnt[bucket_id]

    def clear_non_permanent_memory(self):
        # clear all non-permanent memory
        for bucket_id in self.buckets:
            self.sieve_by_range(bucket_id, 0, 0, 0)

    def get_v_size(self, obj_id: int) -> int:
        return self.v[obj_id].shape[-1]

    def size(self, bucket_id: int) -> int:
        if bucket_id not in self.k:
            return 0
        else:
            return self.k[bucket_id].shape[-1]

    def perm_size(self, bucket_id: int) -> int:
        return self.perm_end_pt[bucket_id]

    def non_perm_size(self, bucket_id: int) -> int:
        return self.size(bucket_id) - self.perm_size(bucket_id)

    def engaged(self, bucket_id: Optional[int] = None) -> bool:
        if bucket_id is None:
            return len(self.buckets) > 0
        else:
            return bucket_id in self.buckets

    @property
    def num_objects(self) -> int:
        return len(self.v)

    @property
    def key(self) -> Dict[int, torch.Tensor]:
        return self.k

    @property
    def value(self) -> Dict[int, torch.Tensor]:
        return self.v

    @property
    def shrinkage(self) -> Dict[int, torch.Tensor]:
        return self.s

    @property
    def selection(self) -> Dict[int, torch.Tensor]:
        return self.e

    def __contains__(self, key):
        return key in self.v
