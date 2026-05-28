from collections import deque
import xxhash
import numpy as np

from .sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                # Fix: Clean up hash_to_block_id mapping to prevent stale references
                # This prevents CUDA illegal memory access when prefix cache tries to
                # reuse a block_id that has already been freed
                if block.hash != -1:
                    cached_id = self.hash_to_block_id.get(block.hash)
                    if cached_id == block_id:
                        del self.hash_to_block_id[block.hash]
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        needs_new_block = len(seq) % self.block_size == 1 and len(seq.block_table) < seq.num_blocks
        return len(self.free_block_ids) >= needs_new_block

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            if len(block_table) >= seq.num_blocks:
                assert last_block.hash == -1
                return
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1

    def prompt_append_blocks_needed(self, seq: Sequence, old_num_tokens: int) -> int:
        old_num_tokens = max(0, int(old_num_tokens or 0))
        return max(0, int(seq.num_blocks) - int(self._prompt_append_existing_blocks(seq, old_num_tokens)))

    def can_prompt_append(self, seq: Sequence, old_num_tokens: int) -> bool:
        return len(self.free_block_ids) >= self.prompt_append_blocks_needed(seq, old_num_tokens)

    def _prompt_append_existing_blocks(self, seq: Sequence, old_num_tokens: int) -> int:
        old_num_tokens = max(0, int(old_num_tokens or 0))
        if old_num_tokens == 0:
            return 0
        old_num_blocks = (old_num_tokens + self.block_size - 1) // self.block_size
        existing_blocks = len(seq.block_table)
        if existing_blocks >= old_num_blocks:
            return old_num_blocks
        if old_num_tokens % self.block_size == 1 and existing_blocks == old_num_blocks - 1:
            return existing_blocks
        raise AssertionError(f"Prompt-append block table mismatch: tokens={old_num_tokens} blocks={existing_blocks} expected>={old_num_blocks - 1}")

    def begin_prompt_append(self, seq: Sequence, old_num_tokens: int) -> None:
        old_num_tokens = max(0, int(old_num_tokens or 0))
        self._prompt_append_existing_blocks(seq, old_num_tokens)
        while len(seq.block_table) < seq.num_blocks:
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)

    def finalize_prompt_append(self, seq: Sequence, old_num_tokens: int) -> None:
        old_num_tokens = max(0, int(old_num_tokens or 0))
        start_block = 0 if old_num_tokens == 0 else old_num_tokens // self.block_size
        prefix_hash = -1
        if start_block > 0 and start_block <= len(seq.block_table):
            prefix_hash = int(self.blocks[seq.block_table[start_block - 1]].hash)
        for block_index in range(start_block, int(seq.num_blocks)):
            block_id = int(seq.block_table[block_index])
            block = self.blocks[block_id]
            token_ids = seq.block(block_index)
            previous_hash = int(block.hash)
            if len(token_ids) == self.block_size:
                new_hash = self.compute_hash(token_ids, prefix_hash)
                if previous_hash != -1:
                    cached_id = self.hash_to_block_id.get(previous_hash)
                    if cached_id == block_id and previous_hash != new_hash:
                        del self.hash_to_block_id[previous_hash]
                block.update(new_hash, token_ids)
                self.hash_to_block_id[new_hash] = block_id
                prefix_hash = new_hash
            else:
                if previous_hash != -1:
                    cached_id = self.hash_to_block_id.get(previous_hash)
                    if cached_id == block_id:
                        del self.hash_to_block_id[previous_hash]
                block.hash = -1
                block.token_ids = []
                break

    def normalize_tail_after_prefill(self, seq: Sequence) -> None:
        if not seq.block_table or len(seq) == 0 or len(seq) % self.block_size != 0:
            return
        last_block = self.blocks[int(seq.block_table[-1])]
        if last_block.hash == -1:
            return
        cached_id = self.hash_to_block_id.get(int(last_block.hash))
        if cached_id == last_block.block_id:
            del self.hash_to_block_id[last_block.hash]
        last_block.hash = -1
        last_block.token_ids = []
