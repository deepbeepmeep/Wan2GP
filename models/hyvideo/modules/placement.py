import torch
import triton
import triton.language as tl

def hunyuan_token_reorder_to_token_major(tensor, fix_len, reorder_len, reorder_num_frame, frame_size):
    """Reorder it from frame major to token major!"""
    assert reorder_len == reorder_num_frame * frame_size
    assert tensor.shape[2] == fix_len + reorder_len

    tensor[:, :, :-fix_len, :] = tensor[:, :, :-fix_len:, :].reshape(tensor.shape[0], tensor.shape[1], reorder_num_frame, frame_size, tensor.shape[3]) \
                                                         .transpose(2, 3).reshape(tensor.shape[0], tensor.shape[1], reorder_len, tensor.shape[3])
    return tensor

def hunyuan_token_reorder_to_frame_major(tensor, fix_len, reorder_len, reorder_num_frame, frame_size):
    """Reorder it from token major to frame major!"""
    assert reorder_len == reorder_num_frame * frame_size
    assert tensor.shape[2] == fix_len + reorder_len

    tensor[:, :, :-fix_len:, :] = tensor[:, :, :-fix_len:, :].reshape(tensor.shape[0], tensor.shape[1], frame_size, reorder_num_frame, tensor.shape[3]) \
                                                         .transpose(2, 3).reshape(tensor.shape[0], tensor.shape[1], reorder_len, tensor.shape[3])
    return tensor


@triton.jit
def hunyuan_sparse_head_placement_kernel(
    query_ptr, key_ptr, value_ptr, # [cfg, num_heads, seq_len, head_dim] seq_len = context_length + num_frame * frame_size
    query_out_ptr, key_out_ptr, value_out_ptr, # [cfg, num_heads, seq_len, head_dim]
    best_mask_idx_ptr, # [cfg, num_heads]
    query_stride_b, query_stride_h, query_stride_s, query_stride_d,
    mask_idx_stride_b, mask_idx_stride_h,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    context_length: tl.constexpr,   
    num_frame: tl.constexpr,        
    frame_size: tl.constexpr,      
    BLOCK_SIZE: tl.constexpr
):
    # Copy query, key, value to output
    # range: [b, h, block_id * block_size: block_id * block_size + block_size, :]
    cfg = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)

    start_id = block_id * BLOCK_SIZE
    end_id = start_id + BLOCK_SIZE
    end_id = tl.where(end_id > seq_len, seq_len, end_id) 

    # Load best mask idx (0 is spatial, 1 is temporal)
    is_temporal = tl.load(best_mask_idx_ptr + cfg * mask_idx_stride_b + head * mask_idx_stride_h)
    
    offset_token = tl.arange(0, BLOCK_SIZE) + start_id
    offset_mask = offset_token < seq_len
    offset_d = tl.arange(0, head_dim)

    if is_temporal:
        frame_id = offset_token // frame_size
        patch_id = offset_token - frame_id * frame_size
        offset_store_token = tl.where(offset_token >= seq_len - context_length, offset_token, patch_id * num_frame + frame_id)

        offset_load = (cfg * query_stride_b + head * query_stride_h + offset_token[:,None] * query_stride_s) + offset_d[None,:] * query_stride_d
        offset_query = query_ptr + offset_load
        offset_key = key_ptr + offset_load
        offset_value = value_ptr + offset_load

        offset_store = (cfg * query_stride_b + head * query_stride_h + offset_store_token[:,None] * query_stride_s) + offset_d[None,:] * query_stride_d
        offset_query_out = query_out_ptr + offset_store
        offset_key_out = key_out_ptr + offset_store
        offset_value_out = value_out_ptr + offset_store

        # Maybe tune the pipeline here
        query = tl.load(offset_query, mask=offset_mask[:,None])
        tl.store(offset_query_out, query, mask=offset_mask[:,None])
        key = tl.load(offset_key, mask=offset_mask[:,None])
        tl.store(offset_key_out, key, mask=offset_mask[:,None])
        value = tl.load(offset_value, mask=offset_mask[:,None])
        tl.store(offset_value_out, value, mask=offset_mask[:,None])


    else:
        offset_load = (cfg * query_stride_b + head * query_stride_h + offset_token[:,None] * query_stride_s) + offset_d[None,:] * query_stride_d
        offset_query = query_ptr + offset_load
        offset_key = key_ptr + offset_load
        offset_value = value_ptr + offset_load

        offset_store = offset_load
        offset_query_out = query_out_ptr + offset_store
        offset_key_out = key_out_ptr + offset_store
        offset_value_out = value_out_ptr + offset_store

        # Maybe tune the pipeline here
        query = tl.load(offset_query, mask=offset_mask[:,None])
        tl.store(offset_query_out, query, mask=offset_mask[:,None])
        key = tl.load(offset_key, mask=offset_mask[:,None])
        tl.store(offset_key_out, key, mask=offset_mask[:,None])
        value = tl.load(offset_value, mask=offset_mask[:,None])
        tl.store(offset_value_out, value, mask=offset_mask[:,None])


def hunyuan_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):
    cfg, num_heads, seq_len, head_dim = query.shape
    BLOCK_SIZE = 128
    assert seq_len == context_length + num_frame * frame_size

    grid = (cfg, num_heads, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    hunyuan_sparse_head_placement_kernel[grid](
        query, key, value, 
        query_out, key_out, value_out, 
        best_mask_idx,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        best_mask_idx.stride(0), best_mask_idx.stride(1),
        seq_len, head_dim, context_length, num_frame, frame_size, 
        BLOCK_SIZE
    )


def ref_hunyuan_sparse_head_placement(query, key, value, best_mask_idx, context_length, num_frame, frame_size):
    cfg, num_heads, seq_len, head_dim = query.shape
    assert seq_len == context_length + num_frame * frame_size

    query_out = query.clone()
    key_out = key.clone()
    value_out = value.clone()

    # Spatial
    query_out[best_mask_idx == 0], key_out[best_mask_idx == 0], value_out[best_mask_idx == 0] = \
        query[best_mask_idx == 0], key[best_mask_idx == 0], value[best_mask_idx == 0]

    # Temporal
    query_out[best_mask_idx == 1], key_out[best_mask_idx == 1], value_out[best_mask_idx == 1] = \
            hunyuan_token_reorder_to_token_major(query[best_mask_idx == 1].unsqueeze(0), context_length, num_frame * frame_size, num_frame, frame_size).squeeze(0), \
            hunyuan_token_reorder_to_token_major(key[best_mask_idx == 1].unsqueeze(0), context_length, num_frame * frame_size, num_frame, frame_size).squeeze(0), \
            hunyuan_token_reorder_to_token_major(value[best_mask_idx == 1].unsqueeze(0), context_length, num_frame * frame_size, num_frame, frame_size).squeeze(0)

    return query_out, key_out, value_out


def test_hunyuan_sparse_head_placement():

    context_length = 226
    num_frame = 11
    frame_size = 4080

    cfg = 2
    num_heads = 48

    seq_len = context_length + num_frame * frame_size
    head_dim = 64

    dtype = torch.bfloat16
    device = torch.device("cuda")

    query = torch.randn(cfg, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    key = torch.randn(cfg, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    value = torch.randn(cfg, num_heads, seq_len, head_dim, dtype=dtype, device=device)

    best_mask_idx = torch.randint(0, 2, (cfg, num_heads), device=device)

    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    value_out = torch.empty_like(value)

    hunyuan_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)
    ref_query_out, ref_key_out, ref_value_out = ref_hunyuan_sparse_head_placement(query, key, value, best_mask_idx, context_length, num_frame, frame_size)

    torch.testing.assert_close(query_out, ref_query_out)
    torch.testing.assert_close(key_out, ref_key_out)
    torch.testing.assert_close(value_out, ref_value_out)


def benchmark_hunyuan_sparse_head_placement():
    import time

    context_length = 226
    num_frame = 11
    frame_size = 4080

    cfg = 2
    num_heads = 48

    seq_len = context_length + num_frame * frame_size
    head_dim = 64

    dtype = torch.bfloat16
    device = torch.device("cuda")

    query = torch.randn(cfg, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    key = torch.randn(cfg, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    value = torch.randn(cfg, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    best_mask_idx = torch.randint(0, 2, (cfg, num_heads), device=device)

    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    value_out = torch.empty_like(value)

    warmup = 10
    all_iter = 1000

    # warmup
    for _ in range(warmup):
        hunyuan_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(all_iter):
        hunyuan_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Triton Elapsed Time: {(end - start) / all_iter * 1e3:.2f} ms")
    print(f"Triton Total Bandwidth: {query.nelement() * query.element_size() * 3 * 2 * all_iter / (end - start) / 1e9:.2f} GB/s")

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(all_iter):
        ref_hunyuan_sparse_head_placement(query, key, value, best_mask_idx, context_length, num_frame, frame_size)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Reference Elapsed Time: {(end - start) / all_iter * 1e3:.2f} ms")
    print(f"Reference Total Bandwidth: {query.nelement() * query.element_size() * 3 * 2 * all_iter / (end - start) / 1e9:.2f} GB/s")


@triton.jit
def hunyuan_hidden_states_placement_kernel(
    hidden_states_ptr, # [cfg, num_heads, seq_len, head_dim] seq_len = context_length + num_frame * frame_size
    hidden_states_out_ptr, # [cfg, num_heads, seq_len, head_dim]
    best_mask_idx_ptr, # [cfg, num_heads]
    hidden_states_stride_b, hidden_states_stride_h, hidden_states_stride_s, hidden_states_stride_d,
    mask_idx_stride_b, mask_idx_stride_h,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    context_length: tl.constexpr,   
    num_frame: tl.constexpr,        
    frame_size: tl.constexpr,      
    BLOCK_SIZE: tl.constexpr
):
    # Copy hidden_states to output
    # range: [b, h, block_id * block_size: block_id * block_size + block_size, :]
    cfg = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)

    start_id = block_id * BLOCK_SIZE
    end_id = start_id + BLOCK_SIZE
    end_id = tl.where(end_id > seq_len, seq_len, end_id) 

    # Load best mask idx (0 is spatial, 1 is temporal)
    is_temporal = tl.load(best_mask_idx_ptr + cfg * mask_idx_stride_b + head * mask_idx_stride_h)
    
    offset_token = tl.arange(0, BLOCK_SIZE) + start_id
    offset_mask = offset_token < seq_len
    offset_d = tl.arange(0, head_dim)

    if is_temporal:
        patch_id = offset_token // num_frame
        frame_id = offset_token - patch_id * num_frame
        offset_store_token = tl.where(offset_token >= seq_len - context_length, offset_token, frame_id * frame_size + patch_id)

        offset_load = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_token[:,None] * hidden_states_stride_s) + offset_d[None,:] * hidden_states_stride_d
        offset_hidden_states = hidden_states_ptr + offset_load

        offset_store = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_store_token[:,None] * hidden_states_stride_s) + offset_d[None,:] * hidden_states_stride_d
        offset_hidden_states_out = hidden_states_out_ptr + offset_store

        # Maybe tune the pipeline here
        hidden_states = tl.load(offset_hidden_states, mask=offset_mask[:,None])
        tl.store(offset_hidden_states_out, hidden_states, mask=offset_mask[:,None])
    else:
        offset_load = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_token[:,None] * hidden_states_stride_s) + offset_d[None,:] * hidden_states_stride_d
        offset_hidden_states = hidden_states_ptr + offset_load

        offset_store = offset_load
        offset_hidden_states_out = hidden_states_out_ptr + offset_store

        # Maybe tune the pipeline here
        hidden_states = tl.load(offset_hidden_states, mask=offset_mask[:,None])
        tl.store(offset_hidden_states_out, hidden_states, mask=offset_mask[:,None])


def hunyuan_hidden_states_placement(hidden_states, hidden_states_out, best_mask_idx, context_length, num_frame, frame_size):
    cfg, num_heads, seq_len, head_dim = hidden_states.shape
    BLOCK_SIZE = 128
    assert seq_len == context_length + num_frame * frame_size

    grid = (cfg, num_heads, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)


    hunyuan_hidden_states_placement_kernel[grid](
        hidden_states, 
        hidden_states_out, 
        best_mask_idx,
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3),
        best_mask_idx.stride(0), best_mask_idx.stride(1),
        seq_len, head_dim, context_length, num_frame, frame_size, 
        BLOCK_SIZE
    )

    return hidden_states_out

def ref_hunyuan_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size):
    cfg, num_heads, seq_len, head_dim = hidden_states.shape
    assert seq_len == context_length + num_frame * frame_size

    # Spatial
    output_hidden_states[best_mask_idx == 0] = hidden_states[best_mask_idx == 0]
    # Temporal
    output_hidden_states[best_mask_idx == 1] = hunyuan_token_reorder_to_frame_major(hidden_states[best_mask_idx == 1].unsqueeze(0), context_length, num_frame * frame_size, num_frame, frame_size).squeeze(0)

def test_hunyuan_hidden_states_placement():

    context_length = 226
    num_frame = 11
    frame_size = 4080

    cfg = 2
    num_heads = 48

    seq_len = context_length + num_frame * frame_size
    head_dim = 64

    dtype = torch.bfloat16
    device = torch.device("cuda")

    hidden_states = torch.randn(cfg, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    best_mask_idx = torch.randint(0, 2, (cfg, num_heads), device=device)

    hidden_states_out1 = torch.empty_like(hidden_states)
    hidden_states_out2 = torch.empty_like(hidden_states)

    hunyuan_hidden_states_placement(hidden_states, hidden_states_out1, best_mask_idx, context_length, num_frame, frame_size)
    ref_hunyuan_hidden_states_placement(hidden_states, hidden_states_out2, best_mask_idx, context_length, num_frame, frame_size)

    torch.testing.assert_close(hidden_states_out1, hidden_states_out2)

def benchmark_hunyuan_hidden_states_placement():
    import time

    context_length = 226
    num_frame = 11
    frame_size = 4080

    cfg = 2
    num_heads = 48

    seq_len = context_length + num_frame * frame_size
    head_dim = 64

    dtype = torch.bfloat16
    device = torch.device("cuda")

    hidden_states = torch.randn(cfg, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    best_mask_idx = torch.randint(0, 2, (cfg, num_heads), device=device)

    hidden_states_out = torch.empty_like(hidden_states)

    warmup = 10
    all_iter = 1000

    # warmup
    for _ in range(warmup):
        hunyuan_hidden_states_placement(hidden_states, hidden_states_out, best_mask_idx, context_length, num_frame, frame_size)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(all_iter):
        hunyuan_hidden_states_placement(hidden_states, hidden_states_out, best_mask_idx, context_length, num_frame, frame_size)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Triton Elapsed Time: {(end - start) / all_iter * 1e3:.2f} ms")
    print(f"Triton Total Bandwidth: {hidden_states.nelement() * hidden_states.element_size() * 2 * all_iter / (end - start) / 1e9:.2f} GB/s")

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(all_iter):
        ref_hunyuan_hidden_states_placement(hidden_states, hidden_states.clone(), best_mask_idx, context_length, num_frame, frame_size)
    torch.cuda.synchronize()
    end = time.time()

    print(f"Reference Elapsed Time: {(end - start) / all_iter * 1e3:.2f} ms")
    print(f"Reference Total Bandwidth: {hidden_states.nelement() * hidden_states.element_size() * 2 * all_iter / (end - start) / 1e9:.2f} GB/s")


if __name__ == "__main__":
    test_hunyuan_sparse_head_placement()
    benchmark_hunyuan_sparse_head_placement()
    test_hunyuan_hidden_states_placement()
    benchmark_hunyuan_hidden_states_placement()
