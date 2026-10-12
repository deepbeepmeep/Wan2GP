# DyMeshVAE of AnimateAnyMesh, from JarrentWu1031/AnimateAnyMesh networks/dymeshvae.py and utils/mesh_utils.py (Apache License 2.0,
# see LICENSE in this folder; it builds on 3DShape2VecSet). Modified by DeepBeepMeep for WanGP: inference only; the encoder keeps
# only the point stream that produces the static-shape latent (the motion stream encodes an existing animation and is dropped from
# the WanGP checkpoint); pytorch3d's farthest point sampling and the dense N x N adjacency mask of the neighbour
# attention are replaced by pure PyTorch code (sequential farthest point sampling, attention over the mesh edges); attention
# through pay_attention.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn.functional as F
from torch import nn

from shared.attention import pay_attention
from shared.utils.phase_progress import check_abort

VERTEX_CHUNK = 8192  # tokens per chunk of the neighbour attention (about 6 edges each)
QUERY_CHUNK = 1 << 15  # vertices per chunk of the final cross-attention


def attend(query, key, value):
    """Attention of (batch, tokens, heads, head_dim) tensors in FP32 (the checkpoint precision; the 16-bit kernels do not apply), as
    upstream's F.scaled_dot_product_attention: value may have a larger head_dim than query and key."""
    return pay_attention([query, key, value], force_attention="sdpa")


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def feed_forward(dim, mult=4):
    return nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()
        self.register_buffer("basis", torch.empty(3, hidden_dim // 2))  # checkpoint buffer: 2^i * pi on each axis
        self.mlp = nn.Linear(hidden_dim + 3, dim)

    def forward(self, points):
        projections = points @ self.basis
        return self.mlp(torch.cat([projections.sin(), projections.cos(), points], dim=-1))


class NeighborAttention(nn.Module):
    """Upstream PreNorm_qkv(NeighborAttention): one head of width 2 * dim, each vertex attending to its mesh neighbours (the 1-ring of
    the welded mesh, without itself). Upstream masks a dense N x N attention with the adjacency matrix; here the softmax runs over the
    edges, the same attention without the N x N mask. A vertex without neighbours gets 0, as the masked SDPA kernel gives."""

    def __init__(self, dim, head_dim):
        super().__init__()
        self.norm_q, self.norm_k, self.norm_v = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.fn = nn.Module()
        self.fn.to_q, self.fn.to_k, self.fn.to_v = (nn.Linear(dim, head_dim, bias=False) for _ in range(3))
        self.fn.to_out = nn.Linear(head_dim, dim)
        self.scale = head_dim ** -0.5

    def forward(self, x, targets, sources, counts):
        """x (tokens, dim); targets/sources: directed edges (both directions of each mesh edge) sorted by target, counts: edges per
        token. Segment reductions over the sorted edges, by chunks of tokens: deterministic, unlike scatter additions."""
        query, key, value = self.fn.to_q(self.norm_q(x)), self.fn.to_k(self.norm_k(x)), self.fn.to_v(self.norm_v(x))
        offsets = [0] + counts.cumsum(0).tolist()  # one host copy: the edge range of each chunk of tokens
        out = torch.empty_like(value)
        for start in range(0, len(x), VERTEX_CHUNK):
            stop = min(start + VERTEX_CHUNK, len(x))
            first, last, lengths = offsets[start], offsets[stop], counts[start:stop]
            edge_targets, edge_sources = targets[first:last], sources[first:last]
            scores = (query[edge_targets] * key[edge_sources]).sum(-1).mul_(self.scale)
            maxima = torch.segment_reduce(scores, "max", lengths=lengths)
            weights = scores.sub_(maxima.repeat_interleave(lengths, output_size=last - first)).exp_()
            totals = torch.segment_reduce(weights, "sum", lengths=lengths)  # 0 for tokens without edges
            out[start:stop] = torch.segment_reduce(value[edge_sources] * weights[:, None], "sum", lengths=lengths).div_(totals.clamp_min(1e-30)[:, None])
            del scores, maxima, weights, totals
        del query, key, value
        return self.fn.to_out(out)


class SyncAttention(nn.Module):
    """Upstream SyncAttention(PreNorm): queries and keys from the first stream; values from the first stream (v1) and, when two
    streams are given, from the second (v2), attended with the same weights."""

    def __init__(self, dim, heads, head_dim, two_streams=True):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.norm_q, self.norm_k, self.norm_v1 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.Module()
        inner = heads * head_dim
        self.attn.to_q, self.attn.to_k, self.attn.to_v1 = (nn.Linear(dim, inner, bias=False) for _ in range(3))
        self.attn.to_out1 = nn.Linear(inner, dim)
        if two_streams:
            self.norm_v2 = nn.LayerNorm(dim)
            self.attn.to_v2 = nn.Linear(dim, inner, bias=False)
            self.attn.to_out2 = nn.Linear(inner, dim)

    def forward(self, q_stream, k_stream, v1_stream, v2_stream=None):
        heads = lambda t: t.unflatten(-1, (self.heads, -1))
        query, key = heads(self.attn.to_q(self.norm_q(q_stream))), heads(self.attn.to_k(self.norm_k(k_stream)))
        values = [heads(self.attn.to_v1(self.norm_v1(v1_stream)))]
        if v2_stream is not None:
            values.append(heads(self.attn.to_v2(self.norm_v2(v2_stream))))
        out = attend(query, key, torch.cat(values, dim=-1) if len(values) > 1 else values[0])
        del query, key, values
        out1 = self.attn.to_out1(out[..., :self.head_dim].flatten(2))
        return out1 if v2_stream is None else (out1, self.attn.to_out2(out[..., self.head_dim:].flatten(2)))


class SyncFeedForward(nn.Module):
    def __init__(self, dim, two_streams=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Module()
        self.ffn.ffn1 = feed_forward(dim)
        if two_streams:
            self.norm2 = nn.LayerNorm(dim)
            self.ffn.ffn2 = feed_forward(dim)

    def forward(self, x1, x2=None):
        out1 = self.ffn.ffn1(self.norm1(x1))
        return out1 if x2 is None else (out1, self.ffn.ffn2(self.norm2(x2)))


class FinalCrossAttention(nn.Module):
    """Upstream PreNorm_qkv(Attention_qkv): every vertex query attends to the decoded tokens (keys from the shape stream, values from
    the motion stream), one head of width 2 * dim."""

    def __init__(self, query_dim, dim, head_dim):
        super().__init__()
        self.norm_q, self.norm_k, self.norm_v = nn.LayerNorm(query_dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.fn = nn.Module()
        self.fn.to_q = nn.Linear(query_dim, head_dim, bias=False)
        self.fn.to_k, self.fn.to_v = nn.Linear(dim, head_dim, bias=False), nn.Linear(dim, head_dim, bias=False)
        self.fn.to_out = nn.Linear(head_dim, query_dim)

    def forward(self, queries, key, value):
        key, value = self.fn.to_k(self.norm_k(key))[:, :, None], self.fn.to_v(self.norm_v(value))[:, :, None]
        return self.fn.to_out(attend(self.fn.to_q(self.norm_q(queries))[:, :, None], key, value)[:, :, 0])


def mesh_edges(faces, count):
    """Directed edges (targets, sources) of the welded mesh, each undirected edge in both directions once, sorted by target, and the
    number of edges of each of the count tokens: the nonzero entries of upstream get_adjacency_matrix (adjacency of the face edges,
    symmetric, no self loops; padded tokens have none)."""
    pairs = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    pairs = torch.cat([pairs, pairs.flip(1)])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    keys = torch.unique(pairs[:, 0] * count + pairs[:, 1])
    targets = keys // count
    return targets, keys % count, torch.bincount(targets, minlength=count)


def farthest_point_order(points, count):
    """pytorch3d sample_farthest_points(points, K=count) as upstream calls it: squared euclidean distances, start at index 0, and
    past the number of points upstream's -1, here 0 (both replaced by random indices, see replace_start)."""
    total = len(points)
    selected = torch.zeros(count, dtype=torch.long, device=points.device)
    distances = torch.full((total,), float("inf"), device=points.device)
    last = selected[:1]
    for i in range(1, min(count, total)):  # indices stay on the device: no host synchronization per point
        torch.minimum(distances, (points - points.index_select(0, last)).square_().sum(-1), out=distances)
        last = distances.argmax().view(1)
        selected[i:i + 1] = last
    return selected


def replace_start(order, total, generator):
    """Upstream replace_negative_indices: every index <= 0 (the start point included) becomes a random index among the total points,
    drawn with torch.rand on the device."""
    random = (torch.rand((1, len(order)), generator=generator, device=order.device)[0] * total).long()
    return torch.where(order <= 0, random, order)


class DyMeshVAE(nn.Module):
    def __init__(self, enc_depth=8, dec_depth=8, dim=256, output_dim=48, latent_dim=32, heads=8, dim_head=64, T=16, n_layers=1, **kwargs):
        super().__init__()
        self.frames, self.latent_dim = T, latent_dim
        self.point_embed = PointEmbed(dim=dim)
        self.neighbor_layers = nn.ModuleList([NeighborAttention(dim, 2 * dim) for _ in range(n_layers)])
        self.enc_blocks = nn.ModuleList([nn.ModuleList([SyncAttention(dim, 1, dim, two_streams=False), SyncFeedForward(dim, two_streams=False)]) for _ in range(enc_depth)])
        self.mean_fc_x0 = nn.Linear(dim, latent_dim)
        self.proj_x0, self.proj_xt = nn.Linear(latent_dim, dim), nn.Linear(latent_dim, dim)
        self.dec_blocks = nn.ModuleList([nn.ModuleList([SyncAttention(dim, heads, dim_head), SyncFeedForward(dim)]) for _ in range(dec_depth)])
        self.fc_query = nn.Linear(dim, dim * 2)
        self.decoder_final_ca = FinalCrossAttention(dim * 2, dim, dim * 2)
        self.to_outputs = nn.Linear(dim * 2, output_dim)

    def embed_points(self, points, faces):
        """Per-vertex embeddings (1, tokens, dim) of the normalized rest positions (1, tokens, 3), real vertices then zero padding
        (upstream pads to max_length; padded rows have no neighbours)."""
        embedding = self.point_embed(points)[0]
        edges = mesh_edges(faces, len(embedding))
        for layer in self.neighbor_layers:
            embedding = embedding + layer(embedding, *edges)
        return embedding[None]

    def encode(self, embedding, order, valid, generator):
        """Static-shape latent x0 (1, tokens, latent_dim) of the embedded mesh: the tokens of the farthest point order among the `valid`
        real vertices, then cross-attention to all vertex embeddings (padding included, as upstream)."""
        tokens = embedding[:, replace_start(order, valid, generator)]
        for attention, feed_forward in self.enc_blocks:
            tokens = tokens + attention(tokens, embedding, embedding)
            tokens = tokens + feed_forward(tokens)
            check_abort()
        return self.mean_fc_x0(tokens)

    def decode(self, latent, embedding, points):
        """Vertex positions (1, T, vertices, 3) of the latent (1, tokens, 2 * latent_dim) for the vertices of `points` (1, vertices, 3)
        and their embeddings."""
        x0, xt = self.proj_x0(latent[..., :self.latent_dim]), self.proj_xt(latent[..., self.latent_dim:])
        for attention, feed_forward in self.dec_blocks:
            attention_0, attention_t = attention(x0, x0, x0, xt)
            x0, xt = x0 + attention_0, xt + attention_t
            ffn_0, ffn_t = feed_forward(x0, xt)
            x0, xt = x0 + ffn_0, xt + ffn_t
            check_abort()
        vertices = points.shape[1]  # the padded rows of the embedding have no output
        offsets = torch.cat([self.to_outputs(self.decoder_final_ca(self.fc_query(embedding[:, i:min(i + QUERY_CHUNK, vertices)]), x0, xt))
                             for i in range(0, vertices, QUERY_CHUNK)], dim=1)
        return points[:, None] + offsets.unflatten(-1, (self.frames, 3)).transpose(1, 2)
