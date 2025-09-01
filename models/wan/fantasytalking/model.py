import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.attention import pay_attention


class AudioProjModel(nn.Module):
    def __init__(self, audio_in_dim=1024, cross_attention_dim=1024):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Linear(audio_in_dim, cross_attention_dim, bias=False)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, audio_embeds):
        context_tokens = self.proj(audio_embeds)
        context_tokens = self.norm(context_tokens)
        return context_tokens  # [B,L,C]
 
class WanCrossAttentionProcessor(nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super().__init__()

        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.k_proj.weight)
        nn.init.zeros_(self.v_proj.weight)

    def __call__(
        self,
        q: torch.Tensor,
        audio_proj: torch.Tensor,
        latents_num_frames: int = 21,
        audio_context_lens = None
    ) -> torch.Tensor:
        """
        audio_proj:   [B, 21, L3, C]
        audio_context_lens: [B*21].
        """
        b, l, n, d = q.shape

        if len(audio_proj.shape) == 4:
            audio_q = q.view(b * latents_num_frames, -1, n, d)  # [b, 21, l1, n, d]
            ip_key = self.k_proj(audio_proj).view(b * latents_num_frames, -1, n, d)
            ip_value = self.v_proj(audio_proj).view(b * latents_num_frames, -1, n, d)
            qkv_list = [audio_q, ip_key, ip_value]
            del q, audio_q, ip_key, ip_value
            audio_x = pay_attention(qkv_list, k_lens =audio_context_lens) #audio_context_lens
            audio_x = audio_x.view(b, l, n, d)
            audio_x = audio_x.flatten(2)
        elif len(audio_proj.shape) == 3:
            ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
            ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
            qkv_list = [q, ip_key, ip_value]
            del q, ip_key, ip_value
            audio_x = pay_attention(qkv_list, k_lens =audio_context_lens) #audio_context_lens
            audio_x = audio_x.flatten(2)
        return audio_x


class FantasyTalkingAudioConditionModel(nn.Module):
    def __init__(self, wan_dit, audio_in_dim: int, audio_proj_dim: int):
        super().__init__()

        self.audio_in_dim = audio_in_dim
        self.audio_proj_dim = audio_proj_dim

    def split_audio_sequence(self, audio_proj_length, num_frames=81):
        """
        Map the audio feature sequence to corresponding latent frame slices.

        Args:
            audio_proj_length (int): The total length of the audio feature sequence
                                    (e.g., 173 in audio_proj[1, 173, 768]).
            num_frames (int): The number of video frames in the training data (default: 81).

        Returns:
            list: A list of [start_idx, end_idx] pairs. Each pair represents the index range
                (within the audio feature sequence) corresponding to a latent frame.
        """
        # Average number of tokens per original video frame
        tokens_per_frame = audio_proj_length / num_frames

        # Each latent frame covers 4 video frames, and we want the center
        tokens_per_latent_frame = tokens_per_frame * 4
        half_tokens = int(tokens_per_latent_frame / 2)

        pos_indices = []
        for i in range(int((num_frames - 1) / 4) + 1):
            if i == 0:
                pos_indices.append(0)
            else:
                start_token = tokens_per_frame * ((i - 1) * 4 + 1)
                end_token = tokens_per_frame * (i * 4 + 1)
                center_token = int((start_token + end_token) / 2) - 1
                pos_indices.append(center_token)

        # Build index ranges centered around each position
        pos_idx_ranges = [[idx - half_tokens, idx + half_tokens] for idx in pos_indices]

        # Adjust the first range to avoid negative start index
        pos_idx_ranges[0] = [
            -(half_tokens * 2 - pos_idx_ranges[1][0]),
            pos_idx_ranges[1][0],
        ]

        return pos_idx_ranges

    def split_tensor_with_padding(self, input_tensor, pos_idx_ranges, expand_length=0):
        """
        Split the input tensor into subsequences based on index ranges, and apply right-side zero-padding
        if the range exceeds the input boundaries.

        Args:
            input_tensor (Tensor): Input audio tensor of shape [1, L, 768].
            pos_idx_ranges (list): A list of index ranges, e.g. [[-7, 1], [1, 9], ..., [165, 173]].
            expand_length (int): Number of tokens to expand on both sides of each subsequence.

        Returns:
            sub_sequences (Tensor): A tensor of shape [1, F, L, 768], where L is the length after padding.
                                    Each element is a padded subsequence.
            k_lens (Tensor): A tensor of shape [F], representing the actual (unpadded) length of each subsequence.
                            Useful for ignoring padding tokens in attention masks.
        """
        pos_idx_ranges = [
            [idx[0] - expand_length, idx[1] + expand_length] for idx in pos_idx_ranges
        ]
        sub_sequences = []
        seq_len = input_tensor.size(1)  # 173
        max_valid_idx = seq_len - 1  # 172
        k_lens_list = []
        for start, end in pos_idx_ranges:
            # Calculate the fill amount
            pad_front = max(-start, 0)
            pad_back = max(end - max_valid_idx, 0)

            # Calculate the start and end indices of the valid part
            valid_start = max(start, 0)
            valid_end = min(end, max_valid_idx)

            # Extract the valid part
            if valid_start <= valid_end:
                valid_part = input_tensor[:, valid_start : valid_end + 1, :]
            else:
                valid_part = input_tensor.new_zeros((1, 0, input_tensor.size(2)))

            # In the sequence dimension (the 1st dimension) perform padding
            padded_subseq = F.pad(
                valid_part,
                (0, 0, 0, pad_back + pad_front, 0, 0),
                mode="constant",
                value=0,
            )
            k_lens_list.append(padded_subseq.size(-2) - pad_back - pad_front)

            sub_sequences.append(padded_subseq)
        return torch.stack(sub_sequences, dim=1), torch.tensor(
            k_lens_list, dtype=torch.long
        )
