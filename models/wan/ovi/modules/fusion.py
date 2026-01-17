
import torch
import torch.nn as nn
from .model import WanLayerNorm, WanModel, WanRMSNorm, rope_apply
from shared.attention import pay_attention
##### Enjoy this spagheti VRAM optimizations done by DeepBeepMeep !
# I am sure you are a nice person and as you copy this code, you will give me officially proper credits:
# Please link to https://github.com/deepbeepmeep/Wan2GP and @deepbeepmeep on twitter  

def reshape_latent(latent, latent_frames):
    return latent.reshape(latent.shape[0], latent_frames, -1, latent.shape[-1] )

def restore_latent_shape(latent):
    return latent.reshape(latent.shape[0], -1, latent.shape[-1] )

class FusionModel(nn.Module):
    def __init__(self, video_config=None, audio_config=None):
        super().__init__()
        has_video = True 
        has_audio = True
        if video_config is not None:
            self.video_model = WanModel(**video_config)
        else:
            has_video = False
            self.video_model = None
            print("Warning: No video model is provided!")
        
        if audio_config is not None:
            self.audio_model = WanModel(**audio_config)
        else:
            has_audio = False
            self.audio_model = None
            print("Warning: No audio model is provided!")

        if has_video and has_audio:
            assert len(self.video_model.blocks) == len(self.audio_model.blocks)
            self.num_blocks = len(self.video_model.blocks)

            self.use_sp = False
            self.inject_cross_attention_kv_projections()

        self.init_weights()
        
    def inject_cross_attention_kv_projections(self):
        for vid_block in self.video_model.blocks:
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()

        
        for audio_block in self.audio_model.blocks:
            audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()


    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        """
        keys in each kwarg:
        e
        seq_lens
        grid_sizes
        freqs
        context
        context_lens
        """
        merged_kwargs = {}
        for key in vid_kwargs:
            merged_kwargs[f"vid_{key}"] = vid_kwargs[key]
        for key in audio_kwargs:
            merged_kwargs[f"audio_{key}"] = audio_kwargs[key]
        return merged_kwargs

    def single_fusion_cross_attention_forward(self,
                                            cross_attn_block,
                                            src_seq,
                                            src_grid_sizes,
                                            src_freqs,
                                            target_seq,
                                            target_seq_lens,
                                            target_grid_sizes,
                                            target_freqs,
                                            context,
                                            context_lens
                                            ):
                                            
        b, n, d = src_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim
        if hasattr(cross_attn_block, "k_img"):
            ## means is i2v block
            q, k, v, k_img, v_img = cross_attn_block.qkv_fn(src_seq, context)
        else:
            ## means is t2v block
            q, k, v = cross_attn_block.qkv_fn(src_seq, context)
            k_img = v_img = None

                    
        qkv_list =[q, k, v]
        del k, v
        x = pay_attention(qkv_list)

        if k_img is not None:
            qkv_list =[q, k_img, v_img]
            del k_img, v_img
            img_x = pay_attention(qkv_list)

            # img_x = flash_attention(q, k_img, v_img, k_lens=None)
            x += img_x

        is_vid = src_grid_sizes.shape[1] > 1
        # compute target attention
        target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
        k_target = cross_attn_block.norm_k_fusion(cross_attn_block.k_fusion(target_seq)).view(b, -1, n, d)
        v_target = cross_attn_block.v_fusion(target_seq).view(b, -1, n, d)
        
        q = rope_apply(q, src_grid_sizes, src_freqs)
        k_target = rope_apply(k_target, target_grid_sizes, target_freqs)
        
        qkv_list =[q, k_target, v_target]
        del q, k_target, v_target
        target_x = pay_attention(qkv_list)

        # target_x = flash_attention(q, k_target, v_target, k_lens=target_seq_lens)
        
        x += target_x
        
        x = x.flatten(2) # [B, L/P, C]

        x = cross_attn_block.o(x)
        return x

    def single_fusion_cross_attention_ffn_forward(self,
                                            attn_block,
                                            src_seq,
                                            src_grid_sizes,
                                            src_freqs,
                                            target_seq,
                                            target_seq_lens,
                                            target_grid_sizes,
                                            target_freqs,
                                            context,
                                            context_lens,
                                            src_e):
        
        src_seq += self.single_fusion_cross_attention_forward(attn_block.cross_attn,
                                                                       attn_block.norm3(src_seq),
                                                                       src_grid_sizes=src_grid_sizes,
                                                                       src_freqs=src_freqs,
                                                                       target_seq=target_seq,
                                                                       target_seq_lens=target_seq_lens,
                                                                       target_grid_sizes=target_grid_sizes,
                                                                       target_freqs=target_freqs,
                                                                       context=context,
                                                                       context_lens=context_lens
                                                                       )

        latent_frames = src_e[0].shape[0]

        y = attn_block.norm2(src_seq).to(torch.bfloat16)
        y = reshape_latent(y , latent_frames)        
        y *= (1 + src_e[4].squeeze(2)) 
        y += src_e[3].squeeze(2)
        y = restore_latent_shape(y)        
        # y = attn_block.ffn(y)


        ffn = attn_block.ffn[0]
        gelu = attn_block.ffn[1]
        ffn2= attn_block.ffn[2]

        y_shape = y.shape
        y = y.view(-1, y_shape[-1])
        chunk_size = int(y.shape[0]/2.7)
        chunks =torch.split(y, chunk_size)
        for y_chunk  in chunks:
            mlp_chunk = ffn(y_chunk)
            mlp_chunk = gelu(mlp_chunk)
            y_chunk[...] = ffn2(mlp_chunk)
            del mlp_chunk 
        y = y.view(y_shape)

        src_seq, y = reshape_latent(src_seq , latent_frames), reshape_latent(y , latent_frames)        
        src_seq.addcmul_(y, src_e[5].squeeze(2))
        src_seq = restore_latent_shape(src_seq)        
        del y

        # # y = attn_block.ffn(attn_block.norm2(src_seq).bfloat16() * (1 + src_e[4].squeeze(2)) + src_e[3].squeeze(2))
        # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        #     src_seq = src_seq + y * src_e[5].squeeze(2)
        return src_seq
        
    def single_fusion_block_forward(self,
                                    vid_block,
                                    audio_block,
                                    vid,
                                    audio,
                                    vid_e,
                                    vid_seq_lens,
                                    vid_grid_sizes,
                                    vid_freqs,
                                    vid_context,
                                    vid_context_lens,
                                    audio_e,
                                    audio_seq_lens,
                                    audio_grid_sizes,
                                    audio_freqs,
                                    audio_context,
                                    audio_context_lens,
                                    ):
        ## audio modulation
        audio_e = audio_block.modulation(audio_e).chunk(6, dim=1)

        # audio self-attention
        audio_y = audio_block.norm1(audio).to(torch.bfloat16)
        audio_y *= (1 + audio_e[1].squeeze(2)) 
        audio_y += audio_e[0].squeeze(2)
        audio_y = audio_block.self_attn(audio_y, audio_seq_lens, audio_grid_sizes, audio_freqs)

        audio.addcmul_(audio_y, audio_e[2].squeeze(2))
        del audio_y

        latent_frames = vid_e.shape[0]

        ## video modulation
        vid_e = vid_block.modulation(vid_e).chunk(6, dim=1)


        # video self-attention
        vid_y = vid_block.norm1(vid).to(torch.bfloat16)
        vid_y = reshape_latent(vid_y , latent_frames)        
        vid_y *= (1 + vid_e[1].squeeze(2)) 
        vid_y += vid_e[0].squeeze(2)
        vid_y = restore_latent_shape(vid_y)        
        vid_y = vid_block.self_attn(vid_y, vid_seq_lens, vid_grid_sizes, vid_freqs)
    

        vid, vid_y = reshape_latent(vid , latent_frames), reshape_latent(vid_y , latent_frames)        
        vid.addcmul_(vid_y, vid_e[2].squeeze(2))
        vid = restore_latent_shape(vid)        
        del vid_y

        # og_audio = audio

        # audio cross-attention
        audio = self.single_fusion_cross_attention_ffn_forward(
            audio_block,
            audio,
            audio_grid_sizes,
            audio_freqs,
            vid,
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
            audio_context,
            audio_context_lens,
            audio_e,
        )
        if audio is None:
            return None, None
        # if torch.equal(og_audio, audio):
        #     print("Audio should be changed after cross-attention!")
        # assert not torch.equal(og_audio, audio), "Audio should be changed after cross-attention!"

        # video cross-attention
        vid = self.single_fusion_cross_attention_ffn_forward(
            vid_block,
            vid,
            vid_grid_sizes,
            vid_freqs,
            audio,
            audio_seq_lens,
            audio_grid_sizes,
            audio_freqs,
            vid_context,
            vid_context_lens,
            vid_e,
        )
        if vid is None:
            return None, None

        return vid, audio

    def forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        computed_slg_layers=None,
        callback = None,
        pipeline= None,
        x_id_list = 0,
        video_freqs = None,
        audio_freqs = None,
    ):  
        vid_list = []
        vid_e_list = []
        audio_list = []
        audio_e_list = []
        kwargs_list = []
        for one_vid_context, one_audio_context in zip(vid_context, audio_context):
            one_vid, one_vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
                x=[vid], t=t, context=[one_vid_context], seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean, freqs=video_freqs
            )
            vid_list.append(one_vid)
            vid_e_list.append(one_vid_e)
            one_vid = one_vid_e = None
            one_audio, one_audio_e, audio_kwargs = self.audio_model.prepare_transformer_block_kwargs(
                x=[audio], t=t, context=[one_audio_context], seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None, first_frame_is_clean=False, freqs=audio_freqs
                )
            audio_list.append(one_audio)
            audio_e_list.append(one_audio_e)
            one_audio = one_audio_e = None

            kwargs_list.append(self.merge_kwargs(vid_kwargs, audio_kwargs))

        for i in range(self.num_blocks):
            """
            1 fusion block refers to 1 audio block with 1 video block.
            """
            if callback != None:
                callback(-1, None, False, True)
            vid_block = self.video_model.blocks[i]
            audio_block = self.audio_model.blocks[i]
            for x_id, one_vid, one_vid_e, one_audio, one_audio_e, one_kwargs in zip(x_id_list, vid_list, vid_e_list, audio_list, audio_e_list, kwargs_list):
                if pipeline._interrupt:
                    return None, None
                if x_id == 1 and computed_slg_layers is not None and i in computed_slg_layers:
                    continue
                # one_vid[...], one_audio[...] = self.single_fusion_block_forward(
                a, b = self.single_fusion_block_forward(
                        vid_block=vid_block,
                        audio_block=audio_block,
                        vid=one_vid,
                        audio=one_audio,
                        **one_kwargs
                    )
                one_vid[...], one_audio[...]  = a, b
            

        for i, (x_id, one_vid, one_vid_e, one_audio, one_audio_e) in enumerate(zip(x_id_list, vid_list, vid_e_list, audio_list, audio_e_list)):
            one_vid = self.video_model.post_transformer_block_out(one_vid, vid_kwargs['grid_sizes'], one_vid_e)
            vid_list[i] = one_vid
            one_vid = None 
            one_audio = self.audio_model.post_transformer_block_out(one_audio, audio_kwargs['grid_sizes'], one_audio_e)
            audio_list[i] = one_audio
            one_audio = None 

        if len(vid_list) == 1:
            return vid_list[0], audio_list[0]
        return vid_list, audio_list

    def init_weights(self):
        if self.audio_model is not None:
            self.audio_model.init_weights()

        if self.video_model is not None:
            self.video_model.init_weights()

        for name, mod in self.video_model.named_modules():
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight.div_(10.0)

    
    def custom_compile(self, compile_kwargs):
        torch.compile(self.single_fusion_block_forward, **compile_kwargs)        
