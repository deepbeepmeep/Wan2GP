import torch
from PIL import Image as PILImage
from shared.utils.phase_progress import text_encoding_progress
PipelineImageInput = PILImage.Image
class QwenImage21Pipeline:

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _get_qwen_prompt_embeds(self, prompt: str | list[str]=None, image: list | None=None, device: torch.device | None=None):
        device = device or torch.device("cuda")
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [' ' if not p else p for p in prompt]
        is_t2i = image is None
        if is_t2i:
            prompts = [self.prompt_template_t2i.format(t) for t in prompt]
        else:
            prompts = []
            condition_pil_list = []
            for t in prompt:
                n_imgs = len(image)
                replace = '<image1><|vision_start|><|image_pad|><|vision_end|>'
                for i in range(2, n_imgs + 1):
                    replace += f' <image{i}><|vision_start|><|image_pad|><|vision_end|>'
                template = self.prompt_template_ti2i.replace('<image1><|vision_start|><|image_pad|><|vision_end|>', replace)
                prompts.append(template.format(t))
            for _ in prompt:
                for img in image:
                    if not isinstance(img, PILImage.Image):
                        img = PILImage.fromarray(img)
                    if img.mode == 'RGBA':
                        white = PILImage.new('RGB', img.size, (255, 255, 255))
                        white.paste(img, mask=img.getchannel('A'))
                        img = white
                    condition_pil_list.append(img)
        processor_kwargs = {'text': prompts, 'padding': True, 'padding_side': 'left', 'return_tensors': 'pt'}
        if not is_t2i:
            processor_kwargs['images'] = condition_pil_list
        model_inputs = self.processor(**processor_kwargs).to(device)
        forward_kwargs = {'input_ids': model_inputs.input_ids, 'attention_mask': model_inputs.attention_mask, 'output_hidden_states': True, 'use_cache': False, 'logits_to_keep': 1}
        if not is_t2i and hasattr(model_inputs, 'pixel_values'):
            forward_kwargs.update(pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw)
        if hasattr(model_inputs, 'mm_token_type_ids'):
            forward_kwargs['mm_token_type_ids'] = model_inputs.mm_token_type_ids
        text_model = getattr(self.text_encoder.model, 'language_model', self.text_encoder.model)
        handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            with text_encoding_progress(text_model.layers, next_status="Preparing Image Conditioning"):
                outputs = self.text_encoder(**forward_kwargs)
        finally:
            handle.remove()
        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = list(self._extract_masked_hidden(hidden_states, model_inputs.attention_mask))
        split_hidden_states = [e[self._drop_idx:] for e in split_hidden_states]
        image_pad_mask = [sample_ids[sample_mask.bool()] == self._img_token_id for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask)]
        image_pad_mask = [e[self._drop_idx:] for e in image_pad_mask]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max((e.size(0) for e in split_hidden_states))
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
        encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
        image_pad_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in image_pad_mask])
        return (prompt_embeds, encoder_attention_mask, image_pad_mask)

    def encode_prompt(self, prompt: str | list[str], image: list[PipelineImageInput] | None=None, device: torch.device | None=None, num_images_per_prompt: int=1, prompt_embeds: torch.Tensor | None=None, prompt_embeds_mask: torch.Tensor | None=None, image_pad_mask: torch.Tensor | None=None):
        device = device or torch.device("cuda")
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask, image_pad_mask = self._get_qwen_prompt_embeds(prompt, image, device)
        elif image_pad_mask is None:
            if image is not None:
                raise ValueError('Pass `image_pad_mask` alongside `prompt_embeds` when the embeddings cover condition images, so the transformer knows which positions hold image tokens.')
            image_pad_mask = prompt_embeds.new_zeros(prompt_embeds.shape[:2], dtype=torch.bool)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
            prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
        if prompt_embeds_mask is not None and prompt_embeds_mask.all():
            prompt_embeds_mask = None
        return (prompt_embeds, prompt_embeds_mask, image_pad_mask)
