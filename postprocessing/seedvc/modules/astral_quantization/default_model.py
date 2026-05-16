import torch
from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor

class AstralQuantizer(torch.nn.Module):
    def __init__(
            self,
            tokenizer_name: str,
            ssl_model_name: str,
            ssl_output_layer: int,
            encoder: torch.nn.Module,
            quantizer: torch.nn.Module,
            skip_ssl: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load SSL model from Huggingface
        self.ssl_model_name = ssl_model_name
        self.ssl_output_layer = ssl_output_layer
        self.ssl_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ssl_model_name)

        if skip_ssl:  # in case the same SSL model has been loaded somewhere else
            self.ssl_model = None
        else:
            self.ssl_model = AutoModel.from_pretrained(ssl_model_name).eval()
            self.ssl_model.encoder.layers = self.ssl_model.encoder.layers[:ssl_output_layer]
            self.ssl_model.encoder.layer_norm = torch.nn.Identity()

    def load_separate_checkpoint(self, checkpoint_path):
        params = torch.load(checkpoint_path, map_location='cpu')['net']
        for key in params.keys():
            for k in list(params[key].keys()):
                if k.startswith("module."):
                    params[key][k[len("module."):]] = params[key][k]
                    del params[key][k]
        self.encoder.load_state_dict(params['encoder'])
        self.quantizer.load_state_dict(params['vq'])
        if self.decoder is not None:
            self.decoder.load_state_dict(params['decoder'])
        if self.asr_decoder is not None:
            self.asr_decoder.load_state_dict(params['predictor'], strict=False)

    def forward(self, waves_16k, wave_16k_lens, ssl_model=None):
        ssl_fn = self.ssl_model if self.ssl_model else ssl_model
        assert ssl_fn is not None, "In case in-class SSL model loading is skipped, external ssl_model must be provided"
        waves_16k_input_list = [
            waves_16k[bib, :wave_16k_lens[bib]].cpu().numpy()
            for bib in range(len(waves_16k))
        ]
        alt_inputs = self.ssl_feature_extractor(
            waves_16k_input_list,
            return_tensors='pt',
            return_attention_mask=True,
            padding=True,
            sampling_rate=16000
        ).to(waves_16k.device)
        feature_lens = alt_inputs.data['attention_mask'].sum(-1) // 320  # frame rate of hubert is 50 Hz

        outputs = ssl_fn(
            alt_inputs.input_values,
            attention_mask=alt_inputs.attention_mask,
        )
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states[:, :feature_lens.max(), :]
        feature_lens = feature_lens.clamp(max=last_hidden_states.size(1))
        last_hidden_states = last_hidden_states.transpose(1, 2)
        x_hidden = self.encoder(last_hidden_states, feature_lens)
        x_hidden = x_hidden.transpose(1, 2)
        x_quantized, indices = self.quantizer(x_hidden)[:2]
        return x_quantized, indices, feature_lens