from torch import nn

from .latent_encoder import LatentEncoder


def add_vista4d_modules(model):
    model.vista4d = True
    model.latent_encoder = LatentEncoder(
        source_init_mode="wan_patch_embed",
        point_cloud_init_mode="wan_patch_embed",
        mask_init_mode="zero_init",
        use_source_masks=True,
        use_point_cloud_masks=True,
        wan_patch_embedding=model.patch_embedding,
        rgb_in_channels=model.in_dim,
        mask_in_channels=2 * 4 * 8 * 8,
    )
    hidden_dim = model.blocks[0].self_attn.q.weight.shape[0]
    for block in model.blocks:
        block.cam_encoder = nn.Linear(6, hidden_dim)
        block.projector = nn.Linear(hidden_dim, hidden_dim)
        if not block.cam_encoder.weight.is_meta:
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
        if not block.projector.weight.is_meta:
            nn.init.eye_(block.projector.weight)
            nn.init.zeros_(block.projector.bias)
