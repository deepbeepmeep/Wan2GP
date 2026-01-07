from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)


def parallelize_dit(model, tp_mesh):
    if tp_mesh.size() > 1:
        plan = {
            "in_layer": ColwiseParallel(),
            "out_layer": RowwiseParallel(
                output_layouts=Replicate(),
            ),
        }
        parallelize_module(model.time_embeddings, tp_mesh, plan)

        plan = {
            "in_layer": ColwiseParallel(
                output_layouts=Replicate(),
            )
        }
        parallelize_module(model.text_embeddings, tp_mesh, plan)
        parallelize_module(model.pooled_text_embeddings, tp_mesh, plan)
        parallelize_module(model.visual_embeddings, tp_mesh, plan)

        for visual_transformer_block in model.visual_transformer_blocks:
            plan = {
                "visual_modulation": PrepareModuleInput(
                    input_layouts=(None),
                    desired_input_layouts=(Replicate()),
                ),
                "visual_modulation.out_layer": ColwiseParallel(
                    output_layouts=Replicate(),
                ),
                "self_attention_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "self_attention.to_query": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "self_attention.to_key": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "self_attention.to_value": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "self_attention.query_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "self_attention.key_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "self_attention.out_layer": RowwiseParallel(
                    output_layouts=Replicate(),
                ),
                "cross_attention_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "cross_attention.to_query": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "cross_attention.to_key": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "cross_attention.to_value": ColwiseParallel(
                    input_layouts=Replicate(),
                ),
                "cross_attention.query_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "cross_attention.key_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "cross_attention.out_layer": RowwiseParallel(
                    output_layouts=Replicate(),
                ),
                "feed_forward_norm": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
                "feed_forward.in_layer": ColwiseParallel(),
                "feed_forward.out_layer": RowwiseParallel(),
            }
            self_attn = visual_transformer_block.self_attention
            self_attn.num_heads = self_attn.num_heads // tp_mesh.size()

            cross_attn = visual_transformer_block.cross_attention
            cross_attn.num_heads = cross_attn.num_heads // tp_mesh.size()

            parallelize_module(visual_transformer_block, tp_mesh, plan)

        plan = {
            "out_layer": ColwiseParallel(
                output_layouts=Replicate(),
            ),
        }
        parallelize_module(model.out_layer, tp_mesh, plan)

    return model


def parallelize_seq(model, tp_mesh):
    if tp_mesh.size() > 1:
        plan_in = {
            "out_layer": PrepareModuleInput(
                    input_layouts=(Replicate(), None, None),
                    desired_input_layouts=(Shard(1), None, None),
                    use_local_output=True
                ),
            }
        parallelize_module(model, tp_mesh, plan_in)
        plan_out = {
            "visual_embeddings": PrepareModuleOutput(
                output_layouts=(Shard(1)),
                desired_output_layouts=(Replicate()),
                )
        }
        parallelize_module(model, tp_mesh, plan_out)

        for i, block in enumerate(model.visual_transformer_blocks):
            plan = {
                "self_attention_norm": SequenceParallel(sequence_dim=0, use_local_output=True),
                "self_attention.to_query": PrepareModuleOutput(
                    output_layouts=(Shard(0)), desired_output_layouts=(Shard(-1))
                ),
                "self_attention.to_key": PrepareModuleOutput(
                    output_layouts=(Shard(0)), desired_output_layouts=(Shard(-1))
                ),
                "self_attention.to_value": PrepareModuleOutput(
                    output_layouts=(Shard(0)), desired_output_layouts=(Shard(-1))
                ),
                "self_attention.out_layer": PrepareModuleInput(
                    input_layouts=(Shard(-1)),
                    desired_input_layouts=(Shard(0)),
                    use_local_output=True,
                ),
                "cross_attention_norm": SequenceParallel(sequence_dim=0, use_local_output=True),
                "cross_attention.to_query": PrepareModuleOutput(
                    output_layouts=(Shard(0)), desired_output_layouts=(Shard(-1))
                ),
                "cross_attention.to_key": PrepareModuleOutput(
                    output_layouts=(Replicate()), desired_output_layouts=(Shard(-1))
                ),
                "cross_attention.to_value": PrepareModuleOutput(
                    output_layouts=(Replicate()), desired_output_layouts=(Shard(-1))
                ),
                "cross_attention.out_layer": PrepareModuleInput(
                    input_layouts=(Shard(-1)),
                    desired_input_layouts=(Shard(0)),
                    use_local_output=True,
                ),
                "feed_forward_norm": SequenceParallel(sequence_dim=0, use_local_output=True),
            }
            self_attn = block.self_attention
            self_attn.num_heads = self_attn.num_heads // tp_mesh.size()
            cross_attn = block.cross_attention
            cross_attn.num_heads = cross_attn.num_heads // tp_mesh.size()
            parallelize_module(block, tp_mesh, plan)
            #shard input of first block and idx for all blocks
            if i == 0:
                parallelize_module(
                    block,
                    tp_mesh,
                    PrepareModuleInput(
                        input_layouts=(Replicate(),None,None,None,None, None),
                        desired_input_layouts=(Shard(0),None,None,None,None, None),
                        use_local_output=True,
                    ),
                )

            if i == len(model.visual_transformer_blocks)-1:
                parallelize_module(
                    block,
                    tp_mesh,
                    PrepareModuleOutput(
                        output_layouts=(Shard(0)),
                        desired_output_layouts=(Replicate())
                    ),
                )        

    return model
