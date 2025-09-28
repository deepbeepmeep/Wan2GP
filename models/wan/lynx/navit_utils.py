# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 

import torch

def vector_to_list(tensor, lens, dim):
    return list(torch.split(tensor, lens, dim=dim))

def list_to_vector(tensor_list, dim):
    lens = [tensor.shape[dim] for tensor in tensor_list]
    tensor = torch.cat(tensor_list, dim)
    return tensor, lens

def merge_token_lists(list1, list2, dim):
    assert(len(list1) == len(list2))
    return [torch.cat((t1, t2), dim) for t1, t2 in zip(list1, list2)]