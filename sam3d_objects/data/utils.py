# Copyright (c) Meta Platforms, Inc. and affiliates.
import optree
import torch
from torch.utils import _pytree


def tree_tensor_map(fn, tree, *rest):
    return optree.tree_map(
        fn,
        tree,
        *rest,
        is_leaf=lambda x: isinstance(x, torch.Tensor),
        none_is_leaf=False,
    )


def tree_reduce_unique(fn, tree, ensure_unique=True, **kwargs):
    values = _pytree.tree_flatten(tree, **kwargs)[0]
    values = tuple(map(fn, values))
    first = values[0]
    if ensure_unique:
        for value in values[1:]:
            if value != first:
                raise RuntimeError(
                    f"different values found, {value} and {first} should be the same"
                )
    return first
