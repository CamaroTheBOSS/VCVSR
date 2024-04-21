import collections

from itertools import repeat
from torch import nn


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


def count_parameters(module: nn.Module):
    n_params = sum(p.numel() for p in module.parameters())
    for suffix in ["k", "mln", "mld"]:
        n_params /= 1000
        if n_params < 1000:
            return f"{round(n_params, 2)}{suffix}"
    return
