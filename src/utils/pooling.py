# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
import torch
from torch import Tensor


def seq_pooling(x: Tensor, invalid: Tensor, mode: str, valid: Optional[Tensor] = None) -> Tensor:
    """
    Args:
        x: [n_sc, n_ag, n_step, hidden_dim] or [n_sc, n_mp, n_mp_pl_node, hidden_dim]
        invalid: [n_sc, n_ag, n_step]
        mode: one of {"max", "last", "max_valid", "last_valid", "mean_valid"}
        valid: [n_sc, n_ag, n_step], ~invalid, just for efficiency

    Returns:
        x_pooled: [n_sc, n_ag, hidden_dim]
    """
    if mode == "max_valid":
        x_pooled = x.masked_fill(invalid.unsqueeze(-1), float("-inf")).amax(2)
    elif mode == "first":
        x_pooled = x[:, :, 0]
    elif mode == "last":
        x_pooled = x[:, :, -1]
    elif mode == "last_valid":
        n_sc, n_ag, n_step = invalid.shape
        if valid is None:
            valid = ~invalid
        idx_last_valid = n_step - 1 - torch.max(valid.flip(2), dim=2)[1]
        x_pooled = x[torch.arange(n_sc).unsqueeze(1), torch.arange(n_ag).unsqueeze(0), idx_last_valid]
    elif mode == "mean_valid":
        if valid is None:
            valid = ~invalid
        x_pooled = x.masked_fill(invalid.unsqueeze(-1), 0.0).sum(2)
        x_pooled = x_pooled / (valid.sum(2, keepdim=True) + torch.finfo(x.dtype).eps)
    else:
        raise NotImplementedError

    return x_pooled.masked_fill(invalid.all(-1, keepdim=True), 0)

