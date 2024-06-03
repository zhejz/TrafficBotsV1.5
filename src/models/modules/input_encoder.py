# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
import torch
from torch import Tensor, nn
from .mlp import MLP


class InputEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attr_dim: int,
        pe_dim: int,
        n_layer: int,
        mlp_dropout_p: float,
        mlp_use_layernorm: bool,
        mode: str,
    ) -> None:
        super().__init__()
        self.mode = mode

        if self.mode == "input":
            mlp_in_dim = attr_dim + pe_dim
            mlp_out_dim = hidden_dim
        elif self.mode == "cat":
            mlp_in_dim = attr_dim
            mlp_out_dim = hidden_dim - pe_dim
            assert mlp_out_dim >= 32, f"Make sure pe_dim is smaller than {hidden_dim-32}!"
        elif self.mode == "add":
            mlp_in_dim = attr_dim
            mlp_out_dim = hidden_dim
            assert pe_dim in (0, hidden_dim), f"Make sure pe_dim equals to hidden_dim={hidden_dim} or 0!"

        self.mlp = MLP(
            [mlp_in_dim] + [mlp_out_dim] * n_layer,
            dropout_p=mlp_dropout_p,
            use_layernorm=mlp_use_layernorm,
            end_layer_activation=False,
        )

    def forward(self, attr: Tensor, pe: Optional[Tensor]) -> Tensor:
        """
        Args:
            invalid: [...], bool
            attr: [..., attr_dim], for input to MLP
            pe: [..., pe_dim], for input/cat/add to MLP

        Returns:
            feature: [..., hidden_dim] float32
        """
        if pe is None:
            feature = self.mlp(attr)
        else:
            if self.mode == "input":
                feature = self.mlp(torch.cat([attr, pe], dim=-1))
            elif self.mode == "cat":
                feature = torch.cat([self.mlp(attr), pe], dim=-1)
            elif self.mode == "add":
                feature = self.mlp(attr) + pe

        return feature
