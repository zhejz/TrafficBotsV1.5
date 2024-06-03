# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
from torch import Tensor, nn
import torch
from .mlp import MLP


class AddNaviLatent(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        dummy: bool,
        mode: str,
        n_layer: int,
        mlp_use_layernorm: bool,
        mlp_dropout_p: float,
        res_add: bool = False,
    ) -> None:
        super().__init__()
        assert mode in ["add", "mul", "cat"]
        self.dummy = dummy

        if not self.dummy:
            self.mode = mode
            self.res_add = res_add
            self.mlp_in = MLP(
                [in_dim] + [hidden_dim] * n_layer, use_layernorm=mlp_use_layernorm, dropout_p=mlp_dropout_p
            )
            _dim = hidden_dim * 2 if self.mode == "cat" else hidden_dim
            self.mlp = MLP([_dim] + [hidden_dim] * n_layer, use_layernorm=mlp_use_layernorm, dropout_p=mlp_dropout_p)

    def forward(self, x: Tensor, z: Optional[Tensor], z_valid: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [n_sc, n_ag, hidden_dim]
            z: [n_sc, n_ag, in_dim], latent or goal
            z_valid: [n_sc, n_ag]

        Returns:
            h: [n_sc, n_ag, hidden_dim], x combined with z
        """
        if not self.dummy:

            if z_valid is None:
                z_valid = torch.ones_like(x[:, :, 0], dtype=bool)
            z_invalid = ~z_valid

            z = self.mlp_in(z)

            if self.mode == "add":
                h = x + z.masked_fill(z_invalid.unsqueeze(-1), 0)
            elif self.mode == "mul":
                h = x * z.masked_fill(z_invalid.unsqueeze(-1), 1)
            elif self.mode == "cat":
                h = torch.cat([x, z.masked_fill(z_invalid.unsqueeze(-1), 0)], dim=-1)

            h = self.mlp(h, z_invalid)

            if self.res_add:  # z_valid: h+x, z_invalid: x
                x = h + x
            else:  # z_valid: h, z_invalid: x
                x = h + x.masked_fill(z_valid.unsqueeze(-1), 0)

        return x
