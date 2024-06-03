# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
from torch import Tensor, nn
import torch
from torch.distributions import Independent, Normal
from .mlp import MLP


class ActionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        n_layer: int,
        mlp_use_layernorm: bool,
        log_std: Optional[float] = None,
        branch_type: bool = False,
        n_ag_type: int = 3,
    ) -> None:
        super().__init__()
        self.branch_type = branch_type
        self.out_dim = action_dim

        if self.branch_type:
            self.mlp_mean = nn.ModuleList(
                [
                    MLP(
                        [hidden_dim] * n_layer + [self.out_dim],
                        end_layer_activation=False,
                        use_layernorm=mlp_use_layernorm,
                    )
                    for _ in range(n_ag_type)
                ]
            )
            if log_std is None:
                self.log_std = None
                self.mlp_log_std = nn.ModuleList(
                    [
                        MLP(
                            [hidden_dim] * n_layer + [self.out_dim],
                            end_layer_activation=False,
                            use_layernorm=mlp_use_layernorm,
                        )
                        for _ in range(n_ag_type)
                    ]
                )
            else:
                self.log_std = nn.ParameterList(
                    [nn.Parameter(log_std * torch.ones(self.out_dim), requires_grad=True) for _ in range(n_ag_type)]
                )

        else:
            self.mlp_mean = MLP(
                [hidden_dim] * n_layer + [self.out_dim], end_layer_activation=False, use_layernorm=mlp_use_layernorm
            )
            if log_std is None:
                self.log_std = None
                self.mlp_log_std = MLP(
                    [hidden_dim] * n_layer + [self.out_dim], end_layer_activation=False, use_layernorm=mlp_use_layernorm
                )
            else:
                self.log_std = nn.Parameter(log_std * torch.ones(self.out_dim), requires_grad=True)

    def forward(self, x: Tensor, valid: Tensor, ag_type: Tensor) -> Independent:
        """
        Args:
            x: [n_sc, n_ag, hidden_dim]
            valid: [n_sc, n_ag], bool
            ag_type: [n_sc, n_ag, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]

        Returns:
            action_dist.mean: [n_sc, n_ag, action_dim]
        """
        n_sc, n_ag, n_type = ag_type.shape
        if self.branch_type:
            mask_type = ~(ag_type & valid.unsqueeze(-1))  # [n_sc, n_ag, 3]
            mean = 0
            for i in range(n_type):
                mean += self.mlp_mean[i](x, mask_type[:, :, i])  # [n_sc, n_ag, self.out_dim]

            log_std = 0
            if self.log_std is None:
                for i in range(n_type):
                    log_std += self.mlp_log_std[i](x, mask_type[:, :, i])
            else:
                for i in range(n_type):
                    # [n_sc, n_ag, self.out_dim]
                    log_std += (
                        self.log_std[i][None, None, :].expand(n_sc, n_ag, -1).masked_fill(mask_type[:, :, [i]], 0)
                    )
        else:
            invalid = ~valid
            mean = self.mlp_mean(x, invalid)  # [n_sc, n_ag, self.out_dim]
            if self.log_std is None:
                log_std = self.mlp_log_std(x, invalid)  # [n_sc, n_ag, self.out_dim]
            else:
                # [self.out_dim] -> [n_sc, n_ag, self.out_dim]
                log_std = self.log_std[None, None, :].expand(n_sc, n_ag, -1)

        return Independent(Normal(mean, log_std.exp()), 1)
