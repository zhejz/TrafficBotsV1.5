# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from typing import List
from torch import Tensor, nn
from omegaconf import DictConfig
from .mlp import MLP
from .transformer_rpe import TransformerBlockRPE
from utils.pooling import seq_pooling


class PolylineEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        tf_cfg: DictConfig,
        n_layer: int,
        mlp_use_layernorm: bool,
        mlp_dropout_p: float,
        use_pointnet: bool,
        pooling_mode: str,
    ) -> None:
        super().__init__()
        self.use_pointnet = use_pointnet
        self.pooling_mode = pooling_mode

        if self.use_pointnet:
            mlp_layers: List[nn.Module] = []
            for _ in range(n_layer):
                mlp_layers.append(
                    MLP([hidden_dim, hidden_dim // 2], dropout_p=mlp_dropout_p, use_layernorm=mlp_use_layernorm)
                )
            self.mlp_layers = nn.ModuleList(mlp_layers)
        else:
            self.transformer = TransformerBlockRPE(n_layer=n_layer, mode="enc_self_attn", d_rpe=-1, **tf_cfg)

    def forward(self, x: Tensor, invalid: Tensor) -> Tensor:
        """c.f. VectorNet and SceneTransformer, Aggregate polyline/track level feature.

        Args:
            x: [n_sc, n_mp, n_mp_pl_node, hidden_dim]
            invalid: [n_sc, n_mp, n_mp_pl_node]

        Returns:
            emb: [n_sc, n_mp, hidden_dim]
        """
        n_sc, n_mp, n_mp_pl_node = invalid.shape

        # ! interaction
        if self.use_pointnet:  # vectornet
            for mlp in self.mlp_layers:
                x = mlp(x, invalid, float("-inf"))  # [n_sc, n_mp, n_mp_pl_node, hidden_dim//2]
                x = torch.cat((x, x.amax(dim=2, keepdim=True).expand(-1, -1, n_mp_pl_node, -1)), dim=-1)
                x.masked_fill_(invalid.unsqueeze(-1), 0)
        else:  # transformer
            x = self.transformer(
                src=x.flatten(0, 1),  # [n_sc*n_mp, n_mp_pl_node, hidden_dim]
                src_padding_mask=invalid.flatten(0, 1),  # [n_sc*n_mp, n_mp_pl_node]
            )[0].view(n_sc, n_mp, n_mp_pl_node, -1)

        # ! pooling
        emb = seq_pooling(x, invalid, self.pooling_mode)

        return emb
