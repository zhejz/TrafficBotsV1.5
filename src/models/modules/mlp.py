# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import List, Tuple, Union, Optional
from torch import Tensor, nn


def _get_activation(activation: str, inplace: bool) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU(inplace=inplace)
    elif activation == "elu":
        return nn.ELU(inplace=inplace)
    elif activation == "rrelu":
        return nn.RReLU(inplace=inplace)
    raise RuntimeError("activation {} not implemented".format(activation))


class MLP(nn.Module):
    def __init__(
        self,
        fc_dims: Union[List, Tuple],
        dropout_p: float = -1.0,
        activation: str = "relu",
        end_layer_activation: bool = True,
        init_weight_norm: bool = False,
        init_bias: Optional[float] = None,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        assert len(fc_dims) >= 2
        assert not (use_layernorm and use_batchnorm)
        layers: List[nn.Module] = []
        for i in range(0, len(fc_dims) - 1):
            fc = nn.Linear(fc_dims[i], fc_dims[i + 1])

            if init_weight_norm:
                fc.weight.data *= 1.0 / fc.weight.norm(dim=1, p=2, keepdim=True)
            if init_bias is not None and i == len(fc_dims) - 2:
                fc.bias.data *= 0
                fc.bias.data += init_bias

            layers.append(fc)

            if (i < len(fc_dims) - 2) or (i == len(fc_dims) - 2 and end_layer_activation):
                if use_layernorm:
                    layers.append(nn.LayerNorm(fc_dims[i + 1]))
                elif use_batchnorm:
                    layers.append(nn.BatchNorm1d(fc_dims[i + 1]))
                layers.append(_get_activation(activation, inplace=True))

            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))

        self.input_dim = fc_dims[0]
        self.output_dim = fc_dims[-1]
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, mask_invalid: Optional[Tensor] = None, fill_invalid: float = 0.0) -> Tensor:
        """
        Args:
            x: [..., input_dim]
            mask_invalid: [...]
        Returns:
            x: [..., output_dim]
        """
        x = self.fc_layers(x.flatten(0, -2)).view(*x.shape[:-1], self.output_dim)
        if mask_invalid is not None:
            x = x.masked_fill(mask_invalid.unsqueeze(-1), fill_invalid)
        return x
