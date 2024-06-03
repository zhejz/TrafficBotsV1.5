# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple, Any
import torch
from torch import Tensor, nn


class MultiAgentGRULoop(nn.Module):
    def __init__(self, hidden_dim: int, n_layer: int, dropout_p: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.rnn = nn.GRU(hidden_dim, hidden_dim, n_layer, dropout=dropout_p)

    def forward(self, x: Tensor, invalid: Tensor, h: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [n_sc, n_ag, (n_step), hidden_dim]
            invalid: [n_sc, n_ag, (n_step)]
            h: [n_layer, n_sc * n_ag, hidden_dim] or None
        Returns:
            x_1: [n_sc, n_ag, (n_step), hidden_dim]
            h_1: [n_layer, n_sc * n_ag, hidden_dim]
        """
        n_sc, n_ag = invalid.shape[0], invalid.shape[1]
        if h is None:
            h = torch.zeros((self.n_layer, n_sc * n_ag, self.hidden_dim), device=x.device)

        if invalid.dim() == 3:
            n_step = invalid.shape[2]
            x_1 = []
            # [n_step, n_sc * n_ag, hidden_dim]
            x = x.movedim(2, 0).flatten(start_dim=1, end_dim=2)
            # [n_step, n_sc * n_ag, 1]
            invalid = invalid.movedim(2, 0).flatten(start_dim=1, end_dim=2).unsqueeze(-1)
            for k in range(n_step):
                x_out, h = self.rnn(x[[k]], h)
                h = h.masked_fill(invalid[[k]], 0.0)
                x_1.append(x_out)  # [1, n_sc * n_ag, hidden_dim]
            x_1 = torch.cat(x_1, dim=0)  # [n_step, n_sc * n_ag, hidden_dim]
            x_1 = x_1.masked_fill(invalid, 0.0).view(n_step, n_sc, n_ag, self.hidden_dim).movedim(0, 2)
            h_1 = None
        elif invalid.dim() == 2:
            input = x.flatten(start_dim=0, end_dim=1).unsqueeze(0)
            input, h = self.rnn(input, h)
            invalid = invalid.flatten(start_dim=0, end_dim=1).unsqueeze(-1)  # [n_sc * n_ag, 1]
            h_1 = h.masked_fill(invalid.unsqueeze(0), 0.0)
            x_1 = input[0].masked_fill(invalid, 0.0).view(n_sc, n_ag, self.hidden_dim)
        return x_1, h_1
