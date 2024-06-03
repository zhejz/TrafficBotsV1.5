# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from typing import Optional, Tuple
from torch import Tensor, nn
from torch.nn import functional as F
from .attention_rpe import AttentionRPE


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "elu":
        return F.elu
    raise RuntimeError("activation should be relu/gelu/elu, not {}".format(activation))


class TransformerBlockRPE(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        d_model: int,
        n_head: int = 4,
        k_feedforward: int = 4,
        dropout_p: float = 0.1,
        bias: bool = True,
        activation: str = "relu",
        out_layernorm: bool = False,
        apply_q_rpe: bool = False,
        n_layer: int = 1,
        mode: str = "enc_self_attn",
        d_rpe: int = -1,
    ) -> None:
        super().__init__()
        assert mode in ("enc_self_attn", "enc_cross_attn", "dec_cross_attn")
        self.mode = mode
        self.layers = nn.ModuleList(
            [
                TransformerRPE(d_model, n_head, k_feedforward, dropout_p, bias, activation, mode, d_rpe, apply_q_rpe)
                for _ in range(n_layer)
            ]
        )

        self.out_layernorm = nn.LayerNorm(d_model) if out_layernorm else None

    def forward(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        rpe: Optional[Tensor] = None,
        decoder_tgt: Optional[Tensor] = None,
        decoder_tgt_padding_mask: Optional[Tensor] = None,
        decoder_rpe: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            src: [n_batch, n_src, d_model]
            src_padding_mask: [n_batch, n_src], bool, if True, src is invalid.
            tgt: [n_batch, n_tgt, d_model]: cross attn, None: self attn, [n_batch, n_src, n_knn]: rpe/knn self attn.
            tgt_padding_mask: [n_batch, n_tgt] or None or [n_batch, n_src, n_knn]. if True, tgt is invalid
            rpe: [n_batch, n_src, n_tgt, d_rpe]
            decoder_tgt: [n_batch, (n_src), n_tgt_decoder, d_model], for decoder self-attn (n_src) if using rpe or knn.
            decoder_tgt_padding_mask: [n_batch, (n_src), n_tgt_decoder] (n_src) if using rpe or knn.
            decoder_rpe: [n_batch, n_src, n_tgt_decoder, d_rpe]
            attn_mask: [n_batch, n_src, n_tgt], bool, if True, attn is disabled for that pair of src/tgt.

        Returns:
            src: [n_batch, n_src, d_model]
            attn_weights: [n_batch, n_src, n_tgt] if need_weights else None

        Remarks:
            absoulte_pe should be already added to src/tgt.
        """
        attn_weights = None

        if self.mode == "enc_self_attn":
            n_batch, n_src, _ = src.shape
            _idx_batch = torch.arange(n_batch)[:, None, None]  # [n_batch, 1, 1]
            _idx_src = torch.arange(n_src)[None, :, None]  # [1, n_src, 1]
            for mod in self.layers:
                if tgt is not None and tgt.dtype == torch.int64:
                    _tgt = src.unsqueeze(1).expand(-1, n_src, -1, -1)[_idx_batch, _idx_src, tgt]
                else:
                    _tgt = tgt
                src, attn_weights = mod(
                    src=src,
                    src_padding_mask=src_padding_mask,
                    tgt=_tgt,
                    tgt_padding_mask=tgt_padding_mask,
                    rpe=rpe,
                    attn_mask=attn_mask,
                    need_weights=need_weights,
                )
        elif self.mode == "enc_cross_attn":
            for mod in self.layers:
                src, attn_weights = mod(
                    src=src,
                    src_padding_mask=src_padding_mask,
                    tgt=tgt,
                    tgt_padding_mask=tgt_padding_mask,
                    rpe=rpe,
                    attn_mask=attn_mask,
                    need_weights=need_weights,
                )
        elif self.mode == "dec_cross_attn":
            n_batch, n_src, _ = src.shape
            _idx_batch = torch.arange(n_batch)[:, None, None]  # [n_batch, 1, 1]
            _idx_src = torch.arange(n_src)[None, :, None]  # [1, n_src, 1]
            for mod in self.layers:
                if decoder_tgt is not None and decoder_tgt.dtype == torch.int64:
                    _decoder_tgt = src.unsqueeze(1).expand(-1, n_src, -1, -1)[_idx_batch, _idx_src, decoder_tgt]
                else:
                    _decoder_tgt = decoder_tgt
                src, attn_weights = mod(
                    src=src,
                    src_padding_mask=src_padding_mask,
                    tgt=tgt,
                    tgt_padding_mask=tgt_padding_mask,
                    rpe=rpe,
                    decoder_tgt=_decoder_tgt,
                    decoder_tgt_padding_mask=decoder_tgt_padding_mask,
                    decoder_rpe=decoder_rpe,
                    attn_mask=attn_mask,
                    need_weights=need_weights,
                )

        if self.out_layernorm is not None:
            src = self.out_layernorm(src)
        return src, attn_weights


class TransformerRPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        k_feedforward: int,
        dropout_p: float,
        bias: bool,
        activation: str,
        mode: str,
        d_rpe: int = -1,
        apply_q_rpe: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else None
        self.activation = _get_activation_fn(activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm_tgt = nn.LayerNorm(d_model)

        if self.mode == "dec_cross_attn":
            self.attn_src = AttentionRPE(
                d_model=d_model, n_head=n_head, dropout_p=dropout_p, bias=bias, d_rpe=d_rpe, apply_q_rpe=apply_q_rpe
            )
            self.norm_src = nn.LayerNorm(d_model)
            self.dropout_src = nn.Dropout(p=dropout_p) if dropout_p > 0 else None

        self.attn = AttentionRPE(
            d_model=d_model, n_head=n_head, dropout_p=dropout_p, bias=bias, d_rpe=d_rpe, apply_q_rpe=apply_q_rpe
        )
        self.linear1 = nn.Linear(d_model, k_feedforward * d_model)
        self.linear2 = nn.Linear(k_feedforward * d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout_p) if dropout_p > 0 else None
        self.dropout2 = nn.Dropout(p=dropout_p) if dropout_p > 0 else None

    def forward(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        rpe: Optional[Tensor] = None,
        decoder_tgt: Optional[Tensor] = None,
        decoder_tgt_padding_mask: Optional[Tensor] = None,
        decoder_rpe: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            src: [n_batch, n_src, d_model]
            src_padding_mask: [n_batch, n_src], bool, if True, src is invalid.
            tgt: [n_batch, (n_src), n_tgt, d_model], None for self attention, (n_src) if using rpe or knn.
            tgt_padding_mask: [n_batch, (n_src), n_tgt], bool, if True, tgt is invalid, (n_src) if using rpe or knn.
            rpe: [n_batch, n_src, n_tgt, d_rpe]
            decoder_tgt: [n_batch, (n_src), n_tgt_decoder, d_model], for decoder self-attn (n_src) if using rpe or knn.
            decoder_tgt_padding_mask: [n_batch, (n_src), n_tgt_decoder] (n_src) if using rpe or knn.
            decoder_rpe: [n_batch, n_src, n_tgt_decoder, d_rpe]
            attn_mask: [n_batch, n_src, n_tgt], bool, if True, attn is disabled for that pair of src/tgt.

        Returns:
            out: [n_batch, n_src, d_model]
            attn_weights: [n_batch, n_src, n_tgt] if need_weights else None

        Remarks:
            absoulte_pe should be already added to src/tgt.
        """
        if self.mode == "dec_cross_attn":  # transformer self-attn for decoder cross attn
            _s = self.norm_src(src)
            if decoder_tgt is None:
                decoder_tgt_padding_mask = src_padding_mask
            else:
                decoder_tgt = self.norm_src(decoder_tgt)
            _s, _ = self.attn_src(_s, decoder_tgt, tgt_padding_mask=decoder_tgt_padding_mask, rpe=decoder_rpe)
            src = src + _s if self.dropout_src is None else src + self.dropout_src(_s)

        src2 = self.norm1(src)
        if tgt is None:
            tgt_padding_mask = src_padding_mask
        else:
            if self.mode == "enc_self_attn":
                tgt = self.norm1(tgt)
            else:
                tgt = self.norm_tgt(tgt)

        # [n_batch, n_src, d_model]
        src2, attn_weights = self.attn(
            src=src2,
            tgt=tgt,
            tgt_padding_mask=tgt_padding_mask,
            attn_mask=attn_mask,
            rpe=rpe,
            need_weights=need_weights,
        )

        src = src + src2 if self.dropout is None else src + self.dropout(src2)
        src2 = self.norm2(src)
        src2 = self.activation(self.linear1(src2))
        src2 = self.linear2(src2) if self.dropout1 is None else self.linear2(self.dropout1(src2))
        src = src + src2 if self.dropout2 is None else src + self.dropout2(src2)

        if src_padding_mask is not None:
            src.masked_fill_(src_padding_mask.unsqueeze(-1), 0.0)
            if need_weights:
                attn_weights.masked_fill_(src_padding_mask.unsqueeze(-1), 0.0)
        return src, attn_weights
