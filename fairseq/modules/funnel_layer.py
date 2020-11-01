# Author: Viswesh Krishna

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor


class FunnelEncoderLayer(nn.Module):
    """Implement transformer encoder layer with query pooling"""

    def __init__(self, args, embed_dim, block_num, block_id, stride, should_pool_query):
        super().__init__()
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(
            args, 'quant_noise_pq_block_size', 8) or 8
        # Funnel Args
        self.stride = stride
        self.should_pool_query = should_pool_query
        self.embed_dim = embed_dim
        self.kv_dim = embed_dim * (
            self.stride if should_pool_query else 1)
        self.block_id = block_id
        self.block_num = block_num
        self.pooling_type = getattr(args, 'pooling_type', 'mean')
        # self.pooling_size = getattr(args, 'pooling_size', True)
        self.separate_cls = getattr(args, 'separate_cls', False)
        self.self_attn = self.build_self_attention(
            self.embed_dim, self.kv_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def build_self_attention(self, embed_dim, kv_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            kdim=kv_dim,
            vdim=kv_dim,
            dropout=args.attention_dropout,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def pool_query(self, x):
        """Pool vector"""
        if x is None:
            return None
        mode = self.pooling_type
        stride = self.stride

        ndims = x.ndim
        # TODO: Add pool query args to parser
        # if self.separate_cls:
        #     if False:
        #         x = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        #     else:
        #         x = torch.cat([x[:, :1], x], dim=1)

        assert ndims == 2 or ndims == 3 or ndims == 4

        # if ndims == 2:
        #     x = x[:, None, :, None]
        # elif ndims == 3:
        #     x = x[:, None, :, :]

        if mode == "mean":
            x = nn.AvgPool1d(stride, stride=stride, ceil_mode=True)(x)
        elif mode == "max":
            x = nn.MaxPool1d(stride, stride=stride, ceil_mode=True)(x)
        elif mode == "min":
            x = -nn.MaxPool1d(stride, stride=stride, ceil_mode=True)(-x)
        else:
            raise NotImplementedError
        # if ndims == 2:
        #     x = x.squeeze(-1).squeeze(1)
        # elif ndims == 3:
        #     x = x.squeeze(1)

        return x

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """Either adapted self-attention, or normal self-attention"""

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        no_pool = x
        if self.should_pool_query:
            print('Before', x.shape)
            x = self.pool_query(x)
            print('After', x.shape)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=no_pool,
            value=no_pool,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x
