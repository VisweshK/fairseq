# Author: Viswesh Krishna

import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils


class FunnelEncoderLayer(nn.Module):
    """Implement transformer encoder layer with query pooling"""

    def __init__(self, args, block_num, block_id stride=(2, 1), should_pool_query=True):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(
            args, 'quant_noise_pq_block_size', 8) or 8
        # Funnel Args
        self.should_pool_query = should_pool_query
        self.query_dim = self.embed_dim // stride[0]
        self.block_id = block_id
        self.block_num = block_num
        self.pooling_type = getattr(args, 'pooling_type', 'mean')
        # self.pooling_size = getattr(args, 'pooling_size', True)
        self.separate_cls = getattr(args, 'separate_cls', True)
        self.self_attn = self.build_self_attention(
            self.embed_dim, self.query_dim, args)
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

    def build_self_attention(self, embed_dim, query_dim, args):
        return MultiheadAttention(
            query_dim,
            kdim=embed_dim,
            vdim=embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def pool_query(self, x, mode="mean", stride=(2, 1)):
        """Pool vector"""
        if x is None:
            return None

        ndims = x.dims()
        if self.separate_cls:
            if self.args.truncate_seq:
                tensor = torch.cat([tensor[:, :1], tensor[:, :-1]], dim=1)
        else:
            tensor = torch.cat([tensor[:, :1], tensor], dim=1)

        assert ndims == 2 or ndims == 3 or ndims == 4

        if ndims == 2:
            tensor = tensor[:, None, :, None]
        elif ndims == 3:
            tensor = tensor[:, None, :, :]

        if mode == "mean":
            tensor = F.avg_pool2d(
                tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "max":
            tensor = F.max_pool2d(
                tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "min":
            tensor = -F.max_pool2d(
                -tensor, stride, stride=stride, ceil_mode=True)
        else:
            raise NotImplementedError
        if ndims == 2:
            tensor = tensor.squeeze(-1).squeeze(1)
        elif ndims == 3:
            tensor = tensor.squeeze(1)

        return tensor

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """Either adapted self-attention, or normal self-attention"""

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        no_pool = x
        if self.should_pool_query:
            x = self.pool_query(x)

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
