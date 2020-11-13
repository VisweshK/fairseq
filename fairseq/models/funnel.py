# Author: Viswesh Krishna

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from .transformer import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
)
from fairseq.modules import (
    FunnelEncoderLayer,
)


@register_model("funnel_transformer")
class FunnelTransformer(TransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--num-blocks', type=int, metavar='N',
                            help='number of blocks in encoder.')
        parser.add_argument('--stride', type=int, metavar='N', default=2,
                            help='pooling stride during downsample.')
        parser.add_argument('--should-upsample', action='store_true',
                            help='upsample with skip connection')
        parser.add_argument('--feature-compress', default=False, action='store_true',
                            help="should compress along feature dimension.")
        parser.add_argument('--time-compress', default=False, action='store_true',
                            help="should compress along time dimension.")
        parser.add_argument('--feature-compress-type', type=str, default="mean", 
                            help="type of feature compression to use.")
        parser.add_argument('--time-compress-type', type=str, default="mean", 
                            help="type of time compression to use.")

    @classmethod
    def build_model(cls, args, task):
        """
        Build new model instance.
        """

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError(
                    "--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = TransformerModel.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = super(cls, cls).build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = super(cls, cls).build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = super(cls, cls).build_decoder(
            args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return FunnelEncoder(args, src_dict, embed_tokens)


class FunnelEncoder(TransformerEncoder):
    """
    Funnel Transformer encoder consisting *args.num_blocks* where each block
    consists of *args.encoder_layers* layers. Each layer is a
    :class:`FunnelEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        assert args.encoder_embed_dim % args.stride == 0
        self.embed_dim = args.encoder_embed_dim
        self.stride = args.stride
        self.num_blocks = args.num_blocks
        self.encoder_layers = args.encoder_layers
        self.should_time_compress = args.time_compress
        self.should_feature_compress = args.feature_compress
        self.should_upsample = args.upsample
        if self.should_time_compress:
            self.compress_encoder_padding_mask_fn = nn.MaxPool1d(
                    self.stride, stride=self.stride, ceil_mode=True)
        # Recreate Layers with Funnel Encoders
        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        for block_num in range(self.num_blocks):
            for block_id in range(self.encoder_layers):
                self.layers.append(
                    self.build_funnel_encoder_layer(
                        args,
                        self.embed_dim // (self.stride ** block_num) if self.should_feature_compress else self.embed_dim,
                        block_num,
                        block_id,
                        self.stride,
                        (block_id == 0 and block_num != 0)
                    )
                )
        self.num_layers = len(self.layers)

    def build_funnel_encoder_layer(self, args, embed_dim, block_num, block_id, stride, should_pool_query):
        return FunnelEncoderLayer(args, embed_dim, block_num, block_id, stride, should_pool_query)
    
    def time_compress_encoder_padding_mask(self, mask):
        """Max pool along time axis"""
        mask = torch.unsqueeze(mask, 0).to(dtype=torch.float32)
        mask = self.compress_encoder_padding_mask_fn(mask) > 0
        return torch.squeeze(mask, 0)

    def upsample(self, x):
        if self.should_feature_compress:
            return nn.Upsample(scale_factor=self.stride ** (self.num_blocks - 1), mode="nearest")(x)
        elif self.should_time_compress:
            return nn.Upsample(scale_factor=self.stride ** (self.num_blocks - 1), mode="nearest")(x.permute(1, 2, 0)).permute(2, 0, 1)
        else:
            return x
        

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(
            src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if layer.block_id == self.encoder_layers - 1 and self.should_time_compress:
                encoder_padding_mask = self.time_compress_encoder_padding_mask(encoder_padding_mask)
            if layer.block_id == self.encoder_layers - 1 and layer.block_num == 0:
                residual = x
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.should_upsample:
            x = residual + self.upsample(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


@ register_model_architecture("funnel_transformer", "funnel_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(
        args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(
        args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(
        args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)

    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(
        args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@ register_model_architecture("funnel_transformer", "funnel_transformer_iwslt_de_en")
def funnel_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.num_blocks = getattr(args, "num_blocks", 4)
    args.upsample = getattr(args, "upsample", True)
    args.feature_compress = getattr(args, "feature_compress", False)
    args.time_compress = getattr(args, "time_compress", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)
