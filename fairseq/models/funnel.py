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
        super().init(args, encoder, decoder)

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

    @classmethod
    def build_model(cls, args, task):
        """
        Build new model instance.
        """
        super(TransformerModel, self).build_model(cls, args, task)
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, encoder_embed_tokens)
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

        self.stride = (args.stride, 1)
        self.num_blocks = args.num_blocks
        self.encoder_layers = args.encoder_layers
        # Recreate Layers with Funnel Encoders
        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        self.layers.extend(
            [build_funnel_encoder_layer(args, block_num, block_id, self.stride, (block_id == 0 and block_num != 0))
             for block_id in range(self.encoder_layers)
             for block_num in range(self.num_blocks)]
        )
        self.num_layers = len(self.layers)

    def build_funnel_encoder_layer(self, args, block_num, block_id, stride, should_pool_query):
        return FunnelEncoderLayer(args, block_num, block_id, stride, should_pool_query)

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
            if layer.block_id == self.args.layers - 1 and layer.block_num == 0:
                residual = x
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.upsample:
            x = residual + \
                nn.Upsample(
                    scale_factor=self.stride[0] ** (self.num_blocks - 1), mode="nearest")

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


@register_model_architecture("funnel_transformer", "funnel_transformer_iwslt_de_en")
def funnel_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.num_blocks = getattr(args, "num_blocks", 6)
    args.num_blocks = getattr(args, "upsample", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)
