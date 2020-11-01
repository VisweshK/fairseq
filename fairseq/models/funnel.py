# Author: Viswesh Krishna

import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models import (
    TransformerEncoder,
    TransformerModel,
)
from fairseq.modules import (
    FunnelEncoderLayer,
)


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

        self.stride = getattr(args, 'stride', (2, 1))
        # Recreate Layers
        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        self.layers.extend(
            [build_funnel_encoder_layer(args, block_num, block_id, stride)
             for block_id in range(args.encoder_layers)
             for block_num in range(args.num_blocks)]
        )
        self.num_layers = len(self.layers)

    def build_funnel_encoder_layer(self, args, block_num, block_id, stride):
        return FunnelEncoderLayer(args, block_num, block_id, stride)
