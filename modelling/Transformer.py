import torch.nn as nn
from modelling.functional import BaseTransformerLayer, TransformerDecoderLayer
from modelling.positional_encoding import PositionalEncoding
import torch

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_len, device=torch.device('cpu')):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, device)

        self.transformer_encoder = nn.ModuleList([BaseTransformerLayer(d_model, n_heads, dim_feedforward, dropout, device) for _ in range(num_encoder_layers)])
        self.transformer_decoder = nn.ModuleList([TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout, device) for _ in range(num_decoder_layers)])

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.pos_encoder(self.embedding(src))
        tgt = self.pos_encoder(self.embedding(tgt))

        for layer in self.transformer_encoder:
            src = layer(src, src_mask)

        for layer in self.transformer_decoder:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        return self.output_layer(tgt) 
        