import math
import torch
from torch import nn
from .sublayers import MultiheadAttention, PositionalFeedForwardNetwork
from .inp_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate):
        super().__init__()

        self.attn = MultiheadAttention(d_model, num_heads)
        self.ffn = PositionalFeedForwardNetwork(d_model, dff)

        self.ln1 = nn.LayerNorm(d_model)  # layer norm not batch due to sequence
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):

        attn_out = self.attn(x, x, x, mask)
        x = self.ln1(x + self.dropout1(attn_out))
        fn_out = self.ffn(x)
        x = self.ln2(x + self.dropout2(fn_out))

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, dff, dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):

        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)

        enc = PositionalEncoding(self.d_model, x.size(1))
        x = enc(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x
