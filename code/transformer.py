import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder
from .mask import create_combined_mask,create_padding_mask

class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model,):
        super().__init__()

        num_layers = 6
        d_model = 512
        num_heads = 8
        dff = 2048
        dropout_rate = 0.1
        self.encoder = Encoder(src_vocab_size,num_layers,d_model,num_heads,dff,dropout_rate)
        self.decoder = Decoder(tgt_vocab_size,num_layers,d_model,num_heads,dff,dropout_rate)
        self.fn = nn.Linear(d_model,tgt_vocab_size)

    def forward(self,src,tgt):

        
        # pad the src
        src_mask = create_padding_mask(src,src,0)
        enc_output = self.encoder(src,src_mask)

        # padd self and also look ahead
        tgt_look_ahead_mask = create_combined_mask(tgt,0)
        tgt_src_mask = create_padding_mask(tgt,src,0)
        dec_output = self.decoder(tgt,enc_output,tgt_look_ahead_mask,tgt_src_mask)

        return self.fn(dec_output)

        