import torch
from torch import nn
from .sublayers import MultiheadAttention,PositionalFeedForwardNetwork
import math
from .inp_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,dff,dropout_rate):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model,num_heads)
        self.cross_attn = MultiheadAttention(d_model,num_heads)
        
        self.ffn = PositionalFeedForwardNetwork(d_model,dff)

        self.ln1 = nn.LayerNorm(d_model) # layer norm not batch due to sequence
        self.ln2 = nn.LayerNorm(d_model) 
        self.ln3 = nn.LayerNorm(d_model) 
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        

    def forward(self,x,encoder_output,look_ahead_mask,padding_mask):
        #decoder self attention with look ahead mask
        self_attn_out = self.self_attn(q=x,k=x,v=x,mask = look_ahead_mask)
        x = self.ln1(x+self.dropout1(self_attn_out))
        

        #cross attention each word pays attention to encoder output
        cross_attn_out = self.cross_attn(q=x,k=encoder_output,v=encoder_output,mask = padding_mask)
        x = self.ln2(x+self.dropout2(cross_attn_out))


        fn_out = self.ffn(x)
        x = self.ln3(x+self.dropout3(fn_out))

        return x

class Decoder(nn.Module):

    def __init__(self,vocab_size,num_layers,d_model,num_heads,dff,dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.decoder_layers = nn.ModuleList(DecoderLayer(d_model,num_heads,dff,dropout_rate) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self,x,encoder_output,look_ahead_mask,padding_mask):

        x = self.embedding(x)*math.sqrt(self.d_model)
        
        enc = PositionalEncoding(self.d_model,x.size(1))
        x = enc(x)
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x,encoder_output,look_ahead_mask,padding_mask)
        return x