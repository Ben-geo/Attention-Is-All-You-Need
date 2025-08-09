import torch
from torch import nn
import math

device = "cuda "  # define it as needed


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, S):
        super().__init__()
        self.d_model = d_model
        self.S = S
        pos_encoder = torch.zeros(S, d_model)

        pos = torch.arange(0, S).float().unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pos_encoder[:, 0::2] = torch.sin(pos * div_term)
        pos_encoder[:, 1::2] = torch.cos(pos * div_term)

        pos_encoder = pos_encoder.unsqueeze(0)
        self.register_buffer("pos_encoder", pos_encoder)

    def forward(self, x):

        x = x + self.pos_encoder.requires_grad_(False).to(device)
        return x
