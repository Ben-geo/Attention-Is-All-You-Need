import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_padding_mask(
    seq_q: torch.Tensor, seq_k: torch.Tensor, pad_idx: int
) -> torch.Tensor:

    mask = (seq_k == pad_idx).unsqueeze(1)  # Shape: (batch_size, 1, seq_len_k)
    mask = mask.expand(
        -1, seq_q.size(1), -1
    )  # Shape: (batch_size, seq_len_q, seq_len_k)
    return mask.to(device)


def create_look_ahead_mask(seq_len: int) -> torch.Tensor:

    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def create_combined_mask(tgt_input: torch.Tensor, pad_idx: int) -> torch.Tensor:

    tgt_padding_mask = create_padding_mask(
        tgt_input, tgt_input, pad_idx
    )  # Shape: (batch_size, tgt_len, tgt_len)
    look_ahead_mask = create_look_ahead_mask(
        tgt_input.size(1)
    )  # Shape: (tgt_len, tgt_len)
    look_ahead_mask = look_ahead_mask.unsqueeze(0).expand(tgt_input.size(0), -1, -1)
    combined_mask = tgt_padding_mask | look_ahead_mask
    return combined_mask
