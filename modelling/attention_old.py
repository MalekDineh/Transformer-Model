import torch
import numpy as np

# 0 not padded
# 1 padded

class Attention(torch.nn.Module):
    def __init__(self, mask_future=False):
        super().__init__()
        self.mask_future = mask_future
    def forward(self, Q, K, V, padding_mask=None):
        d_model = Q.shape[-1] # Embedding dimension
        n_q = Q.shape[1] # Length of input
        n_k = K.shape[1] # Length of output

        # Calculate scaling value
        scaling_factor = 1 / torch.tensor(d_model).sqrt()

        # Calculate modelling logits
        attn_logits = torch.bmm(Q, K.transpose(1,2)) * scaling_factor

        # Apply future mask
        if self.mask_future:
            future_mask = torch.tril(torch.ones(n_q, n_k))
            attn_logits.masked_fill_(future_mask == 0, -torch.inf)

        # Apply padding mask
        if padding_mask != None:
            padding_mask = padding_mask.unsqueeze(dim=1)
            attn_logits.masked_fill_(padding_mask == 0, -torch.inf)

        # Apply softmax to masked logits
        softmax_attn = torch.softmax(attn_logits, -1)

        # Apply modelling weighting to the values
        output = torch.bmm(softmax_attn, V)

        return output
class SelfAttentionHead(torch.nn.Module):
    def __init__(self, d_model, mask_future=False):
        """
        :param d_model (int): dimension of the token embeddings
        :param future_masking (bool): whether future tokens should be masked in the modelling
        """
        super().__init__()

        self.d_model = d_model
        self.mask_future = mask_future
        self.attn = Attention(mask_future=mask_future)
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_query, x_key, x_value, padding_mask=None):
        Q = self.q_proj(x_query)
        K = self.k_proj(x_key)
        V = self.v_proj(x_value)
        return self.attn(Q, K, V, padding_mask)
