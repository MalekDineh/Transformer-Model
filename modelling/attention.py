import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, mask_future=True, device=torch.device('cpu')) -> None:
        """Attention layer.

        Args:
            mask_future (bool, optional): Defaults to True.
        """
        super().__init__()
        self.device = device
        self.mask_future = mask_future

    def forward(self, query, key, value, attention_mask):
        """Forward pass through the attention layer.

        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim).
            key: Key tensor of shape (batch_size, seq_len, embed_dim).
            value: Value tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask: Mask for the attention.

        Returns:
            type: description
        """
        attention = query @ key.transpose(-1, -2)

        attention /= (key.shape[-1] ** 0.5) 

        if self.mask_future:
            forward_mask = torch.tril(torch.ones(query.shape[1], key.shape[1])).to(self.device)
            attention = attention.masked_fill(forward_mask == 0, -torch.inf)

        attention_mask = attention_mask.unsqueeze(1).to(self.device)
        attention = attention.masked_fill(attention_mask == 0, -torch.inf)

        attention = F.softmax(attention, dim=-1)
        
        attention = attention @ value

        return attention
 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask_future=False, device=torch.device('cpu')) -> None:
        """Multi head attention layer.

        Args:
            d_model: Embedding dimension.
            n_heads: Number of attention heads.
            mask_future (bool, optional): Defaults to False.
            dropout (float, optional): Defaults to 0.2.
        """
        super().__init__()
        self.device = device
        self.dk = d_model // n_heads
        
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)

        self.output_transform = nn.Linear(d_model, d_model, bias=False)

        self.self_attention = Attention(mask_future=mask_future, device=device)
    
    def forward(self, x_query, x_key, x_value, attention_mask):
        Q = self.query_transform(x_query)
        K = self.key_transform(x_key)
        V = self.value_transform(x_value)

        Qs = Q.split(self.dk, dim=-1) 
        Ks = K.split(self.dk, dim=-1) 
        Vs = V.split(self.dk, dim=-1) 

        x = []
        for q, k, v in zip(Qs, Ks, Vs):
            x.append(self.self_attention(q, k, v, attention_mask.to(self.device)))

        x_concat = torch.cat(x, dim=-1)
        x = self.output_transform(x_concat)
        return x