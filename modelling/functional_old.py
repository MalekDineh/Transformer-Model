import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedforward(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, input_dim)
        )

    def forward(self, x):
        return self.feedforward(x)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Assuming embed_dim is divisible by num_heads for simplicity
        self.head_dim = embed_dim // num_heads

        self.query_transform = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_transform = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_transform = nn.Linear(embed_dim, embed_dim, bias=False)

        self.output_transform = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Linear projections
        q = self.query_transform(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_transform(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_transform(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        x = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Linear projection
        x = self.output_transform(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.1):
        super(BaseTransformerLayer, self).__init__()

        self.self_attention = MultiheadAttention(input_dim, num_heads)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.feature_transformation = PositionwiseFeedforward(input_dim, feature_dim, dropout)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head self-attention
        attn_output = self.self_attention(x, mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.layer_norm_1(x)

        # Position-wise feedforward
        ff_output = self.feature_transformation(x)
        x = x + self.dropout2(ff_output)
        x = self.layer_norm_2(x)

        return x
