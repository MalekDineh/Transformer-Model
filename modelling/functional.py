from torch import nn
import torch
from modelling.attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, input_dim, feature_dim) -> None:
        """Position wise feed forward layer.

        Args:
            input_dim: Embedding dimension of the input. 
            feature_dim: Hidden dimension of the position wise feed forward layer.
        """
        super().__init__()

        self.linear1 = nn.Linear(input_dim, feature_dim)
        self.linear2 = nn.Linear(feature_dim, input_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the feed forward layer.

        Args:
            x: Tensor of shape (batch_size, context_length, embedding_size).
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.2, device=torch.device('cpu')) -> None:
        """Base transformer layer. Contains self attention and feed forward layer. Used in encoder.

        Args:
            input_dim: Embedding dimension of the input. 
            num_heads: Number of heads in the multi head attention layer.
            feature_dim: Hidden dimension of the position wise feed forward layer.
            dropout (float, optional): Defaults to 0.2.
        """
        super().__init__()
        self.device = device
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=False, device=device) 
        self.feature_transformation = FeedForward(input_dim, feature_dim)
        
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x, attention_mask):
        """Forward pass through the encoder layer.

        Args:
            x: Tensor of shape (batch_size, context_length, embedding_size).
            attention_mask: Mask for the attention.
        """
        # self attention
        y = self.self_attention(x, x, x, attention_mask.to(self.device))
        y *= attention_mask.unsqueeze(-1).float() 

        if self.dropout:
            y = self.dropout(y)

        x = self.layer_norm_1(x + y)

        y = self.feature_transformation(x)
        y *= attention_mask.unsqueeze(-1).float() 

        if self.dropout:
            y = self.dropout(y)

        x = self.layer_norm_2(x + y)

        x *= attention_mask.unsqueeze(-1).float() 

        return x
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.2, device=torch.device('cpu')):
        """Transformer decoder layer. Contains self attention, cross attention and feed forward layer. Used in decoder.

        Args:
            input_dim: Embedding dimension of the input.
            num_heads: Number of heads in the multi head attention layer.
            feature_dim: Hidden dimension of the position wise feed forward layer.
            dropout (float, optional): Defaults to 0.2.
        """
        super().__init__()
        self.device = device
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=True, device=device) 
        self.encoder_attention = MultiHeadAttention(input_dim, num_heads, mask_future=False, device=device)
        self.feature_transformation = FeedForward(input_dim, feature_dim)

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, encoder, encoder_attention_mask, attention_mask):
        """Forward pass through the decoder layer.

        Args:
            x: Tensor of shape (batch_size, context_length, embedding_size).
            encoder: Value tensor from the encoder. Tensor of shape (batch_size, context_length, embedding_size).
            encoder_attention_mask: Mask for the encoder attention.
            attention_mask: Mask for the decoder attention.
        """

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        y = self.self_attention(x, x, x, attention_mask)
        y *= attention_mask.unsqueeze(-1).float() 

        if self.dropout:
            y = self.dropout(y)

        x = self.layer_norm_1(x + y)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(self.device)
        y = self.encoder_attention(x, encoder, encoder, encoder_attention_mask)
        y *= attention_mask.unsqueeze(-1).float() 

        if self.dropout:
            y = self.dropout(y)

        x = self.layer_norm_2(x + y)

        y = self.feature_transformation(x)
        y *= attention_mask.unsqueeze(-1).float() 

        if self.dropout:
            y = self.dropout(y)

        x = self.layer_norm_3(x + y)
        x *= attention_mask.unsqueeze(-1).float() 

        return x