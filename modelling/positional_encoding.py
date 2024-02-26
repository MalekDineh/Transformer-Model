import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for the Transformer model
    """
    def __init__(self, d_model, max_len=512, device=torch.device('cpu')):
        """
        Args:
            d_model (int): The dimension of the model
            max_len (int, optional): The maximum length of the input sequence. Defaults to 512.
            device (torch.device, optional): The device to use. Defaults to torch.device('cpu').
            
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.device = device
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach().to(self.device)


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)
