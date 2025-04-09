import math
import torch
import torch.nn as nn
from torch import Tensor


# Modified from PyTorch Website and ChatGPT proposal
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = dropout
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 2::] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = torch.add(x, self.pe[:x.size(0)])
        if self.dropout is not None:
            x = self.dropout(x)
        return x








if __name__ == '__main__':
    positional_encoding_layer = PositionalEncoding(d_model=4, dropout=0, max_len=14)

    inputs = torch.zeros((3, 14, 4))

    outputs = positional_encoding_layer(inputs)

    print(outputs.detach().numpy())
    inputs = inputs.detach().numpy()
    outputs = outputs.detach().numpy()
    pass
