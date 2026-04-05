import torch 
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        pe_tensor = torch.zeros((max_seq_len, d_model))

        # we need to compute two diff vectors. pos vector and i vector
        # pos vector : the position of that token in the whole sequence (sentence)
        # i vector : the index positions in the embeddings of i'th token's embedding
        # we are doing this because we will need to calculate the corresponding PE numbers for each specific number in the embeddings
        # this is where the concept of odd and even comes in.

        pos = torch.arange(0, max_seq_len).unsqueeze(1) # (max_seq_len, 1)
        i = torch.arange(0, d_model//2) # (d_model//2,)
        den = torch.exp(i * (2 * -math.log(10000.0) / d_model)) # 10000^2i/d_model
        
        pe_tensor[:, 0::2] = torch.sin(pos / den) # even
        pe_tensor[:, 1::2] = torch.cos(pos / den) # odd

        self.register_buffer('buffer_tensor', pe_tensor)

    def forward(self, x):
      # x coming in has shape (batch, seq_len, d_model) where seq_len might be shorter than max_seq_len.
      # But self.buffer_tensor is always (max_seq_len, d_model)
        x =  x + self.buffer_tensor[:x.size(1), :]
        return self.dropout(x)
