import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model is not divisible by number of heads (num_heads)"
        self.head_dim = d_model // num_heads # this is d_k based on research paper -> used for the scaled dot product

        self.Wq = nn.Linear(d_model, d_model) # Query matrix           for ex 512 in and 512 out -> d_model, d_model
        self.Wk = nn.Linear(d_model, d_model) # Key matrix
        self.Wv = nn.Linear(d_model, d_model) # Value matrix
        self.Wo = nn.Linear(d_model, d_model) # output matrix -> this is simply head_dim x num_heads = d_model that's why we just took d_model, d_model

        self.dropout = nn.Dropout(dropout)

    @staticmethod # we use it to simply call this function without the class. so we dont need an instance of the class to get this function. 
    def attention(q, k, v, mask, dropout: nn.Dropout):
      head_dim = q.shape[-1]
    
      attention_scores = (q @ k.transpose(-2,-1)) / math.sqrt(head_dim) # @ means matrix multiplication , and # transpose(-2,-1) -> transposing the last two dims. so it goes from seq_len, head_dim to head_dim , seq_len

      # defining the mask
      if mask:
        attention_scores.masked_fill_(mask == 0, -1e9) # -1e9 is simply -infinity
      attention_scores = attention_scores.softmax(dim = -1)

      if dropout is not None:
        attention_scores = dropout(attention_scores)

        return (attention_scores @ v), attention_scores # this other attention_scores is used for visualizing


    def forward(self, q, k, v, mask=None):
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        batch_size = q.shape[0]
        seq_len = q.shape[1]

        # the transpose methods converts the shape : from (Batch, seq_len, num_heads, head_dim) -> (Batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention(q, k, v, mask, self.dropout)

        # now we have the smaller each head matrices and we combine all of them to get the d_model x d_model matrix and then multiply with the output matrix to get the final multiheadattention output
        # (Batch, num_heads, seq_len, head_dim) -> (Batch, seq_len, num_heads, head_dim) -> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.head_dim) 

        # (Batch, seq_len, d_model) as input to -> (Batch, seq_len, d_model) as output
        return self.Wo(x)