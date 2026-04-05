import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward
from residual_connections import ResidualConnection
from layer_norm import LayerNorm

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, ff_block : FeedForward, dropout):
        super().__init__()
        self.self_attention_block  = self_attention_block
        self.cross_attention_block  = cross_attention_block
        self.ff_block = ff_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) # the x is from the decoder and other two are from the encoder
        x = self.residual_connections[2](x, self.ff_block)

        return x
    
# src_mask    →  "don't look at padding"            (encoder)
# target_mask →  "don't look at padding or future"  (decoder)

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):

        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    
# now one last layer that is the one linear layer that needs to map all the vocab (the one outside decoder layer)
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # from (batch, seq_len, d_model) to ----> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)