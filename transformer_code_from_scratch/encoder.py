import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward
from residual_connections import ResidualConnection
from layer_norm import LayerNorm

class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttention, ff_block : FeedForward, dropout):
        super().__init__()
        self.self_attention_block  = self_attention_block
        self.ff_block = ff_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, src_mask))
        x = self.ff_block[1](x , self.ff_block)
        return x

# because the encoder block repeats 6 times, we will now make the main encoder block
class Encoder(nn.Module):
    
    def __init__(self, layers = nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        