import torch
import torch.nn as nn
from layer_norm import LayerNorm

class ResidualConnection(nn.Module):

    def __init__(self, dropout):
        super().__init()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # this is the add & norm layer