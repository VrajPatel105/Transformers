import math
import torch.nn as nn
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock
from decoder import ProjectionLayer
from attention import MultiHeadAttention
from feed_forward import FeedForward
from positional_encoding import PositionalEncoding


class InputEmbeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embeddings: InputEmbeddings, target_embeddings: InputEmbeddings,  src_pos: PositionalEncoding, target_pos: PositionalEncoding, projection_layer : ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.target_embeddings = target_embeddings
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embeddings(src)
        src = self.src_pos(src)
        return src.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embeddings(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)
    
    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff : int = 2048):

    # creating the embeddings layer
    src_embeddings = InputEmbeddings(d_model, src_vocab_size)
    target_embeddings = InputEmbeddings(d_model, target_vocab_size)

    # creating the positional encoding layers
    src_pe = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pe = PositionalEncoding(d_model, target_seq_len, dropout)

    # creating the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embeddings, target_embeddings, src_pe, target_pe, projection_layer)

    # Initializing the parameters -> good way is to use xavier glorant initialization techniques
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer