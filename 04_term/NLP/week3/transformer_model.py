import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # keeps dimension for what mean is applied (it is applied for all besides batch dimension
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)  # as written in the paper


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        # create a vector of shape (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply the sin to even position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model) - done for tensor calculation support

        self.register_buffer('pe', pe)  # this tensor will be saved in the module

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, seq_len)
        if mask is not None:
            attention_scores.masked_fill(mask==0, -1e9)  # going through softmax it will be very little values
        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.W_q(q)  # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.W_k(k)  # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.W_v(v)  # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        
        # splitting whole matrix on different heads and then transpose to move head dimension on second place
        # we did that because we want each head to see whole seq_length, but only a small part of whole embedding
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)  # (Batch, seq_len, d_model) -> (Batch, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)  # (Batch, seq_len, d_model) -> (Batch, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)  # (Batch, seq_len, d_model) -> (Batch, h, seq_len, d_k)

        x, attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # contiguous because pytorch needs it to transforms the shape of tensor, in other case will do this operation in-place
        # When  you call contiguous(), it actually  makes a  copy of the  tensor such that  the order of its elements in memory is the same as if it had been created from scratch with the same data.

        return self.W_o(x)  # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda tmp_x: self.self_attention_block(tmp_x, tmp_x, tmp_x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttention,
                 cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, trg_mask):
        x = self.residual_connections[0](x, lambda tmp_x: self.self_attention_block(tmp_x, tmp_x, tmp_x, trg_mask))
        x = self.residual_connections[1](x, lambda tmp_x: self.self_attention_block(tmp_x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)


class ProjectionLayer(nn.ModuleList):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Batch, seq_len, d_model -> Batch, seq_len, vocab_size
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbeddings, trg_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, trg_pos: PositionalEncoding,
                 proj_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trg, trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)

    def project(self, x):
        return self.proj_layer(x)


def build_transformer(
        src_vocab_size: int, trg_vocab_size: int, src_seq_len: int, trg_seq_len: int,
        d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int=2048) -> Transformer:
    # create embeddings layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    trg_embed = InputEmbeddings(d_model, trg_vocab_size)

    # create the positional encodings layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

    # create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(
            EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        )

    # create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        encoder_decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(
            DecoderBlock(decoder_self_attention_block, encoder_decoder_self_attention, decoder_feed_forward_block, dropout)
        )

    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create projection layer
    projection_layer = ProjectionLayer(d_model, trg_vocab_size)

    # create the transformer
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        trg_embed=trg_embed,
        src_pos=src_pos,
        trg_pos=trg_pos,
        proj_layer=projection_layer
    )

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer