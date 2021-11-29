import torch
import torch.nn as nn

from tts_modules.FastSpeech.Blocks.LengthRegulator import LengthRegulator
from tts_modules.FastSpeech.Blocks.Transformer.FeedForwardTransformer import FeedForwardTransformer
from tts_modules.FastSpeech.Blocks.Transformer.GraphemeEmbedding import GraphemeEmbedding
from tts_modules.FastSpeech.Blocks.Transformer.PositionalEncoding import PositionalEncoding


class FastSpeech(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, padding_idx: int = 0, p_dropout: float = 0.1,
                 n_transformers: int = 6, n_heads: int = 2, inter_feat: int = 1536, n_mels: int = 80,
                 alpha: float = 1.0
                 ):
        super().__init__()
        self.token_embeddings = GraphemeEmbedding(d_model=embed_dim, vocab_size=vocab_size, padding_idx=padding_idx)
        self.pos_enc_layer = PositionalEncoding(d_model=embed_dim, p_dropout=p_dropout)
        self.first_fft_block = nn.ModuleList(
            [
                FeedForwardTransformer(n_heads=n_heads, d_model=embed_dim, inter_feat=inter_feat, p_dropout=p_dropout,
                                       kernel_size=3
                                       )
                for _ in range(n_transformers)
            ]
        )
        self.length_regulator = LengthRegulator(d_model=embed_dim, p_dropout=p_dropout, alpha=alpha, n_mels=n_mels,
                                                kernel_size=3)
        self.second_fft_block = nn.ModuleList(
            [
                FeedForwardTransformer(n_heads=n_heads, d_model=embed_dim, inter_feat=inter_feat, p_dropout=p_dropout,
                                       kernel_size=3)
                for _ in range(n_transformers)
            ]
        )
        self.final_projection = nn.Linear(in_features=embed_dim, out_features=n_mels)

    def forward(self, x, teacher_durations: torch.Tensor = None, mel_spec_length: int = None):
        embeddings = self.token_embeddings(x)
        positional_embeddings = self.pos_enc_layer(embeddings)

        for fft_block in self.first_fft_block:
            positional_embeddings = fft_block(positional_embeddings, mask=None)
        first_hidden_output = positional_embeddings

        aligned_hidden, log_duration_prediction = self.length_regulator(first_hidden_output, teacher_durations,
                                                                        mel_spec_length)
        positional_aligned_hidden = self.pos_enc_layer(aligned_hidden)

        for fft_block in self.second_fft_block:
            positional_aligned_hidden = fft_block(positional_aligned_hidden, mask=None)
        second_hidden_output = positional_aligned_hidden

        predicted_mel_spec = self.final_projection(second_hidden_output)
        return predicted_mel_spec, log_duration_prediction
