import torch
import torch.nn as nn

from tts_modules.FastSpeech.Blocks.LengthRegulator import LengthRegulator
from tts_modules.FastSpeech.Blocks.Transformer.FeedForwardTransformer import FeedForwardTransformer
from tts_modules.FastSpeech.Blocks.Transformer.GraphemeEmbedding import GraphemeEmbedding
from tts_modules.FastSpeech.Blocks.Transformer.PositionalEncoding import PositionalEncoding
from tts_modules.FastSpeech.Blocks.Transformer.utils import get_mask


class FastSpeech(nn.Module):
    def __init__(self, embed_dim: int = 384, vocab_size: int = 38, padding_idx: int = 0, p_dropout: float = 0.1,
                 n_transformers: int = 6, n_heads: int = 2, inter_feat: int = 1536, n_mels: int = 80,
                 alpha: float = 1.0, return_attention: bool = False
                 ):
        super().__init__()
        self.return_attention = return_attention
        self.attention_scores_list = []
        self.token_embeddings = GraphemeEmbedding(d_model=embed_dim, vocab_size=vocab_size, padding_idx=padding_idx)
        self.pos_enc_layer = PositionalEncoding(d_model=embed_dim, p_dropout=p_dropout)
        self.first_fft_block = nn.ModuleList(
            [
                FeedForwardTransformer(n_heads=n_heads, d_model=embed_dim, inter_feat=inter_feat, p_dropout=p_dropout,
                                       kernel_size=3, return_attention=self.return_attention
                                       )
                for _ in range(n_transformers)
            ]
        )
        self.length_regulator = LengthRegulator(d_model=embed_dim, p_dropout=p_dropout, alpha=alpha, n_mels=n_mels,
                                                kernel_size=3)
        self.second_fft_block = nn.ModuleList(
            [
                FeedForwardTransformer(n_heads=n_heads, d_model=embed_dim, inter_feat=inter_feat, p_dropout=p_dropout,
                                       kernel_size=3, return_attention=self.return_attention)
                for _ in range(n_transformers)
            ]
        )
        self.final_projection = nn.Linear(in_features=embed_dim, out_features=n_mels)

    @property
    def attention_scores(self):
        return self.attention_scores_list

    def clear_attention_scores(self):
        self.attention_scores_list = []

    def forward(self, x, teacher_durations: torch.Tensor = None, mel_spec_length: int = 80):
        mask1 = get_mask(x).unsqueeze(-2)
        embeddings = self.token_embeddings(x)
        positional_embeddings = self.pos_enc_layer(embeddings)

        for fft_block in self.first_fft_block:
            positional_embeddings = fft_block(positional_embeddings, mask=mask1)
            if self.return_attention:
                self.attention_scores_list.append(fft_block.self_attention.get_attention_matrix)
        first_hidden_output = positional_embeddings

        aligned_hidden, log_duration_prediction = self.length_regulator(first_hidden_output, teacher_durations,
                                                                        mel_spec_length)
        mask2 = get_mask(aligned_hidden)
        mask2 = mask2.all(dim=2).unsqueeze(-2)
        positional_aligned_hidden = self.pos_enc_layer(aligned_hidden)

        for fft_block in self.second_fft_block:
            positional_aligned_hidden = fft_block(positional_aligned_hidden, mask=mask2)
            if self.return_attention:
                self.attention_scores_list.append(fft_block.self_attention.get_attention_matrix)
        second_hidden_output = positional_aligned_hidden

        predicted_mel_spec = self.final_projection(second_hidden_output)
        return predicted_mel_spec, log_duration_prediction
