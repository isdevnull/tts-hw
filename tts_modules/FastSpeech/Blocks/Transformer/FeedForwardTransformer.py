from tts_modules.FastSpeech.Blocks.Transformer.MultiHeadedAttentionModule import MultiHeadedAttention
import torch.nn as nn


class ConvolutionalFeedForward(nn.Module):
    def __init__(self, d_model: int, inter_feat: int, *args, **kwargs):
        super().__init__()
        #self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=inter_feat, *args, **kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, groups=d_model, *args, **kwargs),
            nn.Conv1d(in_channels=d_model, out_channels=inter_feat, kernel_size=1)
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=inter_feat, out_channels=inter_feat, groups=inter_feat, *args, **kwargs),
            nn.Conv1d(in_channels=inter_feat, out_channels=d_model, kernel_size=1)
        )
        #self.conv2 = nn.Conv1d(in_channels=inter_feat, out_channels=d_model, *args, **kwargs)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.transpose(1, 2)
        return x


class LayerNormResidualConnection(nn.Module):
    def __init__(self, d_model: int, p_dropout: float = 0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, sublayer):
        # pre-norm instead of post-norm
        return x + self.dropout(sublayer(self.layernorm(x)))


class FeedForwardTransformer(nn.Module):
    def __init__(self, n_heads: int, d_model: int, inter_feat: int, p_dropout: float = 0.1, *args, **kwargs):
        super().__init__()
        self.self_attention = MultiHeadedAttention(
            n_heads=n_heads,
            d_model=d_model,
            d_k=d_model // n_heads,
            d_v=d_model // n_heads
        )
        self.conv_net = ConvolutionalFeedForward(d_model=d_model, inter_feat=inter_feat, padding=1, *args, **kwargs)
        self.res_layer1 = LayerNormResidualConnection(d_model=d_model, p_dropout=p_dropout)
        self.res_layer2 = LayerNormResidualConnection(d_model=d_model, p_dropout=p_dropout)

    def forward(self, x, mask=None):
        output_inter = self.res_layer1(x, lambda y: self.self_attention(y, y, y, mask=mask))
        output_final = self.res_layer2(output_inter, self.conv_net)
        return output_final
