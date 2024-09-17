import torch
import torch.nn as nn

from Encoder import BidirectionalEncoder
from Conductor import ConductorRNN
from Decoder import BottomLevelDecoderRNN
from generate_dataset import SEQUENCE


class MusicVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        lstm_hidden_dim,
        latent_dim,
        conductor_hidden_dim,
        conductor_output_dim,
        decoder_hidden_dim,
        output_dim,
        batch_size,
    ):
        super(MusicVAE, self).__init__()

        # 編碼器
        self.encoder = BidirectionalEncoder(input_dim, lstm_hidden_dim, latent_dim)

        # 指揮RNN
        self.conductor = ConductorRNN(
            latent_dim, conductor_hidden_dim, conductor_output_dim
        )

        # 最底層解碼器RNN
        self.decoder = BottomLevelDecoderRNN(
            conductor_hidden_dim, decoder_hidden_dim, output_dim
        )

        self.batch_size = batch_size

    def forward(self, x, y, teacher_forcing=True, train=True):
        # 編碼器生成潛在向量的 μ 和 σ
        mu, sigma = self.encoder(x, self.batch_size)

        # 通過重參數化技巧從正態分佈中采樣
        z = mu + sigma * torch.randn_like(mu)

        # 指揮RNN生成嵌入向量
        c = self.conductor(z, batch_size=self.batch_size)

        output = self.decoder(
            c,
            length=SEQUENCE,
            target=y,
            batch_size=self.batch_size,
            teacher_forcing=teacher_forcing,
            training=train,
        )
        return output, mu, sigma
