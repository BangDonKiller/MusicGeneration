import torch
import torch.nn as nn

from Encoder import BidirectionalEncoder, AutoEncoder
from Conductor import ConductorRNN
from Decoder import BottomLevelDecoderRNN, AutoEncoder_DecoderRNN
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
        mappings_length,
        autoencoder,
    ):
        super(MusicVAE, self).__init__()

        self.batch_size = batch_size
        self.mappings_length = mappings_length
        self.autoencoder = autoencoder

        if self.autoencoder:
            self.encoder = AutoEncoder(input_dim, lstm_hidden_dim, latent_dim)
            self.decoder = AutoEncoder_DecoderRNN(
                latent_dim, decoder_hidden_dim, output_dim
            )
        else:
            self.encoder = BidirectionalEncoder(input_dim, lstm_hidden_dim, latent_dim)
            self.conductor = ConductorRNN(
                latent_dim, conductor_hidden_dim, conductor_output_dim
            )
            self.decoder = BottomLevelDecoderRNN(
                conductor_hidden_dim, decoder_hidden_dim, output_dim
            )

    def forward(self, x, y, teacher_forcing=True, train=True, epoch=None):

        if self.autoencoder:
            z = self.encoder(x, self.batch_size)
            output = self.decoder(
                z,
                length=SEQUENCE,
                target=y,
                batch_size=self.batch_size,
                teacher_forcing=teacher_forcing,
                training=train,
                mappings_length=self.mappings_length,
                epoch=epoch,
            )

            return output
        else:
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
                mappings_length=self.mappings_length,
            )
            return output, mu, sigma
