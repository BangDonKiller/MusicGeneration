import torch
import torch.nn as nn
import torch.nn.functional as F

from Encoder import BidirectionalEncoder
from Conductor import ConductorRNN
from Decoder import BottomLevelDecoderRNN


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
        note_size,
    ):
        super(MusicVAE, self).__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(i, input_dim) for i in note_size])

        self.encoder = BidirectionalEncoder(input_dim, lstm_hidden_dim, latent_dim)
        self.conductor = ConductorRNN(
            latent_dim, conductor_hidden_dim, conductor_output_dim
        )
        self.decoder = BottomLevelDecoderRNN(
            conductor_hidden_dim, decoder_hidden_dim, output_dim
        )

    def forward(self, x, teacher_forcing=True):

        mu, sigma = self.encoder(x, self.embeddings)
        z = mu + sigma * torch.randn_like(mu)
        # c = self.conductor(z, batch_size=self.batch_size)
        # output, ratio = self.decoder()

        song = [o.max(-1)[-1] for o in output]
        song = torch.stack(song).detach()
        return output, song, mu, sigma

    def reconstruction_loss(self, predict, target):
        loss = 0
        for p, t in zip(predict, target):
            for i in range(p.size(1)):
                loss += F.cross_entropy(p[:, i], t[:, i])
        return loss

    def kl_divergence_loss(mu, sigma):
        return -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.pow(2))
