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
        note_size,
    ):
        super(MusicVAE, self).__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(i, input_dim) for i in note_size])

        self.encoder = BidirectionalEncoder(input_dim, lstm_hidden_dim, latent_dim)
        self.conductor = ConductorRNN(
            latent_dim, conductor_hidden_dim, conductor_output_dim
        )
        self.decoder = BottomLevelDecoderRNN(
            conductor_hidden_dim, decoder_hidden_dim, note_size
        )

        self.latent_dim = latent_dim

    def forward(self, x, truth, teacher_forcing=True):

        mu, logvar = self.encoder(x, self.embeddings)
        z = mu + logvar * torch.randn_like(mu)
        c = self.conductor(z)
        reconstruct = self.decoder(
            c, self.embeddings, length=32, teacher_forcing=teacher_forcing, target=truth
        )

        song = [o.max(-1)[-1] for o in reconstruct]
        song = torch.stack(song).detach()
        return reconstruct, song, mu, logvar

    def reconstruction_loss(self, predict, target):
        loss = 0
        for p, t in zip(predict, target):
            for i in range(p.size(1)):
                loss += F.cross_entropy(p[:, i], t[:, i])
        return loss

    def kl_divergence_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))

    def sample(self, length):
        z = torch.zeros(1, 1, self.latent_dim).normal_(0, 1)
        if next(self.parameters()).is_cuda:
            z = z.cuda()
        c = self.conductor(z)
        output = self.decoder(c, self.embeddings, length=length, teacher_forcing=False)
        output = [o.max(-1)[-1] for o in output]
        output = torch.stack(output).squeeze().transpose(0, 1)
        return output.detach().cpu().numpy()
