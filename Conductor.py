import torch
import torch.nn as nn


class ConductorRNN(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=2):
        super(ConductorRNN, self).__init__()

        self.fc_init = nn.Linear(latent_dim, hidden_dim * 4)
        self.rnn = nn.LSTM(
            latent_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def forward(self, z):
        # z: 潛在向量, 形狀 [batch_size, latent_dim]
        batch_size = z.size(1)

        z = torch.tanh(self.fc_init(z))  # (batch_size, hidden_dim)

        h1 = z[:, :, 0 : self.hidden_dim]
        h2 = z[:, :, self.hidden_dim : 2 * self.hidden_dim]
        c1 = z[:, :, 2 * self.hidden_dim : 3 * self.hidden_dim]
        c2 = z[:, :, 3 * self.hidden_dim : 4 * self.hidden_dim]

        h = torch.cat((h1, h2), dim=0)
        c = torch.cat((c1, c2), dim=0)

        # get embedding from conductor
        conductor_input = torch.zeros(
            size=(batch_size, 2, self.latent_dim), device=z.device
        )

        output, (h, c) = self.rnn(conductor_input, (h, c))

        return output
