import torch
import torch.nn as nn


class ConductorRNN(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=2):
        super(ConductorRNN, self).__init__()

        # 全連接層用於將潛在向量映射為初始狀態
        self.fc_init = nn.Linear(latent_dim, hidden_dim * 4)

        # 指揮RNN
        self.rnn = nn.LSTM(
            latent_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def forward(self, z, batch_size):
        # z: 潛在向量, 形狀 [batch_size, latent_dim]
        # U: 子序列的數量

        z = torch.tanh(self.fc_init(z))  # (batch_size, hidden_dim)

        h1 = z[None, :, 0 : self.hidden_dim]
        h2 = z[None, :, self.hidden_dim : 2 * self.hidden_dim]
        c1 = z[None, :, 2 * self.hidden_dim : 3 * self.hidden_dim]
        c2 = z[None, :, 3 * self.hidden_dim : 4 * self.hidden_dim]

        h = torch.cat((h1, h2), dim=0)
        c = torch.cat((c1, c2), dim=0)

        # get embedding from conductor
        conductor_input = torch.zeros(
            size=(batch_size, 2, self.latent_dim), device=z.device
        )

        # conductor_input = torch.randn(
        #     size=(batch_size, 2, self.latent_dim), device=z.device, requires_grad=True
        # )

        embeddings, _ = self.rnn(conductor_input, (h, c))
        embeddings = torch.unbind(embeddings, dim=1)

        return embeddings
