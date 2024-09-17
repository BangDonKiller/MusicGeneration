import torch
import torch.nn as nn


class BidirectionalEncoder(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, latent_dim):
        super(BidirectionalEncoder, self).__init__()

        # 雙層雙向LSTM
        self.lstm = nn.LSTM(
            input_dim,
            lstm_hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm_hidden_dim = lstm_hidden_dim

        # 全連接層，生成 μ 和 σ
        self.fc_mu = nn.Linear(2 * lstm_hidden_dim, latent_dim)  # 乘2是因為雙向LSTM
        self.fc_sigma = nn.Linear(2 * lstm_hidden_dim, latent_dim)

    def forward(self, x, batch_size):
        # x的形狀是 (batch_size, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)

        h_last = h_n.view(2, 2, batch_size, self.lstm_hidden_dim)
        h_last_forward = h_last[1, 0, :, :]
        h_last_backward = h_last[1, 1, :, :]
        h_last = torch.cat((h_last_forward, h_last_backward), dim=1)

        # 通過全連接層生成潛在向量的 μ 和 σ
        mu = self.fc_mu(h_last)
        sigma = self.fc_sigma(h_last)
        sigma = torch.log1p(torch.exp(sigma))

        return mu, sigma
