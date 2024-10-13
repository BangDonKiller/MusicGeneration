import torch
import torch.nn as nn
import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"


class BidirectionalEncoder(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, latent_dim):
        super(BidirectionalEncoder, self).__init__()

        self.lstms = nn.ModuleList(
            [
                nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True),
                nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True),
                nn.LSTM(
                    input_dim, lstm_hidden_dim, batch_first=True, bidirectional=True
                ),
            ]
        )
        self.context_lstm = nn.LSTM(
            4 * lstm_hidden_dim, lstm_hidden_dim, batch_first=True
        )
        self.linear = nn.Linear(5 * lstm_hidden_dim, lstm_hidden_dim)

        self.mu = nn.Linear(lstm_hidden_dim, latent_dim)
        self.logvar = nn.Linear(lstm_hidden_dim, latent_dim)

    def forward(self, x, embeddings):
        # x的形狀是 (batch_size, seq_len, input_dim)

        x = [e(xi) for e, xi in zip(embeddings, x)]
        x = [lstm(xi, None) for lstm, xi in zip(self.lstms, x)]

        context_lstm_input = torch.cat([lstm_out[0] for lstm_out in x], dim=2)
        _, context_lstm_hidden = self.context_lstm(context_lstm_input, None)

        cat_hidden = []

        for i, lstm_out in enumerate(x):
            if i == 2:
                first_hidden = lstm_out[1][0][:1, :]
                second_hidden = lstm_out[1][0][1:, :]
                temp_cat_hidden = torch.cat((first_hidden, second_hidden), dim=2)
                temp_cat_hidden = torch.cat(
                    (temp_cat_hidden, context_lstm_hidden[0]), dim=2
                )
            else:
                temp_cat_hidden = lstm_out[1][0]

            # 將 temp_cat_hidden 添加到 cat_hidden 列表中
            cat_hidden.append(temp_cat_hidden)

        # 將所有拼接的隱藏狀態合併為一個張量
        cat_hidden = torch.cat(cat_hidden, dim=2)  # 或根據需要的維度進行拼接

        linear_out = self.linear(cat_hidden)

        mu = self.mu(linear_out)
        logvar = self.logvar(linear_out)

        return mu, logvar
