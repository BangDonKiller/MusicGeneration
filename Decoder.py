import torch
import torch.nn as nn
import numpy as np

from generate_dataset import SEQUENCE


class BottomLevelDecoderRNN(nn.Module):
    def __init__(self, conductor_hidden_dim, hidden_dim, output_dim):
        super(BottomLevelDecoderRNN, self).__init__()

        # if LSTM
        # self.fc_init = nn.Linear(conductor_hidden_dim, hidden_dim * 4)

        # if GRU
        self.fc_init = nn.Linear(conductor_hidden_dim, hidden_dim * 2)

        # if LSTM
        # self.rnn_1 = nn.LSTMCell(conductor_hidden_dim + output_dim, hidden_dim)
        # self.rnn_2 = nn.LSTMCell(hidden_dim, hidden_dim)

        # if GRU
        self.rnn_1 = nn.GRUCell(conductor_hidden_dim + output_dim, hidden_dim)
        self.rnn_2 = nn.GRUCell(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim

    def forward(
        self,
        c,
        length=SEQUENCE,
        training=True,
        target=None,
        batch_size=None,
        teacher_forcing=True,
    ):

        # embeddings to gpu
        c = [embed.to(self.device) for embed in c]

        outputs = []
        previous = torch.zeros((batch_size, 130), device=self.device)

        count = 0
        step = 0
        rate = 20

        for embed in c:
            t = torch.tanh(self.fc_init(embed))

            # if LSTM
            # h1 = t[:, 0 : self.hidden_dim]
            # h2 = t[:, self.hidden_dim : 2 * self.hidden_dim]
            # c1 = t[:, 2 * self.hidden_dim : 3 * self.hidden_dim]
            # c2 = t[:, 3 * self.hidden_dim : 4 * self.hidden_dim]

            # if GRU
            h1 = t[:, 0 : self.hidden_dim]
            h2 = t[:, self.hidden_dim : 2 * self.hidden_dim]

            for _ in range(length // 2):
                if training:
                    # TF_rate = self.inverse_sigmoid_schedule(step, rate)
                    if count > 0 and teacher_forcing:
                        previous = target[:, count - 1, :]
                    else:
                        previous = previous.detach()
                else:
                    previous = previous.detach()

                input = torch.cat([embed, previous], dim=1)

                # if LSTM
                # h1, c1 = self.rnn_1(input, (h1, c1))
                # h2, c2 = self.rnn_2(h1, (h2, c2))

                # if GRU
                h1 = self.rnn_1(input, h1)
                h2 = self.rnn_2(h1, h2)

                previous = self.fc_out(h2)
                outputs.append(previous)

                previous = outputs[-1]
                count += 1
                step += 1

        outputs = torch.stack(outputs, dim=1)

        return outputs

    def inverse_sigmoid_schedule(self, step, rate):
        return rate / (rate + np.exp(step / rate))
