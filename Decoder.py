import torch
import torch.nn as nn
import numpy as np
import random

# from generate_dataset import SEQUENCE


class BottomLevelDecoderRNN(nn.Module):
    def __init__(self, conductor_hidden_dim, hidden_dim, output_dim):
        super(BottomLevelDecoderRNN, self).__init__()

        self.fc_init = nn.Linear(conductor_hidden_dim, hidden_dim * 4)

        self.rnn_1 = nn.LSTMCell(conductor_hidden_dim + output_dim, hidden_dim)
        self.rnn_2 = nn.LSTMCell(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim

    def forward(
        self,
        c,
        length=None,
        training=True,
        target=None,
        batch_size=None,
        teacher_forcing=True,
        mappings_length=None,
        epoch=None,
        k=None,
    ):

        c = [embed.to(self.device) for embed in c]

        outputs = []
        previous = torch.zeros((batch_size, mappings_length), device=self.device)
        count = 0

        for embed in c:
            t = torch.tanh(self.fc_init(embed))

            h1 = t[:, 0 : self.hidden_dim]
            h2 = t[:, self.hidden_dim : 2 * self.hidden_dim]
            c1 = t[:, 2 * self.hidden_dim : 3 * self.hidden_dim]
            c2 = t[:, 3 * self.hidden_dim : 4 * self.hidden_dim]

            if teacher_forcing:
                k = k
                ratio = self.inverse_sigmoid_schedule(epoch, k)
            else:
                ratio = None

            for _ in range(length // 2):
                if training:
                    if count > 0 and random.random() < ratio:
                        previous = target[:, count - 1, :]
                    else:
                        previous = previous.detach()
                else:
                    previous = previous.detach()

                input = torch.cat([embed, previous], dim=1)

                # if LSTM
                h1, c1 = self.rnn_1(input, (h1, c1))
                h2, c2 = self.rnn_2(h1, (h2, c2))

                previous = self.fc_out(h2)
                outputs.append(previous)

                previous = outputs[-1]

                count += 1

        outputs = torch.stack(outputs, dim=1)

        return outputs, ratio

    def inverse_sigmoid_schedule(self, step, rate):
        return rate / (rate + np.exp(step / rate))


class AutoEncoder_DecoderRNN(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(AutoEncoder_DecoderRNN, self).__init__()

        # if LSTM
        self.fc_init = nn.Linear(latent_dim, hidden_dim * 4)

        # 將z轉換維度
        self.fc_concat = nn.Linear(latent_dim + output_dim, output_dim)

        # if LSTM
        self.rnn_1 = nn.LSTMCell(output_dim, hidden_dim)
        self.rnn_2 = nn.LSTMCell(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim

    def forward(
        self,
        c,
        length=None,
        training=True,
        target=None,
        batch_size=None,
        teacher_forcing=True,
        mappings_length=None,
        epoch=None,
        k=None,
    ):

        c.to(self.device)

        outputs = []
        previous = torch.zeros((batch_size, mappings_length), device=self.device)

        count = 0

        t = torch.tanh(self.fc_init(c))

        # if LSTM
        h1 = t[:, 0 : self.hidden_dim]
        h2 = t[:, self.hidden_dim : 2 * self.hidden_dim]
        c1 = t[:, 2 * self.hidden_dim : 3 * self.hidden_dim]
        c2 = t[:, 3 * self.hidden_dim : 4 * self.hidden_dim]

        for _ in range(length):
            if teacher_forcing:
                k = k
                ratio = self.inverse_sigmoid_schedule(epoch, k)
            if training:
                if count > 0 and random.random() < ratio:
                    previous = target[:, count - 1, :]
                else:
                    previous = previous.detach()
            else:
                previous = previous.detach()

            if count == 0:
                input = torch.cat([c, previous], dim=1)
                input = self.fc_concat(input)

            else:
                input = previous

            # if LSTM
            h1, c1 = self.rnn_1(input, (h1, c1))
            h2, c2 = self.rnn_2(h1, (h2, c2))

            previous = torch.tanh(self.fc_out(h2))
            outputs.append(previous)

            previous = outputs[-1]

            count += 1

        outputs = torch.stack(outputs, dim=1)

        return outputs, ratio

    def inverse_sigmoid_schedule(self, step, rate):
        return rate / (rate + np.exp(step / rate))
