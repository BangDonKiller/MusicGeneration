import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# from generate_dataset import SEQUENCE


class BottomLevelDecoderRNN(nn.Module):
    def __init__(self, conductor_hidden_dim, hidden_dim, output_dim):
        super(BottomLevelDecoderRNN, self).__init__()

        self.lstm1 = nn.ModuleList(
            [nn.LSTMCell(conductor_hidden_dim * 2, hidden_dim) for _ in range(3)]
        )
        self.context = nn.LSTMCell(hidden_dim * 3, hidden_dim)
        self.lstm2 = nn.ModuleList(
            [
                nn.LSTMCell(conductor_hidden_dim + hidden_dim, hidden_dim)
                for _ in range(3)
            ]
        )
        self.output = nn.ModuleList([nn.Linear(hidden_dim, o) for o in output_dim])
        self.hidden = nn.Linear(conductor_hidden_dim, conductor_hidden_dim * 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, c, embeddings, length=None, teacher_forcing=True, target=None):

        batch_size = c.size(0)
        pointer = 0

        outputs = []
        note = [torch.zeros(batch_size, dtype=torch.long) for _ in range(3)]

        if next(self.parameters()).is_cuda:
            note = [n.cuda() for n in note]

        notes = [e(x) for e, x in zip(embeddings, note)]

        for counts in range(length):
            if counts % 16 == 0:
                hidden = F.tanh(self.hidden(c[:, pointer, :]))
                lstm1_hidden = [hidden for _ in range(3)]
                lstm1_cell = [torch.zeros_like(lstm1_hidden[i]) for i in range(3)]
                lstm2_hidden = [hidden for _ in range(3)]
                lstm2_cell = [torch.zeros_like(lstm2_hidden[i]) for i in range(3)]
                context = hidden
                context_cell = torch.zeros_like(context)
                if counts != 0:
                    pointer += 1

            lstm1_hidden = [
                l1(torch.cat([x, c[:, pointer, :]], -1), (hx, cx))[0]
                for l1, x, hx, cx in zip(self.lstm1, notes, lstm1_hidden, lstm1_cell)
            ]
            lstm1_hidden_unroll = [h for h in lstm1_hidden]
            lstm1_hidden_unroll_cell = [
                torch.zeros_like(h) for h in lstm1_hidden_unroll
            ]

            output = []

            # note unrolling
            for i in range(3):
                context = self.context(
                    torch.cat(lstm1_hidden_unroll, -1), (context, context_cell)
                )[0]
                lstm2_hidden[i] = self.lstm2[i](
                    torch.cat([context, c[:, pointer, :]], -1),
                    (lstm2_hidden[i], lstm2_cell[i]),
                )[0]
                output.append(self.output[i](lstm1_hidden_unroll[i] + lstm2_hidden[i]))

                if teacher_forcing:
                    note[i] = embeddings[i](target[i, :, counts])
                else:
                    note[i] = embeddings[i](output[i].max(-1)[1].detach())
                lstm1_hidden_unroll[i] = self.lstm1[1](
                    torch.cat([note[i], c[:, pointer, :]], -1),
                    (lstm1_hidden_unroll[i], lstm1_hidden_unroll_cell[i]),
                )[0]

            outputs.append(output)
        outputs = [torch.stack(o).transpose(0, 1) for o in zip(*outputs)]

        return outputs

    def inverse_sigmoid_schedule(self, step, rate):
        return rate / (rate + np.exp(step / rate))
