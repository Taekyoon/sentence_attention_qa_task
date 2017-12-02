import torch
import torch.nn as nn
from torch.autograd import Variable


class SentGate(nn.Module):
    def __init__(self, hidden_dim):
        super(SentGate, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, doc_s, doc_len, wns):
        batch_size = doc_s.size(0)
        num_sent = wns.shape[1]

        score = Variable(torch.zeros(batch_size, doc_len, self.hidden_dim * 2)).cuda()

        for i in range(batch_size):
            wn_prev = 0
            for j in range(num_sent):
                wn_next = int(wns[i][j])
                if wn_next == 0:
                    break
                score[i, wn_prev:wn_prev + wn_next, :] = doc_s[i, j, :].repeat(1, wn_next)
                wn_prev = wn_prev + wn_next

        return score
