import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from modules.sentgate import SentGate
from modules.utils import to_2D, to_3D, expand


class SentCoattn(nn.Module):
    def __init__(self, hidden_dim, dropout=.0):
        super(SentCoattn, self).__init__()
        self.hidden_dim = hidden_dim

        self.sentgate = SentGate(self.hidden_dim)

        self.Wc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.Wc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.Wb = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)

        self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                print("Done")
                size = m.weight.size()
                fan_out = size[0]  # number of rows
                fan_in = size[1]  # number of columns
                variance = np.sqrt(2.0 / (fan_in + fan_out))
                m.weight.data.normal_(0.0, variance)

    def forward(self, G, doc_s, doc_h, wns):
        doc_len = doc_h.size(1)
        batch_size = G.size(0)

        G_enc = self.Wc1(self.dropout(to_2D(G, self.hidden_dim * 2)))
        G_enc = to_3D(G_enc, batch_size, self.hidden_dim * 2)
        G_enc = self.sentgate(G_enc, doc_len, wns)  # B x D x 2*H

        doc_h_enc = self.Wc2(self.dropout(to_2D(doc_h, self.hidden_dim * 2)))
        doc_h_enc = to_3D(doc_h_enc, batch_size, self.hidden_dim * 2)

        ones = Variable(torch.ones(batch_size, self.hidden_dim * 2)).cuda()  # B x H
        bias = self.Wb(self.dropout(ones))  # B x H
        bias = expand(bias.unsqueeze(1), doc_h_enc)

        output = G_enc + doc_h_enc + bias

        #Zi = F.tanh(output)  # B x D x 2*H
        Fi = F.sigmoid(output)  # B x D x 2*H

        C_tilde = torch.mul(Fi, doc_h) + torch.mul((1 - Fi), G_enc)

        return C_tilde, Fi
