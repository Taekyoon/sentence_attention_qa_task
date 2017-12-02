import torch
from torch import nn
import torch.nn.functional as F

from modules.utils import exp_mask


class AttentionFlow(nn.Module):
    def __init__(self,
                 input_dim):
        super(AttentionFlow, self).__init__()
        self.input_dim = input_dim

        self.input_dim = input_dim * 6
        self.output_dim = input_dim * 8

        self.linear = nn.Linear(self.input_dim, 1)

    def forward(self, context, query, dm=None, qm=None):
        '''
        :context_batch Dims: Batch Size X Sentence Size X Sequence Length X Feature Size
        :query_batch: Batch Size X Sentence Size(1) X Sequence Length X Feature Size
        :similarity_matrix Dims: Batch Size X context Sentence Size X context Sequence Length X Query Sequence Length
        :attended_query Dims: context Sentence Size X context Sequence Length X Feature Size
        :output Dims: Batch Size X Sentence Size X Context Sequence Length X Feature Size * 4
        '''
        batch_size = context.size()[0]
        context_len = context.size()[1]
        query_len = query.size()[1]

        _context = context.unsqueeze(2).expand(context.size()[0],
                                              context.size()[1],
                                              query_len,
                                              context.size()[2])
        _query = query.unsqueeze(1).expand(query.size()[0],
                                          context_len,
                                          query.size()[1],
                                          query.size()[2])

        concated_vectors = torch.cat((_context, _query, _context * _query), -1).view(-1, self.input_dim)
        similarity_matrix = self.linear(concated_vectors).view(-1, context_len, query_len)

        if dm is not None and qm is not None:
            qm = qm[:, :query_len]
            _context_mask = dm.unsqueeze(2).expand(dm.size()[0],
                                                   dm.size()[1],
                                                   query_len)
            _query_mask = qm.unsqueeze(1).expand(qm.size()[0],
                                                 context_len,
                                                 qm.size()[1])
            similarity_mask = _context_mask * _query_mask
            similarity_matrix = exp_mask(similarity_matrix, similarity_mask)

        batch_arr = []

        for bi in range(batch_size):
            attended_query = self.context2query_attention(similarity_matrix[bi], query[bi])
            attended_context = self.query2context_attention(similarity_matrix[bi], context[bi], attended_query)
            batch_arr.append(attended_context)

        output = torch.stack(batch_arr)

        return output

    def context2query_attention(self, similarity, query):
        attention_weight = F.softmax(similarity)
        output = torch.mm(attention_weight, query)
        return output

    def query2context_attention(self, similarity, context, attended_query):
        feature_size = context.size()[-1]
        attention_weight = F.softmax(torch.max(similarity, 1)[0]) \
            .unsqueeze(-1).repeat(1, feature_size)
        attended_context = context * attention_weight
        attended_context = attended_context.sum(dim=-2).unsqueeze(-2).expand(context.size())

        applied_query = context * attended_query
        applied_context = context * attended_context

        output = torch.cat((context, attended_query, applied_query, applied_context), -1)
        return output
