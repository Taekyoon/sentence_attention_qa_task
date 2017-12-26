import torch
from torch import nn
from modules.utils import relu

from modules.matrix_attention import MatrixAttention
from modules.linear import MultiDimLinear
from modules.utils import last_dim_softmax, weighted_sum, replace_masked_values, masked_softmax


class AttentionFlow(nn.Module):
    def __init__(self,
                 input_dim: int) -> None:
        super(AttentionFlow, self).__init__()

        self.input_dim = input_dim * 3

        self.linear = MultiDimLinear(self.input_dim, 1, bias=True, bias_start=.0)
        self.matrix_attention = MatrixAttention(self.similarity_function)

    def forward(self,
                paragraph: torch.FloatTensor,
                query: torch.FloatTensor,
                dm: torch.FloatTensor = None,
                qm: torch.FloatTensor = None) -> torch.FloatTensor:
        batch_size = paragraph.size(0)
        passage_length = paragraph.size(1)
        encoding_dim = paragraph.size(2)

        passage_question_similarity = self.matrix_attention(paragraph, query).squeeze(-1)

        passage_question_attention = last_dim_softmax(passage_question_similarity, qm)
        passage_question_vectors = weighted_sum(query, passage_question_attention)
        masked_similarity = replace_masked_values(passage_question_similarity,
                                                  qm.unsqueeze(1),
                                                  -1e7)

        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        question_passage_attention = masked_softmax(question_passage_similarity, dm)
        question_passage_vector = weighted_sum(paragraph, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        return passage_question_vectors, tiled_question_passage_vector

    def similarity_function(self,
                            X: torch.FloatTensor,
                            Y: torch.FloatTensor) -> torch.FloatTensor:
        prime_S = torch.cat([X, Y, X * Y], dim=-1)
        S = self.linear(prime_S)

        return S
