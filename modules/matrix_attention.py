import torch
from torch import nn


class MatrixAttention(nn.Module):
    def __init__(self,
                 similarity_function = lambda x, y : x * y) -> None:
        super(MatrixAttention, self).__init__()

        self.similarity_function = similarity_function


    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_1.size()[2])
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_2.size()[2])

        return self.similarity_function(tiled_matrix_1, tiled_matrix_2)