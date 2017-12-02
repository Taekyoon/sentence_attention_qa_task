import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 weight: torch.FloatTensor = None,
                 trainable: bool = True) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.output_dim = embedding_dim

        self.embedding = nn.Embedding(self.num_embeddings, self.output_dim)

        if not weight is None:
            self.embedding.weight = nn.Parameter(weight)

        if not trainable:
            self.embedding.requires_grad = False

    def forward(self, inputs: torch.LongTensor) -> torch.FloatTensor:
        '''
        inputs dim: (batch_size X sequence_length)
        outputs dim: (batch_size X sequence_length X embedding_dim)
        '''
        original_inputs = inputs
        if original_inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))

        embedded = self.embedding(inputs)

        if original_inputs.dim() > 2:
            view_args = list(original_inputs.size()) + [embedded.size(-1)]
            embedded = embedded.view(*view_args)

        return embedded


class Embedding_deprecated(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 projection_dim: int = None,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.output_dim = projection_dim or embedding_dim

        if weight is None:
            print('no weight')
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            self.weight.data.normal_(0, 1)
        else:
            print('has weight')
            if weight.size() != (num_embeddings, embedding_dim):
                raise AttributeError("A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        self._projection = None
        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)

    def forward(self, inputs: torch.LongTensor) -> torch.FloatTensor:
        '''
        inputs dim: (batch_size X sequence_length)
        outputs dim: (batch_size X sequence_length X embedding_dim)
        '''
        original_inputs = inputs
        if original_inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
        embedded = embedding(inputs, self.weight,
                             max_norm=self.max_norm,
                             norm_type=self.norm_type,
                             scale_grad_by_freq=self.scale_grad_by_freq,
                             sparse=self.sparse)
        if original_inputs.dim() > 2:
            view_args = list(original_inputs.size()) + [embedded.size(-1)]
            embedded = embedded.view(*view_args)

        if self._projection:
            projection = self._projection
            embedded = projection(embedded)

        return embedded
