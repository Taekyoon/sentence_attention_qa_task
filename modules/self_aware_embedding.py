from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import Linear, Conv1d
from torch.nn import functional as F


class SelfAwareEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_filters: int,
                 ngram_filter_sizes: Tuple[int, ...],
                 padding_sizes: Tuple[int, ...],
                 keep_prob: float = .8,
                 conv_layer_activation = F.relu,
                 output_dim: Optional[int] = None) -> None:
        super(SelfAwareEmbedding, self).__init__()
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.ngram_filter_sizes = ngram_filter_sizes
        self.padding_sizes = padding_sizes
        self.activation = conv_layer_activation
        self.output_dim = output_dim

        self.dropout = nn.Dropout(1 - keep_prob)
        self.convolution_layers = [Conv1d(in_channels=self.input_dim,
                                           out_channels=self.num_filters,
                                           kernel_size=ngram_size,
                                           padding=padding)
                                    for ngram_size, padding in zip(self.ngram_filter_sizes, self.padding_sizes)]

        for i, conv_layer in enumerate(self.convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        maxpool_output_dim = self.num_filters * len(self.ngram_filter_sizes)
        if self.output_dim:
            self.projection_layer = Linear(maxpool_output_dim, self.output_dim)
        else:
            self.projection_layer = None
            self.output_dim = maxpool_output_dim

    def forward(self, inputs: torch.Tensor) -> torch.FloatTensor:
        '''
        inputs dim: (batch_size X num_tokens X embedding_dim)
        outputs dim: (batch_size X output_dim)
        '''
        original_inputs = inputs
        if original_inputs.dim() > 3:
            inputs = inputs.view(-1, inputs.size(-2), inputs.size(-1))

        inputs = torch.transpose(inputs, 1, 2)
        filter_outputs = [self.activation(convolution_layer(self.dropout(inputs)))
                            for convolution_layer in self.convolution_layers]

        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        maxpool_output = torch.transpose(maxpool_output, 1, 2).contiguous()

        if self.projection_layer:
            maxpool_dim = maxpool_output.size()
            result = self.projection_layer(maxpool_output.view(-1, maxpool_dim[-1]))
            view_args = list(maxpool_dim[:-1]) + [self.output_dim]
            result = result.view(*view_args)
        else:
            result = maxpool_output

        if original_inputs.dim() > 3:
            view_args = list(original_inputs.size()[:2]) + [result.size(-1)]
            result = result.view(*view_args)
        return result

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim


from torch.autograd import Variable

input_dim = 100
output_dim = 100

test_inputs = Variable(torch.randn(10, 20, 100))

model = SelfAwareEmbedding(input_dim, output_dim, ngram_filter_sizes=[5, 7], padding_sizes=[2, 3], output_dim=100)

o = model(test_inputs)

print(o.size())
