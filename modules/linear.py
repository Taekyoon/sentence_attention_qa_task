import torch
from torch import nn
from torch.nn.functional import linear


class MultiDimLinear(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout: float = .0,
                 bias: bool = True,
                 trainable: bool = True) -> None:
        super(MultiDimLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=bias)

        if not trainable:
            self.linear.parameters().requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.FloatTensor:
        '''
        inputs dim: (batch_size X ... X input_dim)
        outputs dim: (batch_size X ... X output_dim)
        '''
        original_inputs = inputs
        if original_inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))

        outputs = self.linear(self.dropout(inputs))

        if original_inputs.dim() > 2:
            view_args = list(original_inputs.size()[:-1]) + [outputs.size(-1)]
            outputs = outputs.view(*view_args)

        return outputs


class MultiDimLinear_deprecated(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 weight: torch.FloatTensor = None,
                 bias: bool = True,
                 trainable: bool = True) -> None:
        super(MultiDimLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        if weight is None:
            weight = torch.FloatTensor(self.output_dim, self.input_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            self.weight.data.normal_(0, 1)
        else:
            if weight.size() != (input_dim, output_dim):
                raise AttributeError("A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        if self.bias:
            self.bias_vector = torch.nn.Parameter(torch.Tensor(self.output_dim))

    def forward(self, inputs: torch.Tensor) -> torch.FloatTensor:
        '''
        inputs dim: (batch_size X ... X input_dim)
        outputs dim: (batch_size X ... X output_dim)
        '''
        original_inputs = inputs
        if original_inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))

        if self.bias:
            outputs = linear(inputs, self.weight, self.bias_vector)
        else:
            outputs = linear(inputs, self.weight)

        if original_inputs.dim() > 2:
            view_args = list(original_inputs.size()[:-1]) + [outputs.size(-1)]
            outputs = outputs.view(*view_args)

        return outputs

'''
model = MultiDimLinear(100, 200, bias=False)

i = torch.autograd.Variable(torch.randn(10,20,30,100))

o = model(i)

print(o.size())
'''