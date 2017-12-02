import torch
from torch import nn
from torch.nn import ModuleList, Linear
from torch.nn import functional as F


class Highway(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 activation=F.relu) -> None:
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = ModuleList([Linear(input_dim, input_dim * 2)
                                  for _ in range(num_layers)])
        self.activation = activation
        for layer in self.layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, to we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        '''
        inputs dim: (batch_size X some other dims X vector_dims)
        '''
        original_inputs = inputs
        if original_inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))

        current_input = inputs
        for layer in self.layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[:, (0 * self.input_dim): (1 * self.input_dim)]
            gate = projected_input[:, (1 * self.input_dim): (2 * self.input_dim)]
            nonlinear_part = self.activation(nonlinear_part)
            gate = torch.nn.functional.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part

        if original_inputs.dim() > 2:
            view_args = list(original_inputs.size())
            current_input = current_input.view(*view_args)

        return current_input