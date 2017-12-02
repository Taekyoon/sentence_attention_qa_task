import torch
from torch import nn
from torch.nn import ModuleList, Linear
from torch.nn import functional as F


class GatedNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 activation = F.tanh) -> None:
        super(GatedNetwork, self).__init__()
        self.input_dim = int(input_dim / 2)
        self.layers = ModuleList([Linear(input_dim, input_dim)
                                    for _ in range(num_layers)])
        self.activation = activation

    def forward(self,
                contextual_vector: torch.FloatTensor,
                summerized_vector: torch.FloatTensor) -> torch.FloatTensor:
        '''
        inputs dim: (batch_size X some other dims X vector_dims)
        '''
        original_summerized_inputs = summerized_vector
        original_contextual_inputs = contextual_vector.contiguous()
        if original_summerized_inputs.dim() > 2:
            summerized_inputs = original_summerized_inputs.view(-1, summerized_vector.size(-1))
            contextual_inputs = original_contextual_inputs.view(-1, contextual_vector.size(-1))


        current_input = summerized_inputs
        for layer in self.layers:
            projected_input = layer(current_input)
            nonlinear_part = projected_input[:, 0 * self.input_dim: 1 * self.input_dim]
            gate = projected_input[:, 1 * self.input_dim: 2 * self.input_dim]
            nonlinear_part = self.activation(nonlinear_part)
            gate = torch.nn.functional.sigmoid(gate)
            current_input = gate * contextual_inputs + (1 - gate) * nonlinear_part

        if original_summerized_inputs.dim() > 2:
            view_args = list(original_summerized_inputs.size()[:-1]) + [current_input.size(-1)]
            current_input = current_input.view(*view_args)

        return current_input