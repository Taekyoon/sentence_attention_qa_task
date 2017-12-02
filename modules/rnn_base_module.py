from typing import Optional

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from modules.utils import get_lengths_from_binary_sequence_mask


class RNNBaseModule(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = .0,
                 bidirectional: bool = False,
                 #output_dim: Optional[int] = None,
                 _module: nn.modules.RNNBase = nn.GRU) -> None:
        super(RNNBaseModule, self).__init__()
        self.module = _module
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.module = _module(input_dim, self.hidden_dim, num_layers,
                              batch_first=True, dropout=self.dropout,
                              bidirectional=self.bidirectional)

    def forward(self,
                inputs: torch.FloatTensor,
                mask: torch.Tensor = None,
                hidden_state: torch.Tensor = None) -> torch.FloatTensor:

        if mask is None:
            output, hidden_state = self.module(inputs, hidden_state)
            return output, hidden_state

        batch_size, total_sequence_length = mask.size()
        num_valid = torch.sum(mask[:, 0]).int().data[0]

        if num_valid < batch_size:
            mask = mask.clone()
            mask[:, 0] = 1

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(inputs,
                                                                                           sequence_lengths)

        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths.data.tolist(),
                                                     batch_first=True)

        packed_sequence_output, final_hidden_state = self.module(packed_sequence_input, hidden_state)
        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)

        if num_valid < batch_size:
            unpacked_sequence_tensor[num_valid:, :, :] = 0.

        sequence_length_difference = total_sequence_length - unpacked_sequence_tensor.size(1)
        if sequence_length_difference > 0:
            zeros = unpacked_sequence_tensor.data.new(batch_size, sequence_length_difference,
                                                      unpacked_sequence_tensor.size(-1)).fill_(0)
            zeros = torch.autograd.Variable(zeros)
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 1)

        output = unpacked_sequence_tensor.index_select(0, restoration_indices)
        hidden_state = final_hidden_state.index_select(1, restoration_indices)

        return output, hidden_state

    def init_hidden(self, batch_size):
        if self.bidirectional:
            h0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())

        return h0


def sort_batch_by_length(tensor: torch.autograd.Variable, sequence_lengths: torch.autograd.Variable):
    """
    Sort a batch first tensor by some specified lengths.

    Parameters
    ----------
    tensor : Variable(torch.FloatTensor), required.
        A batch first Pytorch tensor.
    sequence_lengths : Variable(torch.LongTensor), required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.

    Returns
    -------
    sorted_tensor : Variable(torch.FloatTensor)
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : Variable(torch.LongTensor)
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : Variable(torch.LongTensor)
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    """

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    # This is ugly, but required - we are creating a new variable at runtime, so we
    # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
    # refilling one of the inputs to the function.
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    index_range = Variable(index_range.long())
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)

    return sorted_tensor, sorted_sequence_lengths, restoration_indices