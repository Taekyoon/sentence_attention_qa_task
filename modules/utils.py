import torch
from torch.nn import functional as F
from torch.autograd import Variable


VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def relu(inputs: torch.Tensor) -> torch.Tensor:
    return F.relu(inputs)


def softmax(inputs: torch.FloatTensor) -> torch.FloatTensor:
    return F.softmax(inputs)


def log_softmax(inputs: torch.FloatTensor) -> torch.FloatTensor:
    return F.log_softmax(inputs)


def multi_dim_softmax(inputs: torch.FloatTensor) -> torch.FloatTensor:
    batch_size = inputs.size()[0]
    outputs = torch.stack([F.softmax(inputs[i]) for i in range(batch_size)], 0)

    return outputs.squeeze(-1)


def last_dim_softmax(tensor, mask=None) -> torch.Tensor:
    """
    Takes a tensor with 3 or more dimensions and does a masked softmax over the last dimension.  We
    assume the tensor has shape ``(batch_size, ..., sequence_length)`` and that the mask (if given)
    has shape ``(batch_size, sequence_length)``.  We first unsqueeze and expand the mask so that it
    has the same shape as the tensor, then flatten them both to be 2D, pass them through
    :func:`masked_softmax`, then put the tensor back in its original shape.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor.size()[-1])
    if mask is not None:
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(tensor).contiguous().float()
        mask = mask.view(-1, mask.size()[-1])
    reshaped_result = masked_softmax(reshaped_tensor, mask)
    return reshaped_result.view(*tensor_shape)

def _get_normalized_masked_log_probablities(vector, mask):
    # We calculate normalized log probabilities in a numerically stable fashion, as done
    # in https://github.com/rkadlec/asreader/blob/master/asreader/custombricks/softmax_mask_bricks.py
    input_masked = mask * vector
    shifted = mask * (input_masked - input_masked.max(dim=1, keepdim=True)[0])
    # We add epsilon to avoid numerical instability when the sum in the log yields 0.
    normalization_constant = ((mask * shifted.exp()).sum(dim=1, keepdim=True) + 1e-7).log()
    normalized_log_probabilities = (shifted - normalization_constant)
    return normalized_log_probabilities


def masked_softmax(vector, mask):
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorical cross-entropy loss.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector)
    else:
        # To limit numerical errors from large vector elements outside mask, we zero these out
        mask = mask.type(torch.cuda.FloatTensor)
        result = torch.nn.functional.softmax(vector * mask)
        result = result * mask
        result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
    return result


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.

    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.

    For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
    embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:

        - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
        - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
          query for each document)

    are valid input "vectors", producing tensors of shape:
    ``(batch_size, num_queries, embedding_dim)`` and
    ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def replace_masked_values(tensor, mask, replace_with):
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.
    """
    # We'll build a tensor of the same shape as `tensor`, zero out masked values, then add back in
    # the `replace_with` value.
    mask = mask.type(torch.cuda.FloatTensor)
    one_minus_mask = 1.0 - mask
    values_to_add = replace_with * one_minus_mask
    return tensor * mask + values_to_add


def get_lengths_from_binary_sequence_mask(mask: torch.Tensor):
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.

    Parameters
    ----------
    mask : torch.Tensor, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.

    Returns
    -------
    A torch.LongTensor of shape (batch_size,) representing the lengths
    of the sequences in the batch.
    """
    return mask.long().sum(-1)


def summed_sentence_vector(matrix, sss, ses):
    batch_size = matrix.size(0)
    feature_dim = matrix.size(-1)
    sent_num = sss.shape[1]

    output = Variable(torch.zeros(batch_size, sent_num, feature_dim)).cuda()
    for i in range(batch_size):
        for j in range(sent_num):
            start = sss[i][j]
            end = ses[i][j] + 1
            if j != 0 and sss[i][j] == 0:
                break

            tmp = matrix[i, start:end, :]
            output[i, j, :] = torch.sum(tmp, 0)

    return output

def masked_log_softmax(vector, mask):
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.

    We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``.  You should be masking the result of whatever computation comes out of this in that
    case, anyway, so it shouldn't matter.
    """
    if mask is not None:
        mask = mask.float()
        vector = vector + mask.log()
    return torch.nn.functional.log_softmax(vector)

def to_2D(tensor, dim):
    return tensor.contiguous().view(-1, dim)


def to_3D(tensor, batch, dim):
    return tensor.contiguous().view(batch, -1, dim)


def expand(tensor, target):
    return tensor.expand_as(target)


def exp_mask(val, mask):
    return val + (1 - mask.type(torch.cuda.FloatTensor)) * VERY_NEGATIVE_NUMBER