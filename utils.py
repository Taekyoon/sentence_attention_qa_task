import os
import json
import numpy as np
import pandas as pd

import torch

from copy import deepcopy

from torch.autograd import Variable


def load_data(data_path, size):
    if size == 'small':
        train_path = os.path.join(data_path, "data_train_small.json")
        dev_path = os.path.join(data_path, "data_dev_small.json")
    elif size == 'test':
        train_path = os.path.join(data_path, "data_train_test.json")
        dev_path = os.path.join(data_path, "data_dev_test.json")
    elif size == 'glove':
        train_path = os.path.join(data_path, "data_train_glove.json")
        dev_path = os.path.join(data_path, "data_dev_glove.json")
    elif size == 'glove_small':
        train_path = os.path.join(data_path, "data_train_glove_small.json")
        dev_path = os.path.join(data_path, "data_dev_glove_small.json")
    elif size == 'glove_test':
        train_path = os.path.join(data_path, "data_train_glove_test.json")
        dev_path = os.path.join(data_path, "data_dev_glove_test.json")
    elif size == 'glove_test_small':
        train_path = os.path.join(data_path, "data_train_glove_test_small.json")
        dev_path = os.path.join(data_path, "data_dev_glove_test_small.json")
    elif size == 'glove_testtest_small':
        train_path = os.path.join(data_path, "data_train_glove_testtest_small.json")
        dev_path = os.path.join(data_path, "data_dev_glove_testtest_small.json")
    elif size == 'char':
        train_path = os.path.join(data_path, "data_train_char.json")
        dev_path = os.path.join(data_path, "data_dev_char.json")
    elif size == 'char_test':
        train_path = os.path.join(data_path, "data_train_char_test.json")
        dev_path = os.path.join(data_path, "data_dev_char_test.json")
    elif size == 'char_small':
        train_path = os.path.join(data_path, "data_train_char_small.json")
        dev_path = os.path.join(data_path, "data_dev_char_small.json")
    else:
        train_path = os.path.join(data_path, "data_train.json")
        dev_path = os.path.join(data_path, "data_dev.json")

    with open(train_path) as f:
        train = json.load(f)

    with open(dev_path) as f:
        dev = json.load(f)

    return train, dev

def tensor2np(tensor):
    return tensor.data.cpu().numpy()

def load_pretrained_embedding(dictionary, embed_file):
    if embed_file is None: return None, EMBED_DIM

    with open(embed_file, 'r', encoding='utf-8') as f:
        vocab_embed = {}

        for line in f:
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vocab_embed[word] = list(map(float, line[1:]))
        embed_dim = len(line[1:])
        f.close()

    vocab_size = len(dictionary)
    W = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for w, i in dictionary.items():
        if w in vocab_embed:
            W[i, :] = vocab_embed[w]
            n += 1
    print("%d/%d vocabs are initialized with word2vec embeddings." % (n, vocab_size))
    return W, embed_dim

def accuracy(pred1, target1, pred2, target2):
    batch_size = len(target1)
    correct = 0
    for i in range(batch_size):
        if pred1[i][0] == target1[i] and pred2[i][0] == target2[i]:
            correct += 1

    acc = correct / batch_size
    return correct, acc

def index_accuracy(pred, target):
    batch_size = len(target)
    correct = 0

    for i in range(batch_size):
        if pred[i][0] == target[i]:
            correct += 1

    return correct / batch_size

def accuracy_dev(pred1, target1, pred2, target2, ids):
    batch_size = len(target1)
    correct = 0

    for i in range(batch_size):
        ans_pairs = (list(zip(target1[i], target2[i])))
        pred_pair = (pred1[i][0], pred2[i][0])
        if pred_pair in ans_pairs:
            correct += 1

    acc = correct / batch_size
    return correct, acc

def index_accuracy_dev(pred, target):
    batch_size = len(target)
    correct = 0

    for i in range(batch_size):
        if pred[i][0] in target[i]:
            correct += 1

    return correct / batch_size



def distance_dist(pred1, target1, pred2, target2, sis_hist, eis_hist):
    batch_size = len(target1)

    for i in range(batch_size):
        ans_pairs = (list(zip(target1[i], target2[i])))
        pred_pair = (pred1[i][0], pred2[i][0])
        diff = []
        idx = -1
        for jj in range(len(ans_pairs)):
            diff_s = np.abs(ans_pairs[jj][0] - pred_pair[0])
            diff_e = np.abs(ans_pairs[jj][1] - pred_pair[1])
            diff.append(diff_s + diff_e)
            idx = np.argmin(diff)
        diff_s = pred_pair[0] - ans_pairs[idx][0]
        diff_e = pred_pair[1] - ans_pairs[idx][1]

        if diff_s in sis_hist.keys():
            sis_hist[diff_s] += 1
        else:
            sis_hist[diff_s] = 1

        if diff_e in eis_hist.keys():
            eis_hist[diff_e] += 1
        else:
            eis_hist[diff_e] = 1

    return sis_hist, eis_hist


def exp_moving_avg(model, avg_params, decay_rate=1.0):
    for p, avg_p in zip(model.parameters(), avg_params):
        avg_p.mul_(decay_rate).add_(1.0 - decay_rate, p.data)
    return avg_params


def flatten_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_params(model, flattened):
    for p, flattened_p in zip(model.parameters(), flattened):
        p.data.copy_(flattened_p)
    return model


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Borrowed from ImageNet training in PyTorch project
    https://github.com/pytorch/examples/tree/master/imagenet
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        #print(val, n, self.sum, self.count, self.avg)


def get_best_span(span_start_logits, span_end_logits):
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    max_span_log_prob = [-1e20] * batch_size
    span_start_argmax = [0] * batch_size
    best_word_span = Variable(span_start_logits.data.new()
                              .resize_(batch_size, 2).fill_(0)).long()

    span_start_logits = span_start_logits.data.cpu().numpy()
    span_end_logits = span_end_logits.data.cpu().numpy()

    for b in range(batch_size):  # pylint: disable=invalid-name
        for j in range(passage_length):
            val1 = span_start_logits[b, span_start_argmax[b]]
            if val1 < span_start_logits[b, j]:
                span_start_argmax[b] = j
                val1 = span_start_logits[b, j]

            val2 = span_end_logits[b, j]

            if val1 + val2 > max_span_log_prob[b]:
                best_word_span[b, 0] = span_start_argmax[b]
                best_word_span[b, 1] = j
                max_span_log_prob[b] = val1 + val2
    return best_word_span


def get_best_predict_string(batch_size, best_span, original_passage, token_offsets):
    best_span_strings = []
    for i in range(batch_size):
        passage_str = original_passage[i]
        offsets = token_offsets[i][0]
        predicted_span = tuple(best_span[i].data.cpu().numpy())
        start_offset = offsets[predicted_span[0]][0]
        end_offset = offsets[predicted_span[1]][1]
        best_span_string = passage_str[start_offset:end_offset]
        best_span_strings.append(best_span_string)

    return best_span_strings


def get_text_field_mask(text_tensor):
    """
    Takes the dictionary of tensors produced by a ``TextField`` and returns a mask of shape
    ``(batch_size, num_tokens)``.  This mask will be 0 where the tokens are padding, and 1
    otherwise.

    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we assume that the tensor in
    the dictionary with the lowest number of dimensions has plain token ids.  This allows us to
    also handle cases where the input is actually a ``ListField[TextField]``.

    NOTE: Our functions for generating masks create torch.LongTensors, because using
    torch.byteTensors inside Variables makes it easy to run into overflow errors
    when doing mask manipulation, such as summing to get the lengths of sequences - see below.
    #>>> mask = torch.ones([260]).byte()
    #>>> mask.sum() # equals 260.
    #>>> var_mask = torch.autograd.Variable(mask)
    #>>> var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
    """
    tensor_dims = [(tensor.dim(), tensor) for tensor in text_tensor]
    tensor_dims.sort(key=lambda x: x[0])
    token_tensor = tensor_dims[0][1]

    return (token_tensor != 0).long()


def create_pandas_dataframe():
    return pd.DataFrame()


def save_dataframe_as_cvs(dataframe, file_path):
    dataframe.to_csv(file_path, sep='\t', encoding='utf-8')


def compare_raw_string_and_processed_string_in_false_span_acc(vocab, passage, best_spans, ground_truths, em_match,
                                                              best_predict_strings, dataframe_to_save):
    best_spans, ground_truths = unwrap_to_tensors(best_spans, ground_truths)

    batch_size = best_spans.size(0)
    best_spans = best_spans.view(batch_size, -1)
    ground_truths = ground_truths.view(batch_size, -1)
    correct = best_spans.eq(ground_truths).prod(dim=1).float()

    for i, item in enumerate(correct):
        if item == 0 and em_match[i]:
            try:
                answer_token_numbers = passage[i][best_spans[i][0]:best_spans[i][1]+1].data.tolist()
                raw_string = " ".join([vocab.get_token_from_index(num) for num in answer_token_numbers])
                record = {"span_acc_string": raw_string,
                          "best_predict_string": best_predict_strings[i]}
                dataframe_to_save = dataframe_to_save.append(record, ignore_index=True)
            except:
                pass

    return dataframe_to_save


def unwrap_to_tensors(*tensors):
    """
    If you actually passed in Variables to a Metric instead of Tensors, there will be
    a huge memory leak, because it will prevent garbage collection for the computation
    graph. This method ensures that you're using tensors directly and that they are on
    the CPU.
    """
    return (x.data.cpu() if isinstance(x, torch.autograd.Variable) else x for x in tensors)