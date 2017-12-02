from typing import Tuple

import numpy as np

import torch
from torch import cuda, nn
from torch.autograd import Variable

from embedding import Embedding
from cnn_encoder import CNNEncoder
from rnn_base_module import RNNBaseModule
from highway import Highway
from linear import MultiDimLinear

query_senetence_size = 30
paragraph_sentence_size = 400
word_length = 16

char_window_size = 5

feature_dim = 100


class Bidaf(nn.Module):
    def __init__(self,
                 char_dict_size: int = 10,
                 char_embedding_dim: int = 8,
                 word_dict_size: int = 10,
                 word_embedding_dim: int = 300,
                 char_filter_dim: int = 100,
                 n_gram_sizes: Tuple[int, ...] = [5],
                 rnn_dim: int = 100,
                 keep_prob: bool = .8,
                 bidirectional: bool = True):
        super(Bidaf, self).__init__()

        self.char_embedding = Embedding(char_dict_size, char_embedding_dim)
        self.word_embedding = Embedding(word_dict_size, word_embedding_dim)

        self.char_cnn_encoder = CNNEncoder(char_embedding_dim, char_filter_dim, n_gram_sizes)

        self.highway = Highway(char_filter_dim + word_embedding_dim, num_layers=2)

        self.contextual_embedding = RNNBaseModule(char_filter_dim + word_embedding_dim, rnn_dim,
                                                  keep_prob=keep_prob, bidirectional=bidirectional)

        self.model_layers = RNNBaseModule(rnn_dim*2, rnn_dim, num_layers=2, keep_prob=keep_prob,
                                          bidirectional=bidirectional)

        self.output_module = MultiDimLinear(rnn_dim*2, 1)

    def forward(self, word_inputs, char_inputs):
        char_embedded = self.char_embedding(char_inputs)
        char_embedded = self.char_cnn_encoder(char_embedded)

        word_embedded = self.word_embedding(word_inputs)

        word_embedded = self.highway(torch.cat((char_embedded, word_embedded), dim=word_embedded.dim()-1))

        context_hidden_state = self.contextual_embedding.init_hidden(word_embedded.size(0))
        context_embedded, _ = self.contextual_embedding(word_embedded, hidden_state=context_hidden_state)

        model_hidden_state = self.model_layers.init_hidden(context_embedded.size(0))
        pre_output, _ = self.model_layers(context_embedded, hidden_state=model_hidden_state)

        output = self.output_module(pre_output.contiguous())

        return nn.functional.log_softmax(output)

char_context_arr = np.random.random_integers(9, size=(10, 100, 15))
word_context_arr = np.random.random_integers(9, size=(10, 100))

char_inputs = Variable(torch.LongTensor(char_context_arr).cuda())
word_inputs = Variable(torch.LongTensor(word_context_arr).cuda())

model = Bidaf().cuda()

output = model(word_inputs, char_inputs)

print(output.size())


