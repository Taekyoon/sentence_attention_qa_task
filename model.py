from typing import Tuple

import torch
from torch import nn
from torch.autograd import Variable

from modules.embedding import Embedding
from modules.cnn_encoder import CNNEncoder
from modules.rnn_base_module import RNNBaseModule
from modules.highway import Highway
from modules.attention_flow import AttentionFlow
from modules.linear import MultiDimLinear
from modules.utils import softmax, masked_softmax, weighted_sum,\
                          masked_log_softmax, replace_masked_values,\
                          relu


class Bidaf(nn.Module):
    def __init__(self,
                 char_dict_size: int,
                 word_dict_size: int,
                 char_embedding_dim: int,
                 word_embedding_dim: int,
                 char_filter_dim: int,
                 n_gram_sizes: Tuple[int, ...],
                 hidden_dim: int,
                 word_emb: torch.FloatTensor = None,
                 dropout: float = .2,
                 bidirectional: bool = True,
                 sent_attention: bool = False):
        super(Bidaf, self).__init__()

        self.sent_attention = sent_attention

        self.char_embedding = Embedding(char_dict_size, char_embedding_dim, trainable=False)
        self.word_embedding = Embedding(word_dict_size, word_embedding_dim, trainable=False, weight=word_emb)

        self.char_cnn_encoder = CNNEncoder(char_embedding_dim, char_filter_dim, dropout=dropout,
                                           ngram_filter_sizes=n_gram_sizes)

        self.highway = Highway(char_filter_dim + word_embedding_dim, num_layers=2)

        self.contextual_embedding = RNNBaseModule(char_filter_dim + word_embedding_dim, hidden_dim,
                                                  dropout=dropout, bidirectional=bidirectional)

        self.attention_flow = AttentionFlow(hidden_dim)
        attn_flow_output_dim = hidden_dim * 8

        if sent_attention:
            residual_dim = hidden_dim * 2

            self.start_residual_encoding = MultiDimLinear(attn_flow_output_dim, residual_dim, dropout=dropout, bias=True, bias_start=.0)
            self.sentence_encoder = RNNBaseModule(residual_dim, hidden_dim, dropout=dropout,
                                                  bidirectional=bidirectional)
            self.sent_attention_flow = AttentionFlow(hidden_dim)
            sent_attn_flow_output_dim = hidden_dim * 8
            self.end_residual_encoding = MultiDimLinear(sent_attn_flow_output_dim, residual_dim, dropout=dropout)
            model_layer_input_dim = residual_dim
        else:
            model_layer_input_dim = attn_flow_output_dim

        self.model_layer = RNNBaseModule(model_layer_input_dim, hidden_dim, num_layers=2,
                                         dropout=dropout, bidirectional=bidirectional)

        self.end_index_encoder_layer = RNNBaseModule(model_layer_input_dim + hidden_dim * 6, hidden_dim,
                                                     dropout=dropout, bidirectional=bidirectional)

        self.start_logits_linear = MultiDimLinear(model_layer_input_dim + hidden_dim * 2, 1, dropout=dropout, bias=True, bias_start=.0)
        self.end_logits_linear = MultiDimLinear(model_layer_input_dim + hidden_dim * 2, 1, dropout=dropout, bias=False, bias_start=.0)

    def forward(self,
                paragraph, query,
                char_paragraph=None,
                char_query=None,
                dm=None, qm=None,
                sent_start=None, sent_end=None,
                sent_mask=None):
        batch_size = paragraph.size(0)
        paragraph_length = paragraph.size(1)

        passage_encoded, _ = self.embedding_layer(paragraph, char_paragraph, batch_size, mask=dm)
        query_encoded, query_hidden_states = self.embedding_layer(query, char_query, batch_size, mask=qm)

        query2context, context2query = self.attention_flow(passage_encoded, query_encoded, dm=dm, qm=qm)
        attention_flow_vectors = torch.cat([passage_encoded, query2context,
                                            passage_encoded * query2context,
                                            passage_encoded * context2query], dim=-1)

        if self.sent_attention:
            sentence_encode_input = relu(self.start_residual_encoding(attention_flow_vectors))

            sent_represent_matrix = self.word_aligned_sentence_encoder_layer(sentence_encode_input,
                                                                             sent_start, sent_end,
                                                                             sent_mask,
                                                                             query_hidden_state=query_hidden_states)
            sent2context, context2sent = self.sent_attention_flow(sentence_encode_input,
                                                                  sent_represent_matrix, dm=dm, qm=sent_mask)
            sent_attention_flow_vectors = torch.cat([sentence_encode_input, sent2context,
                                                     sentence_encode_input * sent2context,
                                                     sentence_encode_input * context2sent], dim=-1)
            output_sent_attn_vectors = relu(self.end_residual_encoding(sent_attention_flow_vectors))
            model_layer_input = sentence_encode_input + output_sent_attn_vectors
        else:
            model_layer_input = attention_flow_vectors

        model_hidden_state = self.model_layer.init_hidden(batch_size)
        modeled_passage, _ = self.model_layer(model_layer_input, mask=dm, hidden_state=model_hidden_state)
        modeled_passage_dim = modeled_passage.size(-1)

        start_logits_input = torch.cat([model_layer_input, modeled_passage], dim=-1)
        start_logits = self.start_logits_linear(start_logits_input).squeeze(-1)

        start_probs = masked_softmax(start_logits, dm)
        span_start_representation = weighted_sum(modeled_passage, start_probs)
        tiled_start_representation = span_start_representation.unsqueeze(1).expand(batch_size,
                                                                                   paragraph_length,
                                                                                   modeled_passage_dim)

        encode_end_inputs = torch.cat([model_layer_input, modeled_passage, tiled_start_representation,
                                       modeled_passage * tiled_start_representation], dim=-1)

        encode_end_hidden_state = self.end_index_encoder_layer.init_hidden(batch_size)
        encoded_end_logits, _ = self.end_index_encoder_layer(encode_end_inputs, mask=dm, hidden_state=encode_end_hidden_state)

        end_logits_input = torch.cat([model_layer_input, encoded_end_logits], dim=-1)
        end_logits = self.end_logits_linear(end_logits_input).squeeze(-1)
        end_probs = masked_softmax(end_logits, dm)
        start_log = masked_log_softmax(start_logits, dm)
        end_log = masked_log_softmax(end_logits, dm)
        start_logits = replace_masked_values(start_logits, dm, -1e7)
        end_logits = replace_masked_values(end_logits, dm, -1e7)

        output_dict = {"span_start_logits": start_logits,
                       "span_start_probs": start_probs,
                       "span_end_logits": end_logits,
                       "span_end_probs": end_probs,
                       "span_start_log": start_log,
                       "span_end_log": end_log}

        return output_dict

    def embedding_layer(self, words, chars, batch_size, mask=None):
        char_embedded = self.char_embedding(chars)
        char_embedded = self.char_cnn_encoder(char_embedded)

        word_embedded = self.word_embedding(words)

        word_embedded = self.highway(torch.cat([char_embedded, word_embedded], dim=-1))

        init_hidden_state = self.contextual_embedding.init_hidden(batch_size)
        context_embedded, hidden_state = self.contextual_embedding(word_embedded, mask=mask, hidden_state=init_hidden_state)

        return context_embedded, hidden_state

    def word_aligned_sentence_encoder_layer(self, input, sent_start, sent_end, sent_mask, query_hidden_state=None):
        batch_size = input.size(0)
        feature_dim = input.size(-1)
        max_sent_len = torch.max(sent_mask.sum(dim=-1)).cpu().data.tolist()[0]

        batch_matrix_list = list()
        for i in range(batch_size):
            sent_len = sent_mask[i].sum().cpu().data.tolist()[0]
            sent_vectors = list()
            for s_i in range(sent_len):
                if query_hidden_state is None:
                    initial_hidden_vector = self.sentence_encoder.init_hidden(1)
                else:
                    initial_hidden_vector = query_hidden_state[:, i, :].unsqueeze(1).contiguous()
                sent_encode_input = input[i, sent_start[i][s_i]:sent_end[i][s_i], :].unsqueeze(0)
                sent_encoded, _ = self.sentence_encoder(
                    sent_encode_input, hidden_state=initial_hidden_vector)
                word_attention = softmax(torch.mul(sent_encoded, sent_encode_input).sum(dim=-1))
                sent_vector = weighted_sum(sent_encoded, word_attention)

                sent_vectors.append(sent_vector)

            pad_len = max_sent_len - sent_len
            if pad_len > 0:
                sent_vectors.append(Variable(torch.zeros(pad_len, feature_dim).cuda()))
            sent_matrix = torch.cat(sent_vectors, dim=0)
            batch_matrix_list.append(sent_matrix)

        sent_represent_encoded = torch.stack(batch_matrix_list)

        return sent_represent_encoded
