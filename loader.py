import random
from tqdm import tqdm
import logging
import json
import pickle

import numpy as np

import torch
from torch.autograd import Variable

logger = logging.getLogger(__name__)

DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"


def np2tensor(item, cuda=True):
    if cuda:
        tensor = Variable(torch.from_numpy(item).cuda())
    else:
        tensor = Variable(torch.from_numpy(item))
    return tensor


class CharacterVocabulary:
    def __init__(self, char_counter, max_char_length, min_counts=1):
        self.padding_token = DEFAULT_PADDING_TOKEN
        self.oov_token = DEFAULT_OOV_TOKEN
        self.char_counter = char_counter
        self.token_to_index = {self.padding_token: 0, self.oov_token: 1}
        self.index_to_token = {0: self.padding_token, 1: self.oov_token}
        self.word_indexer = 2
        self.min_counts = min_counts
        self.max_char_length = max_char_length

    def prepare_dataset(self):
        for item in tqdm(self.char_counter):
            if self.char_counter[item] >= self.min_counts:
                self.token_to_index[item] = self.word_indexer
                self.index_to_token[self.word_indexer] = item
                self.word_indexer += 1

    def get_index_from_token(self, token):
        if token not in self.token_to_index:
            return self.token_to_index[self.oov_token]
        return self.token_to_index[token]

    def get_token_from_index(self, index):
        if str(index) not in self.index_to_token:
            return self.index_to_token["1"]
        return self.index_to_token[str(index)]

    def get_vocab_size(self):
        return self.word_indexer

    def get_max_char_len(self):
        return self.max_char_length


class Vocabulary:
    def __init__(self, min_counts=1, file_path=None, glove_path=None):
        self.padding_token = DEFAULT_PADDING_TOKEN
        self.oov_token = DEFAULT_OOV_TOKEN
        self.glove_vocabs = list()
        self.word_counter = dict()
        self.token_to_index = {self.padding_token: 0, self.oov_token: 1}
        self.index_to_token = {0: self.padding_token, 1: self.oov_token}
        self.word_indexer = 2
        self.min_counts = min_counts
        self.use_glove = False

        if file_path is not None:
            self.load_vocab(file_path)

        if glove_path is not None:
            self.use_glove = True
            self.get_glove_vocabs(glove_path)

    def get_glove_vocabs(self, glove_path):
        logger.info("Reading embeddings from file")
        with open(glove_path, 'r', encoding='utf-8') as embeddings_file:
            for line in tqdm(embeddings_file):
                fields = line.strip().split(' ')
                word = fields[0]
                self.glove_vocabs.append(str(word))

    def inject_dataset(self, dataset):
        logging.info("counting words")
        for instance in tqdm(dataset.instances):
            self.count_words(instance['passage'])
            self.count_words(instance['query'])

        logging.info("indexing words")
        self.reset_dict()
        for item in tqdm(self.word_counter):
            if self.use_glove and item not in self.glove_vocabs:
                continue
            if self.word_counter[item] >= self.min_counts:
                self.token_to_index[item] = self.word_indexer
                self.index_to_token[self.word_indexer] = item
                self.word_indexer += 1

    def reset_dict(self):
        self.token_to_index = {self.padding_token: 0, self.oov_token: 1}
        self.index_to_token = {0: self.padding_token, 1: self.oov_token}
        self.word_indexer = 2

    def save_vocab(self, file_path):
        if not self.word_indexer == 0:
            data = {"token_to_index": self.token_to_index,
                    "index_to_token": self.index_to_token,
                    "word_counter": self.word_counter,
                    "min_counts": self.min_counts,
                    "word_indexer": self.word_indexer}

            with open(file_path, 'w') as outfile:
                json.dump(data, outfile)

    def load_vocab(self, file_path):
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            self.token_to_index = dataset_json["token_to_index"]
            self.index_to_token = dataset_json["index_to_token"]
            self.word_counter = dataset_json["word_counter"]
            self.word_indexer = dataset_json["word_indexer"]
            self.min_counts = dataset_json["min_counts"]

    def create_char_vocabulary(self, min_counts=1):
        char_counter = dict()
        max_char_len = -1

        for word in self.word_counter:
            if max_char_len < len(word):
                max_char_len = len(word)
            for ch in word:
                if ch not in char_counter:
                    char_counter[ch] = 1
                else:
                    char_counter[ch] += 1

        return CharacterVocabulary(char_counter, max_char_len, min_counts=min_counts)

    def count_words(self, words):
        for word in words:
            if word not in self.word_counter:
                self.word_counter[word] = 1
            else:
                self.word_counter[word] += 1

    def get_index_from_token(self, token):
        if token not in self.token_to_index:
            return self.token_to_index[self.oov_token]
        return self.token_to_index[token]

    def get_token_from_index(self, index):
        if str(index) not in self.index_to_token:
            return self.index_to_token["1"]
        return self.index_to_token[str(index)]

    def get_vocab_size(self):
        return self.word_indexer


class BatchLoader(object):
    def __init__(self, dataset, vocab, char_vocab, max_passage_len=None, shuffle=False,
                 bucketing=True, cuda=True, batch_size=30):
        self.dataset = dataset.instances
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.batch_index_pool = []

        self.max_passage_len = max_passage_len
        self.batch_size = batch_size
        self.current_position = 0
        self.shuffle = shuffle
        self.cuda = cuda

        if bucketing:
            self.bucketing()

        if self.max_passage_len:
            self.passage_len_limit()

        self.set_batch()
        print("Total number of data: ", len(self.dataset))
        print("===================================")
        print("")

    def set_batch(self):
        self.current_position = 0
        instance_indices = [i for i in range(len(self.dataset))]

        if self.shuffle:
            random.shuffle(self.instance_indices)

        batch_iter = (len(self.dataset) // self.batch_size)
        self.batch_index_pool = [instance_indices[i * self.batch_size:(i + 1) * self.batch_size]
                                 for i in range(batch_iter)]
        last_batch = instance_indices[batch_iter * self.batch_size:]
        if len(last_batch) > 0:
            self.batch_index_pool.append(last_batch)

    def passage_len_limit(self):
        filtered_dataset = []
        for instance in tqdm(self.dataset):
            if len(instance["passage"]) <= self.max_passage_len:
                filtered_dataset.append(instance)
        print("Dataset filtered ", len(filtered_dataset), "/", len(self.dataset))
        print("Dataset Changed!")
        self.dataset = filtered_dataset

    def bucketing(self):
        self.dataset.sort(key=lambda x: len(x["passage"]))

    def get_global_max_sent_len(self):
        return max([len(instance["sentence_token_pos"]) for instance in self.dataset])

    def get_global_sent_len_dist(self):
        sent_len_histogram = dict()
        for instance in self.dataset:
            sent_len = len(instance["sentence_token_pos"])
            if sent_len in sent_len_histogram:
                sent_len_histogram[sent_len] += 1
            else:
                sent_len_histogram[sent_len] = 1
        return sent_len_histogram

    def save_target_instance(self, instance, file_path="./temp.pkl"):
        f = open(file_path, "wb")
        pickle.dump(instance, f)
        f.close()

    def save_global_sent_len_dist(self, name):
        histogram = self.get_global_sent_len_dist()
        self.save_target_instance(histogram, file_path="./" + name + "_sentence_length_histogram.pkl")

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.current_position == len(self.batch_index_pool):
            print("#" * 30)
            print("End Iteration")
            self.set_batch()
            raise StopIteration()

        indicies = self.batch_index_pool[self.current_position]
        self.current_position += 1
        try:
            batch_size = len(indicies)
            max_passage_len = max([len(self.dataset[idx]["passage"]) for idx in indicies])
            max_query_len = max([len(self.dataset[idx]["query"]) for idx in indicies])
            max_sent_num = max([len(self.dataset[idx]["sentence_token_pos"]) for idx in indicies])
            max_word_char_len = self.char_vocab.get_max_char_len()

        except:
            print("error!!")
            print("current pos: ", self.current_position, "length batch pool: ", len(self.batch_index_pool))

        passage = np.zeros((batch_size, max_passage_len), dtype='int64')
        query = np.zeros((batch_size, max_query_len), dtype='int64')
        sent_start = np.zeros((batch_size, max_sent_num), dtype='int64')
        sent_end = np.zeros((batch_size, max_sent_num), dtype='int64')
        sent_mask = np.zeros((batch_size, max_sent_num), dtype='int64')
        char_passage = np.zeros((batch_size, max_passage_len, max_word_char_len), dtype='int64')
        char_query = np.zeros((batch_size, max_query_len, max_word_char_len), dtype='int64')
        span_start, span_end = [], []
        original_passage, token_offset, answer_texts = [], [], []

        for i, idx in enumerate(indicies):
            instance = self.dataset[idx]

            passage_input = [self.vocab.get_index_from_token(word) for word in instance["passage"]]
            query_input = [self.vocab.get_index_from_token(word) for word in instance["query"]]
            char_passage_tokennized = [[self.char_vocab.get_index_from_token(char) for char in word]
                                       for word in instance["passage"]]
            char_query_tokenized = [[self.char_vocab.get_index_from_token(char) for char in word]
                                    for word in instance["query"]]

            sent_start_input = [sent_pos[0] for sent_pos in instance["sentence_token_pos"]]
            sent_end_input = [sent_pos[1] for sent_pos in instance["sentence_token_pos"]]

            char_passage_input = list()
            for word_char in char_passage_tokennized:
                temp_char_passage_np = np.zeros((max_word_char_len))
                temp_char_passage_np[:len(word_char)] = np.array(word_char)
                char_passage_input.append(temp_char_passage_np)
            char_passage_input = np.stack(char_passage_input)

            char_query_input = list()
            for word_char in char_query_tokenized:
                temp_char_query_np = np.zeros((max_word_char_len))
                temp_char_query_np[:len(word_char)] = np.array(word_char)
                char_query_input.append(temp_char_query_np)
            char_query_input = np.stack(char_query_input)

            passage[i, :len(passage_input)] = np.array(passage_input)
            query[i, :len(query_input)] = np.array(query_input)

            sent_start[i, :len(sent_start_input)] = np.array(sent_start_input)
            sent_end[i, :len(sent_end_input)] = np.array(sent_end_input)
            sent_mask[i, :len(sent_start_input)] = 1

            char_passage[i, :len(passage_input), :] = char_passage_input
            char_query[i, :len(query_input), :] = char_query_input

            span_start.append(instance["span_start"])
            span_end.append(instance["span_end"])

            original_passage.append(instance["original_passage"])
            token_offset.append(instance["token_offset"])
            answer_texts.append(instance["answer_texts"])

        span_start = np.array(span_start)
        span_end = np.array(span_end)

        # print (sent_start)
        # print (sent_end)

        batch_dict = {"passage": np2tensor(passage, self.cuda),
                      "query": np2tensor(query, self.cuda),
                      "char_passage": np2tensor(char_passage, self.cuda),
                      "char_query": np2tensor(char_query, self.cuda),
                      "sent_start": sent_start,
                      "sent_end": sent_end,
                      "sent_mask": np2tensor(sent_mask, self.cuda),
                      "span_start": np2tensor(span_start, self.cuda),
                      "span_end": np2tensor(span_end, self.cuda),
                      "original_passage": original_passage,
                      "token_offset": token_offset,
                      "answer_texts": answer_texts}

        return batch_dict


def _read_pretrained_embedding_file(embeddings_filename,
                                    embedding_dim,
                                    vocab):
    """
    Reads a pre-trained embedding file and generates an Embedding layer that has weights
    initialized to the pre-trained embeddings.  The Embedding layer can either be trainable or
    not.

    We use the ``Vocabulary`` to map from the word strings in the embeddings file to the indices
    that we need, and to know which words from the embeddings file we can safely ignore.

    Parameters
    ----------
    embeddings_filename : str, required.
        The path to a file containing pretrined embeddings. The embeddings
        file is assumed to be gzipped and space delimited, e.g. [word] [dim 1] [dim 2] ...
    vocab : Vocabulary, required.
        A Vocabulary object.
    namespace : str, (optional, default=tokens)
        The namespace of the vocabulary to find pretrained embeddings for.
    trainable : bool, (optional, default=True)
        Whether or not the embedding parameters should be optimized.

    Returns
    -------
    A weight matrix with embeddings initialized from the read file.  The matrix has shape
    ``(vocab.get_vocab_size(namespace), embedding_dim)``, where the indices of words appearing in
    the pretrained embedding file are initialized to the pretrained embedding value.
    """
    words_to_keep = vocab.token_to_index
    vocab_size = vocab.get_vocab_size()
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logger.info("Reading embeddings from file")
    with open(embeddings_filename, 'r', encoding='utf-8') as embeddings_file:
        # with gzip.open(embeddings_filename, 'rb') as embeddings_file:
        for line in embeddings_file:
            fields = line.strip().split(' ')
            if len(fields) - 1 != embedding_dim:
                # Sometimes there are funny unicode parsing problems that lead to different
                # fields lengths (e.g., a word with a unicode space character that splits
                # into more than one column).  We skip those lines.  Note that if you have
                # some kind of long header, this could result in all of your lines getting
                # skipped.  It's hard to check for that here; you just have to look in the
                # embedding_misses_file and at the model summary to make sure things look
                # like they are supposed to.
                logger.warning("Found line with wrong number of dimensions (expected %d, was %d): %s",
                               embedding_dim, len(fields) - 1, line)
                continue
            word = fields[0]
            if word in words_to_keep:
                vector = np.asarray(fields[1:], dtype='float32')
                embeddings[word] = vector

    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    logger.info("Initializing pre-trained embedding layer")
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean,
                                                                            embeddings_std)

    for i in range(0, vocab_size):
        word = vocab.get_token_from_index(i)

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if word in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[word])
        else:
            logger.debug("Word %s was not found in the embedding file. Initialising randomly.", word)

    # The weight matrix is initialized, so we construct and return the actual Embedding.
    return embedding_matrix


'''
from allen_squad import Dataset


train_dataset, dev_dataset = Dataset(), Dataset()
path = './data/squad/'
filename = ['unlower_train_dataset.json', 'unlower_dev_dataset.json']

train_dataset.load_dataset(path + filename[0])
dev_dataset.load_dataset(path + filename[1])

vocab = Vocabulary()

vocab.inject_dataset(train_dataset)
print(vocab.word_indexer)
vocab.inject_dataset(dev_dataset)
print(vocab.word_indexer)

vocab.save_vocab(path + 'unlower_vocabulary.json')

char_vocab = vocab.create_char_vocabulary(min_counts=7)
char_vocab.prepare_dataset()
print(char_vocab.get_vocab_size())
print(char_vocab.get_max_char_len())
'''
