import spacy

from nltk.tokenize import sent_tokenize

import json
from tqdm import tqdm
import logging
from collections import Counter
import pickle

logger = logging.getLogger(__name__)


class SpacyWordSplitter(object):
    _spacy_tokenizers = {}

    def __init__(self,
                 language='en_core_web_sm',
                 pos_tags=False,
                 parse=False,
                 ner=False):

        self.spacy = self._get_spacy_model(language, pos_tags, parse, ner)

    def split_words(self, sentence: str):
        return [t for t in self.spacy(sentence) if not t.is_space]

    def _get_spacy_model(self, spacy_model_name, pos_tags, parse, ner):
        options = (spacy_model_name, pos_tags, parse, ner)
        if options not in self._spacy_tokenizers:

            kwargs = {'vectors': False}
            if not pos_tags:
                kwargs['tagger'] = False
            if not parse:
                kwargs['parser'] = False
            if not ner:
                kwargs['entity'] = False

            spacy_model = spacy.load(spacy_model_name, **kwargs)
            self._spacy_tokenizers[options] = spacy_model
        return self._spacy_tokenizers[options]


class SpacySentSplitter(object):
    def __init__(self, word_tokenize=False):
        self.sent_tokenizer = spacy.load("en")
        self.word_tokenize = word_tokenize

        if self.word_tokenize:
            self.word_tokenizer = WordTokenizer()

    def split_sentences(self, paragraph):

        paragraph = self.sent_tokenizer(paragraph)
        sentences = [str(sent) for sent in paragraph.sents]

        if self.word_tokenize:
            sentences = [self.word_tokenizer.tokenize(sent) for sent in sentences]
        return sentences


class SentTokenizer(object):
    def __init__(self, word_tokenize=False):
        self._setn_splitter = SpacySentSplitter(word_tokenize=word_tokenize)

    def tokenize(self, text):
        sents = self._setn_splitter.split_sentences(text)
        return sents


class WordTokenizer(object):
    def __init__(self):
        self._word_splitter = SpacyWordSplitter()

    def tokenize(self, text):
        words = self._word_splitter.split_words(text)
        return words


class SquadReader(object):
    def __init__(self):
        self._tokenizer = WordTokenizer()
        self._sent_tokenizer = SentTokenizer(word_tokenize=True)
        self.cnt_err = 0
        # self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def read(self, file_path, lower=False):
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        instances = []

        for article in tqdm(dataset):
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"]

                sent_tokenized_paragraph = self._sent_tokenizer.tokenize(paragraph)
                tokenized_paragraph = [str(single_token) for line in sent_tokenized_paragraph for single_token in line]
                sent_token_pos = get_sent_token_pos(sent_tokenized_paragraph)

                for question_answer in paragraph_json['qas']:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    answer_texts = [answer['text'] for answer in question_answer['answers']]
                    span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                    span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]

                    instance = self.text_to_instance(question_text,
                                                     paragraph,
                                                     answer_texts,
                                                     char_spans=zip(span_starts, span_ends),
                                                     passage_tokens=tokenized_paragraph,
                                                     sentence_token_pos=sent_token_pos,
                                                     lower=lower)
                    instances.append(instance)

        if not instances:
            raise AttributeError("No instances were read from the given filepath {}. "
                                 "Is the path correct?".format(file_path))
        return Dataset(instances)

    def text_to_instance(self,  # type: ignore
                         question_text,
                         passage_text,
                         answer_texts,
                         sentence_token_pos=None,
                         char_spans=None,
                         passage_tokens=None,
                         lower=False):

        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        char_spans = char_spans or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.

        token_spans = []
        passage_offsets = []

        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), passage_offset = get_word_span(passage_text, passage_tokens, char_span_start,
                                                                   char_span_end)

            token_spans.append((span_start, span_end))
            passage_offsets.append((passage_offset))

        candidate_answers = Counter()
        for span_start, span_end in token_spans:
            candidate_answers[(span_start, span_end)] += 1
        span_start, span_end = candidate_answers.most_common(1)[0][0]

        question_tokens = self._tokenizer.tokenize(question_text)

        instance = {}
        if lower:
            instance["query"] = [token.text.lower() for token in question_tokens]
            instance["passage"] = [token.lower() for token in passage_tokens]
        else:
            instance["query"] = [token.text for token in question_tokens]
            instance["passage"] = [token for token in passage_tokens]

        instance["sentence_token_pos"] = sentence_token_pos
        instance["original_passage"] = passage_text
        instance["answer_texts"] = answer_texts
        instance["token_offset"] = passage_offsets
        instance["span_start"] = span_start
        instance["span_end"] = span_end

        return instance


class Dataset:
    def __init__(self, instances=None):
        self.instances = instances

    def truncate(self, max_instances):
        if len(self.instances) > max_instances:
            self.instances = self.instances[:max_instances]

    def save_dataset(self, file_path):
        if self.instances is not None:
            with open(file_path, 'w') as outfile:
                data = {"data": self.instances}
                json.dump(data, outfile)

    def load_dataset(self, file_path):
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            self.instances = dataset_json['data']


def get_word_span(context, wordss, start, stop):
    spans = get_word(context, wordss)  # first character index and end index for each word
    idxs = []
    for word_idx, span in enumerate(spans):
        if not (stop <= span[0] or start >= span[1]):
            idxs.append(word_idx)

    assert len(idxs) > 0, "{} {} {} {}".format(context, spans, start, stop)
    return (idxs[0], idxs[-1]), spans


def get_word(text, tokens):
    spans = []
    cur_idx = 0
    for token in tokens:
        if text.find(token, cur_idx) < 0:  # fine token chracter idx in text starting from cur_idx
            print(tokens)
            print("{} {} {}".format(token, cur_idx, text))
            raise Exception()
        cur_idx = text.find(token, cur_idx)
        spans.append((cur_idx, cur_idx + len(token)))
        cur_idx += len(token)

    return spans  # the index based on the character


def get_sent_token_pos(sent_tokenized_paragraph):
    # This is a function which returns token position in a paragraph, and position numbers are set
    # to slice the token array. So, when a word tokenized paragraph array which splitted by sentences,
    # this function returns a tuple array which represents start and end token indicies of a sentence.
    # For example, when you input a paragraph array with two sentences like, [[The, Broncos, took, ...... , trailed, .],
    # [Newton, was, limited, ..... , touchdown, .]], this array returns an array with tuples like, [(0, 14),(14, 44)].
    # The (0, 14) means that 0 is a start index and 14 is the an end index + 1. The reason 'an end index + 1' is
    # we need to use this index not to access individual elements, slice an array from start index to end index.
    # So far, when you slicing an word tokenized array like, "word_tokenized_array[0:14]", then you can get a complete
    # tokenized sentence array.
    sent_token_pos = []

    sentence_end, sentence_start = 0, 0
    for sent in sent_tokenized_paragraph:
        sentence_start, sentence_end = sentence_end, sentence_end + len(sent)
        sent_token_pos.append((sentence_start, sentence_end))

    return sent_token_pos


'''
path = "/home/tgchoi/Projects/QA_Project/data/squad/"
files = ["train-v1.1.json", "dev-v1.1.json"]

sr = SquadReader()
train_dataset = sr.read(path + files[0], lower=False)
dev_dataset = sr.read(path + files[1], lower=False)

dev_dataset.save_dataset(file_path="./data/squad/unlower_dev_dataset.json")
train_dataset.save_dataset(file_path="./data/squad/unlower_train_dataset.json")
'''