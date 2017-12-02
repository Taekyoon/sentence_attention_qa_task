import argparse
import time
from tqdm import tqdm
from typing import Tuple

import logging
import logging.handlers

from torch import cuda, optim, nn

from utils import *
from model import Bidaf
from allen_squad import Dataset
from boolean_accuracy import BooleanAccuracy
from squad_em_and_f1 import SquadEmAndF1
from loader import Vocabulary, _read_pretrained_embedding_file, BatchLoader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='./data/squad/')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--dev_batch_size', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--char_emb_dim', type=int, default=16)
    parser.add_argument('--word_emb_dim', type=int, default=100)
    parser.add_argument('--char_filter_dim', type=int, default=100)
    parser.add_argument('--n_gram_sizes', type=Tuple[int, ...], default=[5])
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--use_glove', type=bool, default=True)
    parser.add_argument('--train_emb', default=False)
    parser.add_argument('--dropout', default=0.2)
    parser.add_argument('--bidirectional', default=True)

    parser.add_argument('--log', default='original')
    parser.add_argument('--gpu_device', default=1)
    parser.add_argument('--size', default='')
    parser.add_argument('--glove', default='../data/glove/glove.6B.100d.txt')
    parser.add_argument('--decay_rate', default=0.999)
    parser.add_argument('--checkpoint', default=200)
    parser.add_argument('--cuda_set', default=True)
    parser.add_argument('--seed', default=133)
    parser.add_argument('--log_output', default=True)
    parser.add_argument('--max_passage_len', default=None)
    parser.add_argument('--log_file_name', default="sent_attention_batch_15_add_query_hidden")
    parser.add_argument('--sent_attention', default=True)

    args = parser.parse_args()
    cuda.set_device(args.gpu_device)
    cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    train_dataset, dev_dataset = Dataset(), Dataset()
    filename = ['lower_train_dataset.json', 'lower_dev_dataset.json']

    train_dataset.load_dataset(args.data + filename[0])
    dev_dataset.load_dataset(args.data + filename[1])

    vocab = Vocabulary(file_path=args.data + "lower_vocabulary.json")

    char_vocab = vocab.create_char_vocabulary(min_counts=7)
    char_vocab.prepare_dataset()

    pre_trained_embedding = _read_pretrained_embedding_file(args.glove, args.word_emb_dim, vocab)
    pre_trained_embedding = pre_trained_embedding.cuda()

    print("Preparing batch loader...")
    print("============= Train ===============")
    train_loader = BatchLoader(train_dataset, vocab, char_vocab=char_vocab, max_passage_len=args.max_passage_len,
                               cuda=args.cuda_set, batch_size=args.batch_size)
    print("============= Valid ===============")
    dev_loader = BatchLoader(dev_dataset, vocab, char_vocab=char_vocab, max_passage_len=args.max_passage_len,
                             cuda=args.cuda_set, batch_size=args.dev_batch_size)

    train_loader.save_global_sent_len_dist("train")
    dev_loader.save_global_sent_len_dist("dev")

    vocab_size, char_size = vocab.get_vocab_size(), char_vocab.get_vocab_size()

    model = Bidaf(char_size,
                  vocab_size,
                  args.char_emb_dim,
                  args.word_emb_dim,
                  args.char_filter_dim,
                  args.n_gram_sizes,
                  args.hidden_dim,
                  dropout=args.dropout,
                  word_emb=pre_trained_embedding,
                  sent_attention=args.sent_attention).cuda()

    criterion = nn.NLLLoss().cuda()
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=[0.9, 0.9])
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    avg_params = flatten_params(model)

    print("#" * 15, "Model Info", "#" * 15)
    print("Model: ", model)
    print("Criterion: ", criterion)
    print("Optimizer: ", optimizer)
    print("")

    _logger = logging.getLogger('mylogger')
    file_name = args.log_file_name
    fileHandler = logging.FileHandler('./' + file_name + '.log')
    streamHandler = logging.StreamHandler()
    _logger.addHandler(fileHandler)
    _logger.addHandler(streamHandler)
    _logger.setLevel(logging.DEBUG)

    _logger.info(args)

    print("\n===================       Start Train         ======================")
    for epoch in range(args.epoch):
        '''
        if epoch > 2:
            lr = args.lr * 0.5
            for param_group in optimizer.param_groups:
                print("learning rate change")
                param_group['lr'] = lr
        '''

        train_loss = AverageMeter()
        start_acc = AverageMeter()
        end_acc = AverageMeter()

        span_acc = BooleanAccuracy()
        squad_em_f1 = SquadEmAndF1()
        start_time = time.time()

        for i, data in enumerate(tqdm(train_loader)):
            model.train()

            passage, query = data["passage"], data["query"]
            char_passage, char_query = data["char_passage"], data["char_query"]
            passage_mask, query_mask = torch.clamp(passage, 0, 1).long(), torch.clamp(query, 0, 1).long()
            sent_start, sent_end, sent_mask = data["sent_start"], data["sent_end"], data["sent_mask"]

            optimizer.zero_grad()
            output = model(passage, query,
                           char_paragraph=char_passage, char_query=char_query,
                           dm=passage_mask, qm=query_mask,
                           sent_start=sent_start, sent_end=sent_end, sent_mask=sent_mask)

            loss1 = criterion(output["span_start_log"], data["span_start"])
            loss2 = criterion(output["span_end_log"], data["span_end"])
            loss = loss1 + loss2

            _, val_pred1 = output["span_start_probs"].data.cpu().topk(1)
            _, val_pred2 = output["span_end_probs"].data.cpu().topk(1)

            batch_size = output["span_start_probs"].size(0)

            best_spans = get_best_span(output["span_start_logits"], output["span_end_logits"])
            span_acc(best_spans.data, torch.stack([data["span_start"], data["span_end"]], -1).data)
            best_predict_strings = get_best_predict_string(batch_size, best_spans, data["original_passage"],
                                                           data["token_offset"])
            for b_i in range(batch_size):
                squad_em_f1(best_predict_strings[b_i], data["answer_texts"][b_i])

            start_acc.update(index_accuracy(val_pred1.numpy(), tensor2np(data["span_start"])))
            end_acc.update(index_accuracy(val_pred2.numpy(), tensor2np(data["span_end"])))

            loss.backward()
            optimizer.step()

            avg_params = exp_moving_avg(model, avg_params, decay_rate=args.decay_rate)

            train_loss.update(loss.data[0])

            if i % args.checkpoint == 0 and i != 0:
                em, f1 = squad_em_f1.get_metric(reset=False)
                _span_acc = span_acc.get_metric(reset=False)
                print("")
                message = "Train epoch: %d  iter: %d  start_acc: %1.2f end_acc: %1.2f train_loss: %1.2f span_acc: %1.2f em: %1.2f f1: %1.2f elapsed: %1.2f " \
                          % (epoch + 1, i, start_acc.avg, end_acc.avg, train_loss.avg, _span_acc,
                             em, f1, time.time() - start_time)
                _logger.info(message)
                print("")

        em, f1 = squad_em_f1.get_metric(reset=False)
        _span_acc = span_acc.get_metric(reset=False)
        print("")
        message = "Train epoch: %d  iter: %d  start_acc: %1.2f end_acc: %1.2f train_loss: %1.2f span_acc: %1.2f em: %1.2f f1: %1.2f elapsed: %1.2f " \
                  % (epoch + 1, i, start_acc.avg, end_acc.avg, train_loss.avg, _span_acc,
                     em, f1, time.time() - start_time)
        _logger.info(message)
        print("")

        print("\n" + str(epoch + 1) + "Epoch Done!")
        print("\n====================      Evaluation     ======================")
        model.eval()

        val_loss = AverageMeter()
        val_start_acc = AverageMeter()
        val_end_acc = AverageMeter()

        val_span_acc = BooleanAccuracy()
        val_squad_em_f1 = SquadEmAndF1()

        original_param = flatten_params(model)
        model = load_params(model, avg_params)

        for j, data in enumerate(tqdm(dev_loader)):
            passage, query = data["passage"], data["query"]
            char_passage, char_query = data["char_passage"], data["char_query"]
            passage_mask, query_mask = torch.clamp(passage, 0, 1).long(), torch.clamp(query, 0, 1).long()
            sent_start, sent_end, sent_mask = data["sent_start"], data["sent_end"], data["sent_mask"]

            dev_output = model(passage, query,
                               char_paragraph=char_passage, char_query=char_query,
                               dm=passage_mask, qm=query_mask,
                               sent_start=sent_start, sent_end=sent_end, sent_mask=sent_mask)

            _, val_pred1 = dev_output["span_start_probs"].data.cpu().topk(1)
            _, val_pred2 = dev_output["span_end_probs"].data.cpu().topk(1)

            loss1 = criterion(dev_output["span_start_log"], data["span_start"])
            loss2 = criterion(dev_output["span_end_log"], data["span_end"])
            loss = loss1 + loss2
            val_loss.update(loss.data[0])

            batch_size = dev_output["span_start_probs"].size(0)
            ground_truths = torch.stack([data["span_start"], data["span_end"]], -1)
            best_spans = get_best_span(dev_output["span_start_logits"], dev_output["span_end_logits"])
            val_span_acc(best_spans.data, ground_truths.data)
            best_predict_strings = get_best_predict_string(batch_size, best_spans, data["original_passage"],
                                                           data["token_offset"])

            em_match = []
            for b_i in range(batch_size):
                em_match.append(val_squad_em_f1(best_predict_strings[b_i], data["answer_texts"][b_i]))

            val_start_acc.update(index_accuracy(val_pred1.numpy(), tensor2np(data["span_start"])))
            val_end_acc.update(index_accuracy(val_pred2.numpy(), tensor2np(data["span_end"])))

        model = load_params(model, original_param)
        em, f1 = val_squad_em_f1.get_metric(reset=True)
        _span_acc = val_span_acc.get_metric(reset=True)

        print("")
        message = "val_start: %1.5f val_end: %1.5f val loss: %1.5f span acc: %1.5f em: %1.5f f1: %1.5f" % \
                  (val_start_acc.avg, val_end_acc.avg, val_loss.avg, _span_acc, em, f1)
        _logger.info(message)
        print("")

if __name__ == '__main__':
    main()
