import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import *
import numpy  as np
import itertools

hidden_embed = 10


class RelaLayer(nn.Module):
    def __init__(self, hidden_embed):  # cut words by sentences

        '''
        :param doc_h: encoded hidden vectors for doc B X D X H
        :param qry_h: encoded hidden vectors for qry B X Q X H
        :param doc: embedded hidden vectors for doc words B X D X H
        :param qry: encoded hidden vectors for qry B X Q X H
        :param sent_start: Sentence start indices
        '''

        super(RelaLayer, self).__init__()
        self.hidden_embed = hidden_embed
        self.r_fc1s = nn.Linear(hidden_embed * 3, hidden_embed)
        self.r_fc2s = nn.Linear(hidden_embed, hidden_embed)
        self.r_fc3s = nn.Linear(hidden_embed, hidden_embed)
        self.relus = nn.ReLU()

        self.r_fc1w = nn.Linear(hidden_embed * 3, hidden_embed)
        self.r_fc2w = nn.Linear(hidden_embed, hidden_embed)
        self.r_fc3w = nn.Linear(hidden_embed, hidden_embed)
        self.reluw = nn.ReLU()

    def forward(self, doc_h, qry_h, doc, qry, sent_start, sent_end):
        #batch_size = doc_h.size(0)
        #doc_len = doc_h.size(1)
        num_sentence = len(sent_start)

        ave_qry_h = torch.mean(qry_h, 1) / qry_h.size(1)


        doc_h_by_sentence_ave = []
        word_h_by_sentence = []

        for i in range(num_sentence):

            doc_h_by_sentence = doc_h[:, sent_start[i]:sent_end[i]+1, :]
            print(doc_h)


            doc_h_by_sentence_ave.append(torch.sum(doc_h_by_sentence, 1)/doc_h_by_sentence.size(1))  # sent_id X B X D X H

            result_list=[]

        for f, b in itertools.permutations(doc_h_by_sentence_ave, 2):
            feed = torch.cat([f, b, ave_qry_h],1)

                 #64 x 450
            feed = self.r_fc1s(feed)
            feed = self.r_fc2s(feed)
            output = self.relus(feed)

            result_list.append(output)

            #print(torch.stack(result_list).size())

        tmps = torch.stack(result_list)
        context_vector = tmps.sum(0)/tmps.size(0)


        for i in range(num_sentence):
            sentence_now = doc_h[:, (sent_start[i]):(sent_end[i] + 1), :]
            num_of_w_in_sent = sentence_now.size(1)
            for j in range( num_of_w_in_sent):
                word_h_by_sentence.append(sentence_now[:,j,:])


            for f, b in itertools.permutations(word_h_by_sentence, 2):
                feed = torch.cat([f, b, ave_qry_h], 1)
                feed = self.r_fc1s(feed)
                feed = self.r_fc2s(feed)
                output = self.relus(feed)  # 64 x 1 x 150

                result_list.append(output)

            tmpw = torch.stack(result_list)
            sentence_vector = tmpw.sum(0)/tmpw.size(0)
              #  print(sentence_vector.size(), context_vector.size())


            sent_context_vector = (sentence_vector + context_vector) /2 # 64 x 1 x 150
            sent_context_vector = sent_context_vector.unsqueeze(1).repeat(1, num_of_w_in_sent, 1)
            doc[:, (sent_start[i]):(sent_end[i] + 1), :] = (doc_h[:,(sent_start[i]):(sent_end[i] + 1),:] + sent_context_vector)/2
        return doc # word_embed + sentence_vec + context_vec




''' 
doc_h = Variable(torch.randn(64,150,hidden_embed ))
qry_h = Variable(torch.randn(64, 7, hidden_embed ))
doc = Variable(torch.randn(64, 150, hidden_embed ))
qry = Variable(torch.randn(64, 7, hidden_embed ))
#hey = torch.round(torch.rand(16)*150).sort()
#sent_start = hey[0:-1]
#sent_end = hey[1:]
#print(sent_start, sent_end)
sent_start = [0, 24, 48, 60, 90, 110, 130]
sent_end = [23, 47, 59, 89, 109, 129, 149]
relayer = RelaLayer(hidden_embed )
doc_h = relayer(doc_h, qry_h, doc, qry, sent_start, sent_end)

print("test**")
print(doc_h)
'''