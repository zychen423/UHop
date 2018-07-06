import torch as th
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np
import math



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.word_embedding = nn.Embedding(args.word_embedding.shape[0], args.word_embedding.shape[1])
        self.word_embedding.weight = nn.Parameter(th.from_numpy(args.word_embedding).float())
        self.word_embedding.weight.requires_grad = False
        self.rela_embedding = nn.Embedding(args.rela_vocab_size, args.word_embedding.shape[1])
        nn.init.xavier_normal(self.rela_embedding.weight)
        self.rnn = nn.LSTM(input_size=args.emb_size, hidden_size=args.hidden_size,
                          num_layers=1, batch_first=False,
                          dropout=args.dropout_rate, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=args.hidden_size*2, hidden_size=args.hidden_size,
                          num_layers=1, batch_first=False,
                          dropout=args.dropout_rate, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.args = args
        self.cos = nn.CosineSimilarity(dim=1)
        self.tanh = nn.Tanh()
        return

    def forward(self, *inputs):
        ques_x, rela_text_x, rela_x = inputs[0], inputs[1], inputs[2]
        ques_x = th.transpose(ques_x, 0, 1)
        rela_text_x = th.transpose(rela_text_x, 0, 1)
        rela_x = th.transpose(rela_x, 0, 1)

       
        #print(ques_x)
        #print(rela_text_x)
        #print(rela_x)
        ques_x = self.word_embedding(ques_x)
        rela_text_x = self.word_embedding(rela_text_x)
        rela_x = self.rela_embedding(rela_x)
        ques_x = self.dropout(ques_x)
        rela_text_x = self.dropout(rela_text_x)
        rela_x = self.dropout(rela_x)
        

        ques_hs1, hidden_state = self.encode(ques_x)
        rela_hs, hidden_state = self.encode(rela_x, hidden_state)
        #print(index)
        #print(hidden_state_c.shape)
        rela_text_hs, hidden_state = self.encode(rela_text_x, hidden_state)

        #print('ques_hs1', ques_hs1)
        ques_hs2, _ = self.rnn2(ques_hs1) 
        ques_hs2 = self.dropout(ques_hs2)

        ques_hs = ques_hs1 + ques_hs2
        ques_hs = ques_hs.permute(1, 2, 0)
        ques_h = F.avg_pool1d(ques_hs, kernel_size=ques_hs.shape[2], stride=None)
        rela_hs = th.cat([rela_hs, rela_text_hs], 0)
        rela_hs = rela_hs.permute(1, 2, 0)
        rela_h = F.avg_pool1d(rela_hs, kernel_size=rela_hs.shape[2], stride=None)

        ques_h = ques_h.squeeze(2)
        rela_h = rela_h.squeeze(2)

        output = self.cos(ques_h, rela_h)
        return output

    def encode(self, input, hidden_state=None, return_sequence=True):
        if hidden_state==None:
            outputs, (h_output, c_output) = self.rnn(input)
        else:
            h_0, c_0 = hidden_state
            outputs, (h_output, c_output) = self.rnn(input, (h_0, c_0))
        #print('outputs', outputs)
        #print('outputs', outputs)
        outputs = self.dropout(outputs)
        h_output = self.dropout(h_output)
        c_output = self.dropout(c_output)
        if return_sequence == False:
            return outputs[-1], (h_output, c_output)
        else:
            return outputs, (h_output, c_output)
    

