import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # Word Embedding layer
        self.word_embedding = nn.Embedding(args.word_embedding.shape[0], args.word_embedding.shape[1])
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(args.word_embedding).float())
        self.word_embedding.weight.requires_grad = False # fix the embedding matrix
        # Rela Embedding layer
        self.rela_embedding = nn.Embedding(args.rela_vocab_size, args.word_embedding.shape[1])
        nn.init.xavier_normal_(self.rela_embedding.weight)
        # LSTM layer
        self.bilstm = nn.LSTM(args.emb_size, args.hidden_size, num_layers=1, 
                              bidirectional=True, batch_first=False, dropout=args.dropout_rate)
        self.dropout = nn.Dropout(args.dropout_rate)
        # Attention
        W = torch.empty(args.hidden_size*2, args.hidden_size*2)
        nn.init.xavier_normal_(W)
        self.W = nn.Parameter(W)
        W2 = torch.empty(args.hidden_size*2, args.hidden_size*2)
        nn.init.xavier_normal_(W2)
        self.W2 = nn.Parameter(W2)
        # CNN layer
        self.cnn_1 = nn.Conv1d(args.hidden_size*4, args.num_filters, 1)
        self.cnn_2 = nn.Conv1d(args.hidden_size*4, args.num_filters, 3)
        self.cnn_3 = nn.Conv1d(args.hidden_size*4, args.num_filters, 5)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(args.num_filters, 1, bias=False)
        self.linear2 = nn.Linear(args.hidden_size*2, args.hidden_size*2, bias=True)
        self.ques_bilstm = nn.LSTM(args.hidden_size*2, args.hidden_size, num_layers=1, 
                                bidirectional=True, batch_first=True, dropout=args.dropout_rate)
        self.method = args.reduce_method
        self.prev_linear = nn.Linear(args.emb_size, args.hidden_size*2)

    def reduce_prev(self, rela_previous, word_previous, question_out):
        atten_previous = self.abwim_atten(question_out, word_previous, rela_previous)
        question_out = question_out - atten_previous
        question_out = question_out.permute(0, 2, 1)
        question_out = question_out.reshape(-1, atten_previous.shape[1])
        question_out = self.linear2(question_out)
        question_out = question_out.reshape(atten_previous.shape[0], -1, atten_previous.shape[1])
        question_out = question_out.permute(0, 2, 1)
        return question_out

    def abwim_atten(self, question_out, word_relation, rela_relation, dump_atten=False):
        rela_relation = torch.transpose(rela_relation, 0, 1)
        word_relation = torch.transpose(word_relation, 0, 1)
        rela_relation = self.rela_embedding(rela_relation)
        rela_relation = self.dropout(rela_relation)
        word_relation = self.word_embedding(word_relation)
        word_relation = self.dropout(word_relation)
        word_relation_out, word_relation_hidden = self.bilstm(word_relation)
        rela_relation_out, _ = self.bilstm(rela_relation, word_relation_hidden)
        word_relation_out = self.dropout(word_relation_out)
        rela_relation_out = self.dropout(rela_relation_out)
        relation = torch.cat([rela_relation_out, word_relation_out], 0)
        relation = relation.permute(1,0,2) # b tr h
        # relation = [batch x timestep x hidden]
        # W = [hidden_p x hidden_q]
        # energy = [batch x timestep_p x hidden_q], question = [batch x hidden_q x timestep_q]
        energy = torch.matmul(relation, self.W)
        energy = torch.matmul(energy, question_out) # b tr tq
        tmp_energy = energy.view(energy.shape[0], energy.shape[1]*energy.shape[2])
        tmp_energy = tmp_energy / math.sqrt(energy.shape[1])
        alpha = F.softmax(tmp_energy, dim=-1)
        alpha = alpha.view(energy.shape[0], energy.shape[1], energy.shape[2])
        alpha = alpha.unsqueeze(3) # b tr tq 1
        relation = relation.unsqueeze(2) # b tr 1 h
        atten_relation = alpha * relation # b tr tq h => b (1) tq h
        atten_relation = torch.sum(atten_relation, 1)
        atten_relation = atten_relation.permute(0, 2, 1) # b h tq
        if dump_atten:
            return alpha[0].squeeze(), atten_relation
        return atten_relation

    def forward(self, *inputs):
        question, word_relation, rela_relation = inputs[0], inputs[1], inputs[2]
        word_previous, rela_previous = inputs[3], inputs[4]

        question = torch.transpose(question, 0, 1)
        question = self.word_embedding(question)
        question = self.dropout(question)
        question_out, _ = self.bilstm(question)
        question_out = question_out.permute(1,2,0)
        question_out = self.dropout(question_out)

#        self.bilstm.flatten_parameters()
        for word_prev, rela_prev in zip(word_previous, rela_previous):
            if rela_prev.shape[0]>0:
                question_out = self.reduce_prev(rela_prev, word_prev, question_out)
#                question_out = self.abwim_atten(question_out, word_prev, rela_prev)

        # attention layer
        #energy_tmp = energy.view(energy.shape[0], energy.shape[1]*energy.shape[2])
        #alpha = F.softmax(energy_tmp, dim=-1)
        #alpha = alpha.view(energy.shape[0], energy.shape[1], energy.shape[2])
        atten_relation = self.abwim_atten(question_out, word_relation, rela_relation)
        M = torch.cat((question_out, atten_relation), 1)
        h1 = F.max_pool1d(self.activation(self.cnn_1(M)), M.shape[2])
        h1 = self.dropout(h1)
        h2 = F.max_pool1d(self.activation(self.cnn_2(M)), M.shape[2]-2)
        h2 = self.dropout(h2)
        h3 = F.max_pool1d(self.activation(self.cnn_3(M)), M.shape[2]-4)
        h3 = self.dropout(h3)
        h = torch.cat((h1, h2, h3),2)
        h = torch.max(h, 2)[0]
        score = self.linear(h).squeeze()
        return score
