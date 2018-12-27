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
        print(self.method)

    def reduce_prev(self, rela_previous, word_previous, question_out):
        rela_previous = torch.transpose(rela_previous, 0, 1)
        word_previous = torch.transpose(word_previous, 0, 1)
        rela_previous = self.rela_embedding(rela_previous)
        rela_previous = self.dropout(rela_previous)
        word_previous = self.word_embedding(word_previous)
        word_previous = self.dropout(word_previous)
        word_previous_out, word_previous_hidden = self.bilstm(word_previous)
        rela_previous_out, _ = self.bilstm(rela_previous, word_previous_hidden)
        word_previous_out = self.dropout(word_previous_out)
        rela_previous_out = self.dropout(rela_previous_out)
        previous = torch.cat([rela_previous_out, word_previous_out], 0)
        previous = previous.permute(1,0,2)
        energy = torch.matmul(previous, self.W)
        energy = torch.matmul(energy, question_out)
        tmp_energy = energy.view(energy.shape[0], energy.shape[1]*energy.shape[2])
        tmp_energy = tmp_energy / math.sqrt(energy.shape[1])
        alpha = F.softmax(tmp_energy, dim=-1)
        alpha = alpha.view(energy.shape[0], energy.shape[1], energy.shape[2])
        alpha = alpha.unsqueeze(3)
        previous = previous.unsqueeze(2)
        atten_previous = alpha * previous
        atten_previous = torch.sum(atten_previous, 1)
        atten_previous = atten_previous.permute(0, 2, 1)
        if self.method == 'bilstm':
            question_out = question_out - atten_previous
            question_out = question_out.permute(0, 2, 1)
            question_out, _ = self.ques_bilstm(question_out)
            question_out = question_out.permute(0, 2, 1)
        elif self.method == 'bilstm2':
            atten_previous = atten_previous.permute(0, 2, 1)
            atten_previous, _ = self.ques_bilstm(atten_previous)
            atten_previous = atten_previous.permute(0, 2, 1)
            question_out = question_out - atten_previous
        elif self.method == 'dense_relu':
            question_out = question_out - atten_previous
            question_out = question_out.permute(0, 2, 1)
            question_out = question_out.reshape(-1, atten_previous.shape[1])
            question_out = self.activation(self.linear2(question_out))
            question_out = question_out.reshape(atten_previous.shape[0], -1,
                                                                atten_previous.shape[1])
            question_out = question_out.permute(0, 2, 1)
        elif self.method == 'sub':
            if rela_previous.shape[0]==0:
                print('GG')
            question_out = question_out - atten_previous
        else:
            question_out = question_out - atten_previous
            question_out = question_out.permute(0, 2, 1)
            question_out = question_out.reshape(-1, atten_previous.shape[1])
            question_out = self.linear2(question_out)
            question_out = question_out.reshape(atten_previous.shape[0], -1,
                                                                atten_previous.shape[1])
            question_out = question_out.permute(0, 2, 1)
        return question_out

    def forward(self, *inputs):
        question, word_relation, rela_relation = inputs[0], inputs[1], inputs[2]
        word_previous, rela_previous = inputs[3], inputs[4]

        question = torch.transpose(question, 0, 1)
        rela_relation = torch.transpose(rela_relation, 0, 1)
        word_relation = torch.transpose(word_relation, 0, 1)

        question = self.word_embedding(question)
        question = self.dropout(question)
        rela_relation = self.rela_embedding(rela_relation)
        rela_relation = self.dropout(rela_relation)
        word_relation = self.word_embedding(word_relation)
        word_relation = self.dropout(word_relation)

#        self.bilstm.flatten_parameters()
        question_out, _ = self.bilstm(question)
        question_out = question_out.permute(1,2,0)
        question_out = self.dropout(question_out)
        if rela_previous.shape[0]>0:
            question_out = self.reduce_prev(rela_previous, word_previous, question_out)
#            non_pad_rela = rela_previous.ne(0)
#            non_pad_word = word_previous.ne(0)
#            non_pad_mask = torch.cat([non_pad_rela, non_pad_word], 1).unsqueeze(-1).float()
#            rela_previous = self.rela_embedding(rela_previous)
#            word_previous = self.word_embedding(word_previous)
#            previous = torch.cat([rela_previous, word_previous], 1)
#            previous = torch.sum(non_pad_mask*previous, 1)
#            previous /= torch.sum(non_pad_mask, 1)
#            previous = previous.unsqueeze(1)
#            previous = self.prev_linear(previous)
#            previous = self.dropout(previous)
#
#            energy = torch.matmul(previous, self.W2)
#            energy = torch.matmul(energy, question_out)
#            #energy = torch.matmul(previous, question_out)
#            tmp_energy = energy.view(energy.shape[0], energy.shape[1]*energy.shape[2])
#            tmp_energy = tmp_energy / math.sqrt(energy.shape[1])
#            if self.method=='log_minus':
#                print(1)
#                alpha = F.log_softmax(tmp_energy, dim=-1)
#                alpha = 1-alpha
#                alpha = F.softmax(alpha, dim=-1)
#            elif self.method=='softmin':
#                alpha = F.softmin(tmp_energy, dim=-1)
#            elif self.method=='one_minus':
#                alpha = F.softmax(tmp_energy, dim=-1)
#                alpha = 1-alpha
#                alpha = F.softmax(alpha, dim=-1)
#            elif self.method=='sum_normal':
#                alpha = F.softmax(tmp_energy, dim=-1)
#                alpha = 1-alpha
#                alpha = alpha/torch.sum(alpha, dim=-1, keepdim=True)
#            alpha = alpha.view(energy.shape[0], energy.shape[1], energy.shape[2])
#            alpha = alpha.unsqueeze(3)
#            previous = previous.unsqueeze(2)
#            atten_previous = alpha * previous
#            atten_previous = torch.sum(atten_previous, 1)
#            question_out = atten_previous.permute(0, 2, 1)

        word_relation_out, word_relation_hidden = self.bilstm(word_relation)
        rela_relation_out, _ = self.bilstm(rela_relation, word_relation_hidden)
        word_relation_out = self.dropout(word_relation_out)
        rela_relation_out = self.dropout(rela_relation_out)
        relation = torch.cat([rela_relation_out, word_relation_out], 0)
        relation = relation.permute(1,0,2)

        # attention layer
        #energy_tmp = energy.view(energy.shape[0], energy.shape[1]*energy.shape[2])
        #alpha = F.softmax(energy_tmp, dim=-1)
        #alpha = alpha.view(energy.shape[0], energy.shape[1], energy.shape[2])
        energy = torch.matmul(relation, self.W)
        energy = torch.matmul(energy, question_out)
        tmp_energy = energy.view(energy.shape[0], energy.shape[1]*energy.shape[2])
        tmp_energy = tmp_energy / math.sqrt(energy.shape[1])
        alpha = F.softmax(tmp_energy, dim=-1)
        alpha = alpha.view(energy.shape[0], energy.shape[1], energy.shape[2])
        alpha = alpha.unsqueeze(3)
        relation = relation.unsqueeze(2)
        atten_relation = alpha * relation
        atten_relation = torch.sum(atten_relation, 1)
        atten_relation = atten_relation.permute(0, 2, 1)
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

