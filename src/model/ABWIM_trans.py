import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
from model.transformer import Transformer

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
        # CNN layer
        self.cnn_1 = nn.Conv1d(args.hidden_size*4, args.num_filters, 1)
        self.cnn_2 = nn.Conv1d(args.hidden_size*4, args.num_filters, 3)
        self.cnn_3 = nn.Conv1d(args.hidden_size*4, args.num_filters, 5)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(args.num_filters, 1, bias=False)

        self.transformer = Transformer(
            n_src_vocab=args.rela_vocab_size+args.word_embedding.shape[0], len_max_seq=100,
            d_word_vec=args.emb_size, d_model=args.hidden_size*2, d_inner=512, n_layers=6, 
            n_head=4, d_k=args.hidden_size//2, d_v=args.hidden_size//2, dropout=args.dropout_rate)

    def forward(self, *inputs):
        question, word_relation, rela_relation = inputs[0], inputs[1], inputs[2]
        ques_pos, rela_pos = inputs[3], inputs[4]

        ques_embedding = self.word_embedding(question)
        ques_embedding = self.dropout(ques_embedding)
        rela_embedding = self.rela_embedding(rela_relation)
        rela_embedding = self.dropout(rela_embedding)
        word_embedding = self.word_embedding(word_relation)
        word_embedding = self.dropout(word_embedding)

        question_out = self.transformer(question, ques_embedding, ques_pos)
        question_out = torch.transpose(question_out, 1, 2)

        embedding = torch.cat([word_embedding, rela_embedding], 1)
        relation = torch.cat([word_relation, rela_relation], 1)
        relation = self.transformer(relation, embedding, rela_pos)

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
        return self.sigmoid(score)

