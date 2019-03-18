import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
#from pytorch_pretrained_bert import BertModel

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # Word Embedding layer
        self.word_embedding = nn.Embedding(args.word_embedding.shape[0], args.word_embedding.shape[1])
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(args.word_embedding).float())
        self.word_embedding.weight.requires_grad = False # fix the embedding matrix
        # Rela Embedding layer
        self.rela_embedding = nn.Embedding(args.rela_vocab_size, args.word_embedding.shape[1])
        nn.init.xavier_normal(self.rela_embedding.weight)
        # LSTM layer
        self.bilstm = nn.LSTM(args.emb_size, args.hidden_size, num_layers=1, 
                              bidirectional=True, batch_first=False, dropout=args.dropout_rate)
        self.dropout = nn.Dropout(args.dropout_rate)
        # Attention
        q_hidden_size = 768 if args.q_representation == "bert" else args.hidden_size*2
        W = torch.empty(args.hidden_size*2, q_hidden_size)
        nn.init.xavier_normal(W)
        self.W = nn.Parameter(W)
        # CNN layer
        self.cnn_1 = nn.Conv1d(q_hidden_size+args.hidden_size*2, args.num_filters, 1)
        self.cnn_2 = nn.Conv1d(q_hidden_size+args.hidden_size*2, args.num_filters, 3)
        self.cnn_3 = nn.Conv1d(q_hidden_size+args.hidden_size*2, args.num_filters, 5)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(args.num_filters, 1, bias=False)
#        self.bert = BertModel.from_pretrained(args.pretrained_bert, cache_dir="../../.pretrained_bert")
#        for param in self.bert.parameters():
#            param.requires_grad = False
        self.q_representation = args.q_representation

    def forward(self, *inputs):
        #question, word_relation, rela_relation, word_prev, rela_prev = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        question, ques_mask, word_relation, rela_relation = inputs[0], inputs[1], inputs[2], inputs[3]
        rela_relation = torch.transpose(rela_relation, 0, 1)
        word_relation = torch.transpose(word_relation, 0, 1)

        if(self.q_representation=="bert"):
            question = torch.squeeze(question, 1)
            ques_mask = torch.squeeze(ques_mask, 1)
            ques_segment = torch.zeros_like(question, dtype=torch.long)
            question, pooled_question = self.bert(question, ques_segment, ques_mask, output_all_encoded_layers=False)
            question_out = torch.transpose(question, 1, 2)
        else:
            question = torch.transpose(question, 0, 1)
            question = self.word_embedding(question)
            question = self.dropout(question)
            question_out, _ = self.bilstm(question)
            question_out = question_out.permute(1,2,0)
            question_out = self.dropout(question_out)

        rela_relation = self.rela_embedding(rela_relation)
        rela_relation = self.dropout(rela_relation)
        word_relation = self.word_embedding(word_relation)
        word_relation = self.dropout(word_relation)
#        self.bilstm.flatten_parameters()
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
        return self.sigmoid(score)

