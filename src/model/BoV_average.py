import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np
import math



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.word_embedding = nn.Embedding(args.word_embedding.shape[0], args.word_embedding.shape[1])
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(args.word_embedding).float())
        if args.train_embedding == True:
            self.word_embedding.weight.requires_grad = True
        else:
            self.word_embedding.weight.requires_grad = False
        self.rela_embedding = nn.Embedding(args.rela_vocab_size, args.word_embedding.shape[1])
        nn.init.xavier_normal(self.rela_embedding.weight)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.args = args
        self.cos = nn.CosineSimilarity(dim=1)
        return

    def forward(self, *inputs):
        ques_x, rela_text_x, rela_x = inputs[0], inputs[1], inputs[2]
        ques_x = torch.transpose(ques_x, 0, 1)
        rela_text_x = torch.transpose(rela_text_x, 0, 1)
        rela_x = torch.transpose(rela_x, 0, 1)

       
        #print(ques_x)
        #print(rela_text_x)
        #print(rela_x)
        ques_x = self.word_embedding(ques_x)
        rela_text_x = self.word_embedding(rela_text_x)
        rela_x = self.rela_embedding(rela_x)
        ques_x = self.dropout(ques_x)
        rela_text_x = self.dropout(rela_text_x)
        rela_x = self.dropout(rela_x)

        ques = torch.mean(ques_x, 0)
        rela_text = torch.mean(rela_text_x, 0)
        rela_x = torch.mean(rela_x, 0)
        rela = rela_text + rela_x
        score = self.cos(ques, rela)
        return score



