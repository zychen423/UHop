import sys
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np
import math
import data_utility
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
sys.setrecursionlimit(50000)
class FeatureExtractionModel(nn.Module):
    def __init__(self, text_emb, rela_emb, bilstm, W, cnn1, cnn2, cnn3):
        super(FeatureExtractionModel, self).__init__()
        self.word_embedding = text_emb
        self.rela_embedding = rela_emb
        # LSTM layer
        self.bilstm = bilstm
        # Attention
        self.W = W
        # CNN layer
        self.cnn_1 = cnn1
        self.cnn_2 = cnn2
        self.cnn_3 = cnn3
        self.activation = nn.ReLU()

    def forward(self, question, word_relation, rela_relation):
        question = th.transpose(question, 0, 1)
        rela_relation = th.transpose(rela_relation, 0, 1)
        word_relation = th.transpose(word_relation, 0, 1)

        question = self.word_embedding(question)
        #question = self.dropout(question)
        rela_relation = self.rela_embedding(rela_relation)
        #rela_relation = self.dropout(rela_relation)
        word_relation = self.word_embedding(word_relation)
        #word_relation = self.dropout(word_relation)
#        self.bilstm.flatten_parameters()
        question_out, _ = self.bilstm(question)
        question_out = question_out.permute(1,2,0)
        #question_out = self.dropout(question_out)
        word_relation_out, word_relation_hidden = self.bilstm(word_relation)
        rela_relation_out, _ = self.bilstm(rela_relation, word_relation_hidden)
        #word_relation_out = self.dropout(word_relation_out)
        #rela_relation_out = self.dropout(rela_relation_out)
        relation = th.cat([rela_relation_out, word_relation_out], 0)
        relation = relation.permute(1,0,2)
        #print('relation', relation)
        # attention layer
        energy = th.matmul(relation, self.W)
        #print('energy', energy)
        energy = th.matmul(energy, question_out)
        #print('energy', energy.shape)
        energy_tmp = energy.view(energy.shape[0], energy.shape[1]*energy.shape[2])
        alpha = F.softmax(energy_tmp, dim=-1)
        alpha = alpha.view(energy.shape[0], energy.shape[1], energy.shape[2]) 
        return alpha

def extract_relas(tuples):
    pos_relas = []
    neg_relas = []
    for t in tuples:
        if t[2] == 1:
            pos_relas.append('.'.join(t[1] + [t[0]]))
        elif t[2] == 0:
            neg_relas.append('.'.join(t[1] + [t[0]]))
        else:
            print('error'); exit()
    return pos_relas, neg_relas

def to_variable(l):
    v = Variable(th.from_numpy(np.array(l))).cuda()
    return v

import sys
model_path = sys.argv[1]
rela_tokenizer_path = sys.argv[2]
model = th.load(model_path)

layer_list = list(model.children())
print(layer_list)


text_emb = layer_list[1]
rela_emb = layer_list[2]
bilstm = layer_list[3]
cnn1 = layer_list[4]
cnn2 = layer_list[5]
cnn3 = layer_list[6]
W = model.W

feature_extractor = FeatureExtractionModel(text_emb, rela_emb, bilstm, W, cnn1, cnn2, cnn3)
feature_extractor = feature_extractor.eval()
with open('tokenizer/ques_tokneizer_300.pkl', 'rb') as f:
    ques_tokenizer = pickle.load(f)
with open(rela_tokenizer_path, 'rb') as f:
    rela_tokenizer = pickle.load(f)
datas = data_utility.load_test_data()
for index, (i, ques_text, step_list) in enumerate(datas):
    if index != 337:
        continue
    for rela_len in range(0, len(step_list)-1):
        try:
            print('\r', index, end='')
            pos_relas, neg_relas = extract_relas(step_list[rela_len])
            t_ques_text = ques_tokenizer.process([ques_text])
            t_pos_rela_text, t_pos_rela = rela_tokenizer.process(pos_relas)
            #print(t_pos_rela)
            #print(t_pos_rela_text)
            v_ques_text = to_variable(t_ques_text)
            v_pos_rela = to_variable(t_pos_rela)
            v_pos_rela_text = to_variable(t_pos_rela_text)
            #print(v_ques_text)
            #print(v_pos_rela_text)
            #print(v_pos_rela)
            A = feature_extractor(v_ques_text, v_pos_rela_text, v_pos_rela)
            #print(ques_text, pos_relas)
            print(A)
            A = A.cpu().detach().numpy()
            rela_texts = rela_tokenizer.split(pos_relas[0]) 
            relas = rela_tokenizer.split_rela(pos_relas[0]) 
            ques_tokens = ques_text.split()
            print(ques_tokens)
            print(rela_texts)
            print(relas)
            if len(ques_tokens) < 5:
                ques_tokens = ['<PAD>'] * (5 - len(ques_tokens)) + ques_tokens
            df = pd.DataFrame( A[0])
            #columns = ques_tokens, 
            #                           index = rela_texts + relas)
            ax = sns.heatmap( data= df, linewidth=0.1, cmap='Reds', annot=False, 
                    square=True, cbar=False)
            #ax.set_title('TITLE HERE')
            #ax.set_xlabel('x_label')
            #ax.set_ylabel('y_label')
            #ax.xaxis.tick_top()
            ax.xaxis.set_ticks_position('top')


            # labels rotation
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            #ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            plt.savefig(f'./{index}_{rela_len}.jpg', dpi=100)
        except ValueError:
            continue

