import pickle
from collections import defaultdict
import itertools
import operator
from torch.utils.data import DataLoader, Dataset
import torch
from functools import reduce
from itertools import accumulate
import random
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel

PATH = {}
PATH['wq'] = '../data/baseline/KBQA_RE_data/webqsp_relations/MODE_data.txt'
#PATH['wq'] = '../data/baseline/KBQA_RE_data/webqsp_relations/WebQSP.RE.MODE.with_boundary.withpool.dlnlp.txt'
PATH['sq'] = '../data/baseline/KBQA_RE_data/sq_relations/MODE.replace_ne.withpool'
for i in [1,2,3]:
    PATH[f'pq{i}'] = f'../data/PQ/baseline/PQ{i}/MODE_data1.txt'
    PATH[f'pql{i}'] = f'../data/PQ/baseline/PQL{i}/MODE_data1.txt'
for i in range(11):
    PATH[f'wpq{i}'] = f'../data/PQ/exp3/baseline/{i}/MODE_data.txt'
PATH['exp4'] = '../data/PQ/exp4/baseline/MODE_data.txt'

def quick_collate(batch):
    return batch[0]

def random_split(dataset, train_p, valid_p):
    random.shuffle(dataset.data_objs)
    return Subset(dataset.data_objs[:int(len(dataset)*train_p)]), Subset(dataset.data_objs[int(len(dataset)*train_p):]) 

class Subset(Dataset):
    def __init__(self, data_objs):
        self.data_objs = data_objs
    def __getitem__(self, idx):
        return self.data_objs[idx]
    def __len__(self):
        return len(self.data_objs)

class PerQuestionDataset(Dataset):
    def __init__(self, args, mode, word2id, rela2id, rela_token2id):
        super(PerQuestionDataset, self).__init__()
#        self.bert_tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert)
        self.data_objs = self._get_data(args, mode, word2id, rela2id, rela_token2id)
    def _get_data(self, args, mode, word2id, rela2id, rela_token2id):
        # get candidate's bert representation now
#        bert_encoder = BertModel.from_pretrained(args.pretrained_bert, cache_dir="../../.pretrained_bert")
#        bert_encoder = bert_encoder.cuda()

        data_objs = []
        id2rela = {v: k for k, v in rela2id.items()}
        file_path = PATH[args.dataset.lower()]
        print(file_path)
        with open(file_path.replace('MODE', mode), 'r') as f:
            for i, line in enumerate(f):
                print(f'\rreading line {i}', end='')
                anses, candidates, question = line.strip().split('\t')
                # modify because of 'noNegativeAnswer' in sq data
                candidates = [x for x in candidates.split() if x not in anses.split() and x!='noNegativeAnswer']
                # modify for "#head_entity#" label in sq dataset
                ques = question.replace('$ARG1', '').replace('$ARG2', '').replace('<e>', 'TOPIC_ENTITY').replace('#head_entity#', 'TOPIC_ENTITY').strip()
                for ans in anses.split():
                    data = self._numericalize((i, ques, ans, candidates), word2id, rela_token2id)
                    #data = self._bert_numericalize((i, ques, ans, candidates), word2id, rela_token2id)
                    data_objs.append(data)
                    #
                    #index, tokens, masks, segments = self._bert_padding(data)
                    #with torch.no_grad():
                    #    _, pooled_layer = bert_encoder(tokens, segments, masks)
                    #    pooled_layer = pooled_layer.detach().cpu()
                    #    data_objs.append((index, ques, pooled_layer[0], pooled_layer[1:]))
                    #
        return data_objs
    def _bert_padding(self, data):
        index, ques, ans, candidates = data[0], data[1], data[2], data[3]
        relas = [ans]+candidates
        maxlen = max([len(x) for x in relas])
        tokens, masks, segments = [], [], []
        for rela in relas:
            token = ques + rela + (maxlen-len(rela))*[0]
            mask = [1]*(len(ques)+len(rela)) + [0]*(maxlen-len(rela))
            segment = [0]*len(ques) + [1]*maxlen
            tokens.append(token)
            masks.append(mask)
            segments.append(segment)
        tokens = torch.LongTensor(tokens).cuda()
        masks = torch.LongTensor(masks).cuda()
        segments = torch.LongTensor(segments).cuda()
        return index, tokens, masks, segments

    def _bert_tokenize(self, seq, is_rela=False):
        if is_rela:
            seq = seq.replace('_', '.').replace('.', ' ').strip()
        seq = self.bert_tokenizer.tokenize(seq)
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        return seq
    def _bert_numericalize(self, data, word2id, rela2id):
        index, ques, ans, candidates = data[0], data[1], data[2], data[3]
        ques = self._bert_tokenize(ques)
        candidates = [self._bert_tokenize(x, is_rela=True) for x in candidates if x!=ans]
        ans = self._bert_tokenize(ans, is_rela=True)
        return index, ques, ans, candidates
    def _numericalize(self, data, word2id, rela2id):
        index, ques, ans, candidates = data[0], data[1], data[2], data[3]
        ques = self._numericalize_str(ques, word2id, [' '])
        #ques = self._bert_tokenize(ques)
        candidates = [(self._numericalize_str(x, rela2id, ['.']), 
            self._numericalize_str(x, word2id, ['.', '_'])) for x in candidates if x != ans]
        ans = (self._numericalize_str(ans, rela2id, ['.']), self._numericalize_str(ans, word2id, ['.', '_']))
        return index, ques, ans, candidates
    def _numericalize_str(self, string, map2id, dilemeter):
        strings = string.split('.')
        string = ''
        for i, s in enumerate(strings):
            if i%3 == 0 and i > 0:
                string += '..'
            string += s
        if len(dilemeter) == 2:
            string = string.replace(dilemeter[1], dilemeter[0])
        dilemeter = dilemeter[0]
        tokens = string.strip().split(dilemeter)
        tokens = [map2id[x] if x in map2id else map2id['<unk>'] for x in tokens]
        return tokens
    def __len__(self):
        return len(self.data_objs)
    def __getitem__(self, index):
        return self.data_objs[index]
