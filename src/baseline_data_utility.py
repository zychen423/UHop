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

PATH = {}
PATH['wq'] = '../data/baseline/KBQA_RE_data/webqsp_relations/WebQSP.RE.MODE.with_boundary.withpool.dlnlp.txt'
PATH['sq'] = '../data/baseline/KBQA_RE_data/sq_relations/MODE.replace_ne.withpool'
for i in [1,2,3]:
    PATH[f'pq{i}'] = f'../data/PQ/PQ{i}'
    PATH[f'pql{i}'] = f'../data/PQ/PQL{i}'
for i in range(11):
    PATH[f'wpq{i}'] = f'../data/PQ/exp3/baseline/{i}/MODE_data.txt'
#    PATH[f'wpq{i}'] = f'../data/PQ/WPQL/baseline/{i}/MODE_data.txt'
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
        self.data_objs = self._get_data(args, mode, word2id, rela2id, rela_token2id)
    def _get_data(self, args, mode, word2id, rela2id, rela_token2id):
        data_objs = []
        id2rela = {v: k for k, v in rela2id.items()}
#        baseline_path, UHop_path = path_finder.path_finder()
#        wpq_path = path_finder.WPQ_PATH()
#        if 'wpq' in args.dataset:
#            file_path = wpq_path.baseline[args.dataset]
#        el
#        if 'pq' in args.dataset and 'w' not in args.dataset:
#            file_path = baseline_path.data(args.dataset)
#        else:
        file_path = PATH[args.dataset]
        print(file_path)
        with open(file_path.replace('MODE', mode), 'r') as f:
            for i, line in enumerate(f):
                print(f'\rreading line {i}', end='')
                anses, candidates, question = line.strip().split('\t')
                # modify because of 'noNegativeAnswer' in sq data
                candidates = [x for x in candidates.split() if x not in anses.split() and x!='noNegativeAnswer']
                #candidates = [id2rela[int(x)].replace('..', '.') for x in candidates.split() if x not in anses.split() and x!='noNegativeAnswer']
                # modify for "#head_entity#" label in sq dataset
                ques = question.replace('$ARG1', '').replace('$ARG2', '').replace('<e>', 'TOPIC_ENTITY').replace('#head_entity#', 'TOPIC_ENTITY').strip()
                for ans in anses.split():
                    #ans = id2rela[int(ans)].replace('..', '.')
                    data = self._numericalize((i, ques, ans, candidates), word2id, rela_token2id)
                    data_objs.append(data)
                if i==0:
                    print(ans)
        return data_objs
    def _numericalize(self, data, word2id, rela2id):
        index, ques, ans, candidates = data[0], data[1], data[2], data[3]
        ques = self._numericalize_str(ques, word2id, [' '])
        candidates = [(self._numericalize_str(x, rela2id, ['.']), 
            self._numericalize_str(x, word2id, ['.', '_'])) for x in candidates if x != ans]
        ans = (self._numericalize_str(ans, rela2id, ['.']), self._numericalize_str(ans, word2id, ['.', '_']))
        return index, ques, ans, candidates
    def _numericalize_str(self, string, map2id, dilemeter):
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
